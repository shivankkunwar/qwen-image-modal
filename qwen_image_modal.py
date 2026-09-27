"""
Qwen-Image-2.1 on Modal (Starter plan, $30/month credits).

Two parts:
  * QwenImage (GPU, L40S) - loads the model once per container and generates.
  * api (small CPU container) - FastAPI app that validates requests and hands
    real work to the GPU, so typos, option lookups and status polling never
    wake the GPU. Interactive docs at <api url>/docs. See docs/api.md.

One-time setup:
    pip install -U modal pydantic
    modal setup                                   # browser login
    modal secret create huggingface HF_TOKEN=hf_xxx
    modal run qwen_image_modal.py::download_weights   # ~30 GB -> Volume, CPU only

Generate from the CLI (ephemeral app, scales to zero when done):
    modal run qwen_image_modal.py --prompt "a neon shop sign that reads SHVNK"
    modal run qwen_image_modal.py --prompt "cartoon dragon sticker" --transparent --n 4
    modal run qwen_image_modal.py --prompt "make it a rainy night" --images cafe.png
    modal run qwen_image_modal.py --prompts-file prompts.txt

Deploy the API (still scales to zero, billed only while busy):
    modal deploy qwen_image_modal.py
"""

import base64
import io
import random
import time
from typing import Literal

import modal
from pydantic import BaseModel, Field, model_validator

APP_NAME = "qwen-image-21"
MODEL_ID = "Qwen/Qwen-Image-2.1"
# QwenImage21Pipeline is only on diffusers main right now. Pin a commit that is
# known to have it (and the use_kv_cache arg) so a future main push can't break you.
DIFFUSERS_COMMIT = "6256aa7666cedd47443adc8f82da9a10e110b09c"
GPU = "L40S"  # 48 GB: bf16 model (~27-30 GB loaded) fits with headroom
CACHE_DIR = "/models"

# ---------------------------------------------------------------------------
# Limits and presets (shared by the API, the GPU worker and the CLI)
# ---------------------------------------------------------------------------
MAX_SIDE = 2752  # model's largest native side
MIN_SIDE = 256
MAX_REFERENCE_IMAGES = 10  # model limit for editing / composition
MAX_IMAGES_PER_REQUEST = 4
MAX_UPLOAD_BYTES = 20 * 1024 * 1024  # per reference image
MAX_UPLOAD_SIDE = 8192
RGBA_PREFIX = "This is an RGBA image with transparency. "

AspectRatio = Literal["1:1", "4:3", "3:4", "3:2", "2:3", "16:9", "9:16"]
# 2k = the model's native presets (model card); 1k = same shapes at ~1 megapixel.
SIZE_PRESETS = {
    "1k": {"1:1": (1024, 1024), "4:3": (1152, 864), "3:4": (864, 1152), "3:2": (1248, 832),
           "2:3": (832, 1248), "16:9": (1344, 768), "9:16": (768, 1344)},
    "2k": {"1:1": (2048, 2048), "4:3": (2400, 1792), "3:4": (1792, 2400), "3:2": (2528, 1696),
           "2:3": (1696, 2528), "16:9": (2752, 1536), "9:16": (1536, 2752)},
}
# Pipeline's output_resolution: output size when following a reference image's
# shape, and the size reference images are resized to.
OUTPUT_RESOLUTION = {"1k": 1024, "2k": 2048}
MIME_TYPES = {"png": "image/png", "webp": "image/webp", "jpeg": "image/jpeg"}
EXTENSIONS = {"png": "png", "webp": "webp", "jpeg": "jpg"}


class GenerateRequest(BaseModel):
    """Everything the model can do, in one request. Only `prompt` is required."""

    prompt: str = Field(min_length=1, max_length=4000, description="What to create, or the edit instruction when images are given.")
    images: list[str] = Field(
        default_factory=list, max_length=MAX_REFERENCE_IMAGES,
        description="Reference images (base64 or data URLs) for editing/composition. Order matters: "
                    "the prompt can say 'the first image', 'the second image'. Mark local edits by drawing on the image.",
    )
    size: Literal["1k", "2k"] = Field("1k", description="1k is ~1 megapixel (fast). 2k is native quality, ~4x slower.")
    aspect_ratio: AspectRatio | None = Field(
        None, description="Preset shape. Default: 1:1, or the first reference image's shape when editing.")
    width: int | None = Field(None, ge=MIN_SIDE, le=MAX_SIDE, multiple_of=32, description="Exact width; overrides size/aspect_ratio. Set with height.")
    height: int | None = Field(None, ge=MIN_SIDE, le=MAX_SIDE, multiple_of=32, description="Exact height. Set with width.")
    num_images: int = Field(1, ge=1, le=MAX_IMAGES_PER_REQUEST, description="Variations; image k uses seed + k.")
    steps: int = Field(40, ge=1, le=100, description="Denoising steps. 40 is Qwen's recommendation; fewer is faster but rougher.")
    seed: int | None = Field(None, ge=0, le=2**32 - 1, description="Same seed + same settings = same image. Random if omitted.")
    transparent: bool = Field(False, description="Transparent (RGBA) output. Adds Qwen's RGBA prompt prefix. Needs png or webp.")
    negative_prompt: str | None = Field(None, max_length=4000, description="What to avoid. Only used with guidance_scale > 1.")
    guidance_scale: float = Field(1.0, ge=1.0, le=10.0, description="true_cfg_scale. 1 = off (recommended). >1 enables negative_prompt and doubles generation time.")
    output_format: Literal["png", "webp", "jpeg"] = "png"
    quality: int = Field(95, ge=1, le=100, description="webp/jpeg quality. Ignored for png.")
    use_kv_cache: bool = Field(True, description="Faster. Changes pixels slightly, so keep it fixed when reproducing a seed.")

    @model_validator(mode="after")
    def _check_combinations(self):
        if (self.width is None) != (self.height is None):
            raise ValueError("set both width and height, or neither")
        if self.width and self.aspect_ratio:
            raise ValueError("use either width/height or aspect_ratio, not both")
        if self.transparent and self.output_format == "jpeg":
            raise ValueError("jpeg can't store transparency; use png or webp")
        if self.guidance_scale > 1 and not self.negative_prompt:
            raise ValueError("guidance_scale > 1 needs a negative_prompt")
        if self.negative_prompt and self.guidance_scale <= 1:
            raise ValueError("negative_prompt only works with guidance_scale > 1 (try 4.0; doubles generation time)")
        return self

    def target_size(self, has_references: bool) -> tuple[int, int] | None:
        """(width, height) to generate, or None to follow the first reference image's shape."""
        if self.width:
            return self.width, self.height
        if self.aspect_ratio:
            return SIZE_PRESETS[self.size][self.aspect_ratio]
        return None if has_references else SIZE_PRESETS[self.size]["1:1"]


# ---------------------------------------------------------------------------
# Container images
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.6",
        "torchvision",  # Qwen3-VL's processor (the text encoder) imports it
        "transformers>=5.17",
        "accelerate",
        "peft",
        "pillow",
        "pydantic>=2",
        "huggingface_hub[hf_xet]",
        f"git+https://github.com/huggingface/diffusers@{DIFFUSERS_COMMIT}",
    )
    .env({"HF_HUB_CACHE": CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
)
# The API container doesn't need torch, so it gets a tiny image that starts fast.
api_image = modal.Image.debian_slim(python_version="3.12").uv_pip_install(
    "fastapi[standard]", "pillow", "pydantic>=2"
)

# These imports only run inside the GPU container, not on your laptop or in the API.
with image.imports():
    import torch
    from diffusers import QwenImage21Pipeline

app = modal.App(APP_NAME, image=image)
weights = modal.Volume.from_name("qwen-image-21-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")
# Shared key-value store: the GPU writes "step 12/40" here, the API reads it.
progress = modal.Dict.from_name("qwen-image-21-progress", create_if_missing=True)


# ---------------------------------------------------------------------------
# Step 1: download weights once into a Volume (CPU container, no GPU billing)
# ---------------------------------------------------------------------------
@app.function(
    volumes={CACHE_DIR: weights},
    secrets=[hf_secret],
    cpu=4,
    memory=8192,
    timeout=60 * 60,
)
def download_weights():
    from huggingface_hub import snapshot_download

    t0 = time.time()
    path = snapshot_download(MODEL_ID)
    weights.commit()  # persist so GPU containers see the files
    print(f"Downloaded {MODEL_ID} to {path} in {time.time() - t0:.0f}s")


# ---------------------------------------------------------------------------
# Step 2: the GPU worker
# ---------------------------------------------------------------------------
@app.cls(
    gpu=GPU,
    volumes={CACHE_DIR: weights},
    secrets=[hf_secret],
    scaledown_window=120,  # stay warm 2 min between requests, then scale to zero
    max_containers=1,      # hard cap on parallel GPUs = hard cap on burn rate
    timeout=20 * 60,       # worst case: 4 images at 2k with guidance on
)
class QwenImage:
    @modal.enter()
    def load(self):
        t0 = time.time()
        self.pipe = QwenImage21Pipeline.from_pretrained(
            MODEL_ID, torch_dtype=torch.bfloat16
        ).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)
        print(f"Model loaded in {time.time() - t0:.1f}s")

    @modal.method()
    def generate(self, req: dict, references: list[bytes] | None = None) -> dict:
        """req: GenerateRequest fields minus `images`; references: raw image bytes."""
        from PIL import Image

        r = GenerateRequest(**req)  # re-validate: cheap, and covers CLI callers
        references = references or []
        if len(references) > MAX_REFERENCE_IMAGES:
            raise ValueError(f"at most {MAX_REFERENCE_IMAGES} reference images")
        # Keep alpha only when asked for transparent output.
        refs = [Image.open(io.BytesIO(b)).convert("RGBA" if r.transparent else "RGB") for b in references]

        prompt = r.prompt
        if r.transparent and not prompt.startswith("This is an RGBA image"):
            prompt = RGBA_PREFIX + prompt
        size = r.target_size(has_references=bool(refs))
        base_seed = r.seed if r.seed is not None else random.randint(0, 2**32 - 1)
        job_id = modal.current_function_call_id()

        t_start = time.time()
        results = []
        for k in range(r.num_images):
            seed = (base_seed + k) % 2**32
            kwargs = dict(
                prompt=prompt,
                negative_prompt=r.negative_prompt,
                true_cfg_scale=r.guidance_scale,
                num_inference_steps=r.steps,
                output_resolution=OUTPUT_RESOLUTION[r.size],
                use_kv_cache=r.use_kv_cache,
                generator=torch.Generator("cuda").manual_seed(seed),
                callback_on_step_end=self._progress_reporter(job_id, k, r.num_images, r.steps),
            )
            if refs:
                kwargs["image"] = refs
            if size:
                kwargs["width"], kwargs["height"] = size

            t0 = time.time()
            img = self.pipe(**kwargs).images[0]
            seconds = round(time.time() - t0, 1)
            print(f"{img.width}x{img.height} @ {r.steps} steps, seed {seed}, {len(refs)} refs: {seconds}s")
            results.append({
                "seed": seed,
                "width": img.width,
                "height": img.height,
                "format": r.output_format,
                "seconds": seconds,
                "data": self._encode(img, r),
            })

        return {
            "images": results,
            "prompt_used": prompt,
            "reference_images": len(refs),
            "seconds": round(time.time() - t_start, 1),
        }

    @staticmethod
    def _encode(img, r: GenerateRequest) -> bytes:
        if not r.transparent and img.mode == "RGBA":
            img = img.convert("RGB")  # opaque output: drop the unused alpha channel
        buf = io.BytesIO()
        if r.output_format == "png":
            img.save(buf, format="PNG")
        elif r.output_format == "webp":
            img.save(buf, format="WEBP", quality=r.quality)
        else:
            img.convert("RGB").save(buf, format="JPEG", quality=r.quality)
        return buf.getvalue()

    @staticmethod
    def _progress_reporter(job_id, index, total_images, steps):
        """Diffusers calls this after every step; we publish progress at most once a second."""
        last = [0.0]

        def callback(pipe, step, timestep, callback_kwargs):
            now = time.time()
            if job_id and (now - last[0] >= 1 or step + 1 == steps):
                last[0] = now
                try:
                    progress.put(job_id, {"image": index + 1, "images": total_images,
                                          "step": step + 1, "steps": steps})
                except Exception:
                    pass  # progress is nice-to-have; never fail a generation over it
            return callback_kwargs

        return callback


# ---------------------------------------------------------------------------
# Step 3: the HTTP API (CPU). Validates, then hands work to the GPU.
# ---------------------------------------------------------------------------
def decode_reference_images(encoded: list[str]) -> list[bytes]:
    from PIL import Image

    out = []
    for i, s in enumerate(encoded):
        if s.startswith("data:"):
            s = s.split(",", 1)[-1]
        try:
            raw = base64.b64decode("".join(s.split()), validate=True)
            with Image.open(io.BytesIO(raw)) as im:
                im.verify()
                w, h = im.size
        except Exception:
            raise ValueError(f"images[{i}] is not a valid base64-encoded image")
        if len(raw) > MAX_UPLOAD_BYTES:
            raise ValueError(f"images[{i}] is over {MAX_UPLOAD_BYTES // 2**20} MB")
        if max(w, h) > MAX_UPLOAD_SIDE:
            raise ValueError(f"images[{i}] is {w}x{h}; max side is {MAX_UPLOAD_SIDE}px")
        out.append(raw)
    return out


def serialize_result(result: dict, job_id: str | None = None) -> dict:
    """Jobs get image URLs (fetched separately); sync calls get inline data URLs."""
    images = []
    for i, img in enumerate(result["images"]):
        meta = {k: v for k, v in img.items() if k != "data"}
        if job_id:
            meta["url"] = f"/v1/jobs/{job_id}/images/{i}"
        else:
            b64 = base64.b64encode(img["data"]).decode()
            meta["data_url"] = f"data:{MIME_TYPES[img['format']]};base64,{b64}"
        images.append(meta)
    return {**{k: v for k, v in result.items() if k != "images"}, "images": images}


@app.function(image=api_image, cpu=0.5, memory=1024, max_containers=1)
@modal.concurrent(max_inputs=50)  # one small container serves many polls/requests
@modal.asgi_app(requires_proxy_auth=True)  # callers need Modal-Key / Modal-Secret
def api():
    from fastapi import FastAPI, HTTPException
    from fastapi.responses import JSONResponse, Response

    web = FastAPI(
        title="Qwen-Image-2.1 API",
        version="1.0.0",
        description="Text-to-image, editing and multi-image composition. "
                    "Use /v1/jobs for UIs (progress + no long-held requests), /v1/generate for scripts.",
    )
    gpu = QwenImage()

    def references_or_400(req: GenerateRequest) -> list[bytes]:
        try:
            return decode_reference_images(req.images)
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))

    async def job_result(job_id: str):
        """The finished result, None while running; raises the job's error if it failed."""
        if not job_id.startswith("fc-"):
            raise HTTPException(404, "unknown job id")
        try:
            return await modal.FunctionCall.from_id(job_id).get.aio(timeout=0)
        except (TimeoutError, modal.exception.TimeoutError):
            return None
        except modal.exception.OutputExpiredError:
            raise HTTPException(404, "job results expired (they're kept for 7 days)")
        except modal.exception.NotFoundError:
            raise HTTPException(404, "unknown job id")

    @web.get("/health")
    def health():
        return {"ok": True}

    @web.get("/v1/options")
    def options():
        """Everything a UI needs to build its form: presets, limits, full request schema."""
        return {
            "size_presets": {
                tier: {ar: {"width": w, "height": h} for ar, (w, h) in shapes.items()}
                for tier, shapes in SIZE_PRESETS.items()
            },
            "output_formats": list(MIME_TYPES),
            "limits": {
                "min_side": MIN_SIDE, "max_side": MAX_SIDE, "side_multiple": 32,
                "max_reference_images": MAX_REFERENCE_IMAGES,
                "max_images_per_request": MAX_IMAGES_PER_REQUEST,
                "max_upload_mb": MAX_UPLOAD_BYTES // 2**20,
            },
            "request_schema": GenerateRequest.model_json_schema(),
        }

    @web.post("/v1/jobs", status_code=202)
    async def create_job(req: GenerateRequest):
        """Start a generation and return right away. Poll GET /v1/jobs/{id}."""
        refs = references_or_400(req)
        call = await gpu.generate.spawn.aio(req.model_dump(exclude={"images"}), refs)
        return {"id": call.object_id, "status": "starting", "status_url": f"/v1/jobs/{call.object_id}"}

    @web.get("/v1/jobs/{job_id}")
    async def get_job(job_id: str):
        """202 while starting/running (with progress), 200 when done or failed."""
        try:
            result = await job_result(job_id)
        except HTTPException:
            raise
        except Exception as e:
            return {"id": job_id, "status": "failed", "error": f"{type(e).__name__}: {e}"}
        if result is None:
            p = await progress.get.aio(job_id)
            # No progress yet = waiting for a GPU / loading the model (cold start ~1 min).
            return JSONResponse(status_code=202, content={
                "id": job_id, "status": "running" if p else "starting", "progress": p,
            })
        return {"id": job_id, "status": "done", **serialize_result(result, job_id)}

    @web.get("/v1/jobs/{job_id}/images/{index}")
    async def get_job_image(job_id: str, index: int):
        """The raw image file, usable directly as <img src>."""
        try:
            result = await job_result(job_id)
        except HTTPException:
            raise
        except Exception as e:
            raise HTTPException(409, f"job failed: {e}")
        if result is None:
            raise HTTPException(409, "job is still running")
        if not 0 <= index < len(result["images"]):
            raise HTTPException(404, "no image at that index")
        img = result["images"][index]
        filename = f"qwen_{img['seed']}.{EXTENSIONS[img['format']]}"
        return Response(
            content=img["data"],
            media_type=MIME_TYPES[img["format"]],
            headers={
                "Cache-Control": "private, max-age=86400",  # a job's images never change
                "Content-Disposition": f'inline; filename="{filename}"',
            },
        )

    @web.post("/v1/generate")
    async def generate_sync(req: GenerateRequest):
        """Wait for the images and return them inline as data URLs. Simple, for scripts."""
        refs = references_or_400(req)
        result = await gpu.generate.remote.aio(req.model_dump(exclude={"images"}), refs)
        return serialize_result(result)

    return web


# ---------------------------------------------------------------------------
# Step 4: local CLI - runs on your laptop, sends work straight to the GPU
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = "A glowing neon shop sign that reads 'QWEN ON MODAL', rainy alley at night, cinematic photo",
    prompts_file: str = "",
    images: str = "",  # comma-separated reference image paths, for editing
    n: int = 1,
    size: str = "1k",
    aspect_ratio: str = "",
    width: int = 0,
    height: int = 0,
    steps: int = 40,
    seed: int = -1,
    transparent: bool = False,
    negative_prompt: str = "",
    guidance_scale: float = 1.0,
    output_format: str = "png",
    out_dir: str = "qwen_outputs",
):
    from pathlib import Path

    if prompts_file:
        prompts = [p.strip() for p in Path(prompts_file).read_text().splitlines() if p.strip()]
    else:
        prompts = [prompt]
    references = [Path(p.strip()).read_bytes() for p in images.split(",") if p.strip()]

    # Validate locally so mistakes fail instantly instead of after a GPU cold start.
    reqs = [
        GenerateRequest(
            prompt=p, num_images=n, size=size, aspect_ratio=aspect_ratio or None,
            width=width or None, height=height or None, steps=steps,
            seed=seed if seed >= 0 else None, transparent=transparent,
            negative_prompt=negative_prompt or None, guidance_scale=guidance_scale,
            output_format=output_format,
        ).model_dump(exclude={"images"})
        for p in prompts
    ]

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    stamp = int(time.time())
    t0 = time.time()
    count = 0
    for i, result in enumerate(QwenImage().generate.map(reqs, [references] * len(reqs))):
        for k, img in enumerate(result["images"]):
            path = out / f"{stamp}_{i:03d}_{k}_seed{img['seed']}.{EXTENSIONS[img['format']]}"
            path.write_bytes(img["data"])
            count += 1
            print(f"saved {path} ({img['width']}x{img['height']}, {img['seconds']}s)")
    print(f"{count} image(s) in {time.time() - t0:.0f}s (includes cold start)")
