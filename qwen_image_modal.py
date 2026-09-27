"""
Qwen-Image-2.1 on Modal (Starter plan, $30/month credits).

One-time setup:
    pip install -U modal
    modal setup                                   # browser login
    modal secret create huggingface HF_TOKEN=hf_xxx
    modal run qwen_image_modal.py::download_weights   # ~30 GB -> Volume, CPU only

Generate (ephemeral app, scales to zero when done):
    modal run qwen_image_modal.py --prompt "a neon shop sign that reads SHVNK"
    modal run qwen_image_modal.py --prompt "cartoon dragon sticker" --transparent --n 4
    modal run qwen_image_modal.py --prompts-file prompts.txt

Deploy as a persistent endpoint (still scales to zero, billed only while busy):
    modal deploy qwen_image_modal.py
"""

import io
import random
import time

import modal

APP_NAME = "qwen-image-21"
MODEL_ID = "Qwen/Qwen-Image-2.1"
# QwenImage21Pipeline is only on diffusers main right now. Pin a commit that is
# known to have it (and the use_kv_cache arg) so a future main push can't break you.
DIFFUSERS_COMMIT = "6256aa7666cedd47443adc8f82da9a10e110b09c"
GPU = "L40S"  # 48 GB: bf16 model (~27-30 GB loaded) fits with headroom
CACHE_DIR = "/models"
MAX_SIDE = 2752  # model's largest native side

# ---------------------------------------------------------------------------
# Container image
# ---------------------------------------------------------------------------
image = (
    modal.Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .uv_pip_install(
        "torch>=2.6",
        "transformers>=5.17",
        "accelerate",
        "peft",
        "pillow",
        "huggingface_hub[hf_xet]",
        "fastapi[standard]",
        f"git+https://github.com/huggingface/diffusers@{DIFFUSERS_COMMIT}",
    )
    .env({"HF_HUB_CACHE": CACHE_DIR, "HF_XET_HIGH_PERFORMANCE": "1"})
)

# These imports only run inside the container, not on your laptop.
with image.imports():
    import torch
    from diffusers import QwenImage21Pipeline

app = modal.App(APP_NAME, image=image)
weights = modal.Volume.from_name("qwen-image-21-weights", create_if_missing=True)
hf_secret = modal.Secret.from_name("huggingface")


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
    timeout=15 * 60,
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

    def _run(self, prompt, width, height, steps, seed, transparent) -> bytes:
        if width % 32 or height % 32:
            raise ValueError("width and height must be divisible by 32")
        if max(width, height) > MAX_SIDE:
            raise ValueError(f"max side is {MAX_SIDE}px")
        if transparent and not prompt.startswith("This is an RGBA image"):
            # Qwen's recommended prefix for native alpha-channel output
            prompt = "This is an RGBA image with transparency. " + prompt
        if seed is None:
            seed = random.randint(0, 2**31 - 1)

        t0 = time.time()
        img = self.pipe(
            prompt=prompt,
            width=width,
            height=height,
            num_inference_steps=steps,
            true_cfg_scale=1.0,   # Qwen recommends no CFG; >1 doubles cost
            use_kv_cache=True,    # reuse prompt KV across steps
            generator=torch.Generator("cuda").manual_seed(seed),
        ).images[0]
        print(f"{width}x{height} @ {steps} steps, seed {seed}: {time.time() - t0:.1f}s")

        buf = io.BytesIO()
        img.save(buf, format="PNG")  # PNG keeps the alpha channel
        return buf.getvalue()

    @modal.method()
    def generate(
        self,
        prompt: str,
        width: int = 1024,
        height: int = 1024,
        steps: int = 40,
        seed: int | None = None,
        transparent: bool = False,
    ) -> bytes:
        return self._run(prompt, width, height, steps, seed, transparent)

    # HTTP endpoint. requires_proxy_auth means callers need your Modal-Key /
    # Modal-Secret headers, so a leaked URL can't drain your credits.
    @modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)
    def web(self, req: dict):
        from fastapi import HTTPException
        from fastapi.responses import Response

        # Bad input -> 400 with a readable message instead of an opaque 500.
        try:
            png = self._run(
                prompt=req["prompt"],
                width=int(req.get("width", 1024)),
                height=int(req.get("height", 1024)),
                steps=int(req.get("steps", 40)),
                seed=req.get("seed"),
                transparent=bool(req.get("transparent", False)),
            )
        except KeyError as e:
            raise HTTPException(status_code=400, detail=f"missing field: {e}")
        except ValueError as e:
            raise HTTPException(status_code=400, detail=str(e))
        return Response(content=png, media_type="image/png")


# ---------------------------------------------------------------------------
# Step 3: local CLI - runs on your laptop, fans work out to the GPU
# ---------------------------------------------------------------------------
@app.local_entrypoint()
def main(
    prompt: str = "A glowing neon shop sign that reads 'QWEN ON MODAL', rainy alley at night, cinematic photo",
    prompts_file: str = "",
    n: int = 1,
    width: int = 1024,
    height: int = 1024,
    steps: int = 40,
    transparent: bool = False,
    out_dir: str = "qwen_outputs",
):
    from pathlib import Path

    if prompts_file:
        prompts = [p.strip() for p in Path(prompts_file).read_text().splitlines() if p.strip()]
    else:
        prompts = [prompt] * n

    out = Path(out_dir)
    out.mkdir(exist_ok=True)
    k = len(prompts)
    seeds = [random.randint(0, 2**31 - 1) for _ in range(k)]

    t0 = time.time()
    results = QwenImage().generate.map(
        prompts, [width] * k, [height] * k, [steps] * k, seeds, [transparent] * k
    )
    for i, png in enumerate(results):
        path = out / f"{int(time.time())}_{i:03d}_seed{seeds[i]}.png"
        path.write_bytes(png)
        print(f"saved {path}")
    print(f"{k} image(s) in {time.time() - t0:.0f}s (includes cold start)")
