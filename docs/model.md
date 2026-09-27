# Model: Qwen-Image-2.1 (facts as of 2026-09-27)

- Released 2026-09-20 by Alibaba's Qwen team. Open weights on Hugging Face
  (`Qwen/Qwen-Image-2.1`), ModelScope, GitHub.
- Architecture: 7B-parameter single-stream DiT (32 layers) as the image generator
  + Qwen3-VL 8B as text/vision encoder + a 16x RGBA autoencoder (VAE).
- One model does text-to-image AND editing (up to 10 reference images).
- Native transparency: outputs RGBA PNGs. Trigger it by starting the prompt with
  "This is an RGBA image with transparency."
- Native 2K: max side 2752 px. Dimensions must be divisible by 32.
- Prefix KV cache: prompt/reference tokens computed once, reused every step
  (`use_kv_cache=True`).

## Memory / speed (community measurements)
- Fully loaded bf16: ~27–30 GB VRAM → needs a 48 GB GPU (L40S) for comfort.
- Weights on disk: ~30 GB (bf16).
- RTX 4090: ~7.5 s per 1024×1024 image; ~30 s at 2048×2048; 4096² takes minutes
  and artifacts. L40S is roughly 4090-class.

## Pipeline API (diffusers)
- Class: `diffusers.QwenImage21Pipeline` – currently only on diffusers **main**
  (not in stable release). We pin commit
  `6256aa7666cedd47443adc8f82da9a10e110b09c` (used by PrunaAI's release).
- Requires transformers >= 5.17.
- Recommended: `num_inference_steps=40`, `true_cfg_scale=1.0` (no CFG).
  CFG only turns on with `true_cfg_scale>1` AND a `negative_prompt`, and it
  doubles the cost per step.
- Editing: pass `image=PIL.Image` (or a list) plus an instruction prompt.
  - List order matters: the model reads images in the order given, so prompts
    can say "the first image" / "the second image". Max 10 references.
  - There is **no `mask=` argument**. Local edits are marked visually: draw a
    circle/scribble on the image, or pass a black/white mask as an extra image,
    and refer to it in the prompt.
  - With `image=` given, omit `width`/`height` and the output follows the
    first image's aspect ratio; `output_resolution` (default 1024) sets the size
    and also resizes the condition images.
  - `use_kv_cache` changes pixels slightly: same seed with it on vs off gives
    different (equally valid) images. Keep it fixed for reproducibility.
- Not in this pipeline: ControlNet/pose/depth conditioning, a dedicated
  inpaint pipeline, or an upscaler (checked 2026-09-27).

```python
pipe = QwenImage21Pipeline.from_pretrained("Qwen/Qwen-Image-2.1", torch_dtype=torch.bfloat16).to("cuda")
img = pipe(prompt="...", num_inference_steps=40, true_cfg_scale=1.0, use_kv_cache=True,
           generator=torch.Generator("cuda").manual_seed(42)).images[0]
```

## Native aspect-ratio presets (2K)
2048×2048, 2400×1792 (4:3), 1792×2400 (3:4), 2528×1696 (3:2), 1696×2528 (2:3),
2752×1536 (16:9), 1536×2752 (9:16). Smaller test sizes: 1024², 1152×864,
1248×832, 1536×864.

## Sources
- https://huggingface.co/Qwen/Qwen-Image-2.1
- https://huggingface.co/docs/diffusers/main/api/pipelines/qwenimage21
- https://recipes.vllm.ai/Qwen/Qwen-Image-2.1
- https://www.mindstudio.ai/blog/qwen-image-2-1-local-install
