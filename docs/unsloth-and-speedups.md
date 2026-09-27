# Unsloth quants and speed-ups

## Unsloth (unsloth/Qwen-Image-2.1-FP8, -GGUF)
Verdict for Modal: **don't use**. Quantization shrinks memory; it doesn't make
images better.
- Unsloth's own test vs full precision: INT8 LPIPS 0.064, FP8 LPIPS 0.112
  (lower = closer to original) → both measurably lossy vs bf16.
- Only saves money if it lets you drop to a cheaper 24 GB GPU, but Unsloth's
  table lists 24 GB as INT8/FP8 at 512×512 only (GGUF Q4 at 1024). Worse
  quality on a slower card.
- Useful for running locally on a consumer GPU/Mac, not for our L40S.

## Real speed levers (in order of impact)
1. **Step-distilled checkpoint** – PrunaAI/Pruna-Qwen-Image-2.1: 5 or 8 steps
   instead of 40, same QwenImage21Pipeline. ~5x throughput. Released
   2026-09-27; compare quality on your prompts first. Needs its own sigma
   schedule + scheduler shift=1.0 (see its model card). Research license.
2. `use_kv_cache=True` (already on).
3. Keep CFG off (`true_cfg_scale=1.0`).
4. Batching prompts through one warm container.
5. torch.compile – only for long warm sessions.

## Sources
- https://unsloth.ai/docs/models/qwen-image-2.1
- https://huggingface.co/PrunaAI/Pruna-Qwen-Image-2.1
