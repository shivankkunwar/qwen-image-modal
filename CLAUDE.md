# Project: Qwen-Image-2.1 on Modal (free tier)

## What this is
A single-file Modal app (`qwen_image_modal.py`) that runs Alibaba's Qwen-Image-2.1
image model on a serverless GPU, using Modal's Starter plan ($0 plan, $30/month
compute credits). Goal: generate images cheaply, scale to zero when idle.

Owner: Shivank (frontend engineer, JS/TS + Vue/Nuxt, learning Python/infra/LLM
inference). Explain Python/infra decisions briefly when making changes.

## Read these before changing anything
- `SETUP.md` – the step-by-step run guide (keep it in sync with the code)
- `docs/model.md` – model facts, pipeline API, recommended settings
- `docs/modal-architecture.md` – why Functions (@app.cls) and not Servers
- `docs/costs.md` – credit math; protect the $30 budget
- `docs/license.md` – what's allowed commercially
- `docs/unsloth-and-speedups.md` – quantization verdict + faster options

## Hard rules
1. Protect the budget: keep `max_containers=1` and a short `scaledown_window`
   unless Shivank asks otherwise. Never set `min_containers>0` without asking
   (that bills a GPU 24/7, ~$1.95/hr on L40S).
2. Keep weights in the Modal Volume `qwen-image-21-weights`; never download
   weights inside the GPU container at request time.
3. Keep diffusers pinned to a commit (QwenImage21Pipeline is only on main).
   If you bump it, confirm `QwenImage21Pipeline` and `use_kv_cache` still exist.
4. Web endpoints must keep `requires_proxy_auth=True`.
5. Don't enable GPU memory snapshots by default (see docs/modal-architecture.md).
6. Model/Modal APIs change fast (model released 2026-09-20). Check current docs
   before relying on memory: Modal docs (modal.com/docs), the HF model card
   (huggingface.co/Qwen/Qwen-Image-2.1), diffusers docs for QwenImage21Pipeline.

## Commands
- One-time: `pip install -U modal && modal setup`
- Secret: `modal secret create huggingface HF_TOKEN=hf_xxx`
- Download weights: `modal run qwen_image_modal.py::download_weights`
- Generate: `modal run qwen_image_modal.py --prompt "..."`
- Batch: `modal run qwen_image_modal.py --prompts-file prompts.txt`
- Edit images: `modal run qwen_image_modal.py --prompt "..." --images a.png,b.png`
- Deploy API: `modal deploy qwen_image_modal.py` → https://shivankkunwar100--qwen-image-21-api.modal.run (reference: `docs/api.md`)
- Local proxy (holds Modal keys, forwards /v1/*, /docs): `node --env-file=api/.env api/server.mjs`
- On Shivank's Windows machine `modal` isn't on PATH: use `python -m modal ...`
- Logs / usage: Modal dashboard → Apps → qwen-image-21; Usage page for credits

## Ideas backlog (only if asked)
- Try step-distilled checkpoint (PrunaAI/Pruna-Qwen-Image-2.1, 5–8 steps) → ~5x more images per credit; compare quality first
- Image-editing endpoint (pipeline accepts `image=` with up to 10 references)
- Small web UI (Nuxt/Vue) that calls the deployed endpoint
