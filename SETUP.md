# Simple setup – Qwen-Image-2.1 on Modal

Total time: ~20–30 min (mostly the 30 GB download). Cost of setup: ~nothing.

## 1. Install Modal and log in (2 min)
```bash
pip install -U modal pydantic
modal setup
```
A browser tab opens → log in with GitHub. Done.
On Windows, if `modal` isn't found, use `python -m modal ...` everywhere below.

## 2. Get a Hugging Face token (2 min)
1. Go to https://huggingface.co/settings/tokens → "Create new token" → type **Read**.
2. Open https://huggingface.co/Qwen/Qwen-Image-2.1 and accept the license if it asks.

## 3. Give the token to Modal (1 min)
```bash
modal secret create huggingface HF_TOKEN=hf_paste_your_token_here
```

## 4. Download the model once (~5–25 min, CPU only, basically free)
From inside this folder:
```bash
modal run qwen_image_modal.py::download_weights
```
Wait for: `Downloaded Qwen/Qwen-Image-2.1 ... in XXXs`

## 5. Make your first image (~1–2 min first time)
```bash
modal run qwen_image_modal.py --prompt "a cozy Bengaluru cafe at night, cinematic photo"
```
The PNG lands in `qwen_outputs/`. First run is slower (loading 30 GB onto the GPU).

## 6. Try more (CLI)
```bash
# 4 variations of one prompt (seeds s, s+1, s+2, s+3)
modal run qwen_image_modal.py --prompt "cartoon dragon sticker" --n 4

# transparent background PNG
modal run qwen_image_modal.py --prompt "cartoon dragon sticker" --transparent

# edit an image / combine several (comma-separated, order matters)
modal run qwen_image_modal.py --prompt "make it a rainy night" --images qwen_outputs/cafe.png
modal run qwen_image_modal.py --prompt "put the lamp from the first image on the desk in the second" --images lamp.png,desk.png

# shape and quality: presets or exact size
modal run qwen_image_modal.py --prompt "..." --aspect-ratio 16:9 --size 2k
modal run qwen_image_modal.py --prompt "..." --width 1536 --height 864

# reproduce an image: same prompt + same seed
modal run qwen_image_modal.py --prompt "..." --seed 2023230770

# many prompts at once (cheapest way – one warm GPU does them all)
modal run qwen_image_modal.py --prompts-file prompts.txt
```
Other flags: `--steps`, `--negative-prompt "..." --guidance-scale 4`,
`--output-format webp|jpeg`. All options are explained in docs/api.md.

## 7. The API
```bash
modal deploy qwen_image_modal.py
```
Prints the API URL (`https://<workspace>--qwen-image-21-api.modal.run`). It's
protected: every call needs a proxy token.
1. Modal dashboard → Settings → **Proxy Auth Tokens** → create one (you get a key `wk-...` and secret `ws-...`).
2. Copy `api/.env.example` to `api/.env` and fill in the URL, key and secret.
3. Start the local proxy (keeps the keys off the browser, adds them for you):
```bash
node --env-file=api/.env api/server.mjs
```
4. Open http://127.0.0.1:8787/docs for interactive docs, or:
```bash
curl -X POST http://127.0.0.1:8787/v1/jobs -H "Content-Type: application/json" -d '{"prompt": "a red kite over a lake"}'
```
Then poll `GET /v1/jobs/<id>` and download `GET /v1/jobs/<id>/images/0`.
Full reference: docs/api.md. Still scales to zero – you pay only while generating (+2 min idle).

## 8. Watch your credits
Modal dashboard → **Usage**. Measured: ~$0.01 per 1024×1024 image (18 s),
plus ~$0.03–0.05 each time the GPU cold-starts. 2k is ~4x the cost. Batch your prompts.

## If something breaks
| Error | Fix |
|---|---|
| `Secret 'huggingface' not found` | Redo step 3 |
| 401/403 during download | Accept license on the HF model page; check token |
| `cannot import QwenImage21Pipeline` | diffusers commit changed – see docs/model.md |
| `Input should be a multiple of 32` | Use sizes like 1024, 1152, 1536, 864, or `--aspect-ratio` |
| `No module named 'pydantic'` | `pip install pydantic` |
| API returns 401 | Check `MODAL_KEY`/`MODAL_SECRET` in `api/.env` |
| CUDA out of memory | Lower resolution; keep batch at 1 |
| Very slow first image | Normal – cold start loads 30 GB |
