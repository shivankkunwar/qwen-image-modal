# Simple setup – Qwen-Image-2.1 on Modal

Total time: ~20–30 min (mostly the 30 GB download). Cost of setup: ~nothing.

## 1. Install Modal and log in (2 min)
```bash
pip install -U modal
modal setup
```
A browser tab opens → log in with GitHub. Done.

## 2. Get a Hugging Face token (2 min)
1. Go to https://huggingface.co/settings/tokens → "Create new token" → type **Read**.
2. Open https://huggingface.co/Qwen/Qwen-Image-2.1 and accept the license if it asks.

## 3. Give the token to Modal (1 min)
```bash
modal secret create huggingface HF_TOKEN=hf_paste_your_token_here
```

## 4. Download the model once (15–25 min, CPU only, basically free)
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

## 6. Try more
```bash
# 4 variations of one prompt
modal run qwen_image_modal.py --prompt "cartoon dragon sticker" --n 4

# transparent background PNG
modal run qwen_image_modal.py --prompt "cartoon dragon sticker" --transparent

# many prompts at once (cheapest way – one warm GPU does them all)
modal run qwen_image_modal.py --prompts-file prompts.txt

# widescreen
modal run qwen_image_modal.py --prompt "..." --width 1536 --height 864
```

## 7. (Optional) Turn it into an API
```bash
modal deploy qwen_image_modal.py
```
- Copy the URL it prints (ends in `...-web.modal.run`).
- Modal dashboard → Settings → Proxy Auth Tokens → create one.
- Call it:
```bash
curl -L -X POST "YOUR_URL" \
  -H "Modal-Key: YOUR_KEY" -H "Modal-Secret: YOUR_SECRET" \
  -H "Content-Type: application/json" \
  -d '{"prompt": "a red kite over a lake"}' --output kite.png
```
`-L` matters: requests over 150 s (e.g. a cold start) get a 303 redirect that
must be followed. Still scales to zero – you pay only while it's generating (+2 min idle).

## 7b. (Optional) Local API wrapper – `api/server.mjs`
A zero-dependency Node server that holds your Modal keys and exposes a simple
`POST /generate`. Use this from scripts or a future web UI – never put
Modal-Key/Secret in browser code.
```bash
cp api/.env.example api/.env     # then fill in URL, key, secret
node --env-file=api/.env api/server.mjs
```
Call it:
```bash
curl -X POST http://127.0.0.1:8787/generate -H "Content-Type: application/json" -d '{"prompt": "a red kite over a lake", "width": 1536, "height": 864}' --output kite.png
```
Body fields: `prompt` (required), `width`, `height`, `steps`, `seed`, `transparent`.
Errors come back as JSON `{ "error": ..., "detail": ... }`.

## 8. Watch your credits
Modal dashboard → **Usage**. Rough guide: ~$0.004–0.006 per 1024×1024 image,
plus ~$0.03–0.05 each time the GPU cold-starts. Batch your prompts.

## If something breaks
| Error | Fix |
|---|---|
| `Secret 'huggingface' not found` | Redo step 3 |
| 401/403 during download | Accept license on the HF model page; check token |
| `cannot import QwenImage21Pipeline` | diffusers commit changed – see docs/model.md |
| `width and height must be divisible by 32` | Use sizes like 1024, 1152, 1536, 864 |
| CUDA out of memory | Lower resolution; keep batch at 1 |
| Very slow first image | Normal – cold start loads 30 GB |
