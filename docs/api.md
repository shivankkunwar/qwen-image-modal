# API reference (v1)

Base URL: `https://shivankkunwar100--qwen-image-21-api.modal.run` (proxy auth:
`Modal-Key` / `Modal-Secret` headers). Locally, go through `api/server.mjs`
at `http://127.0.0.1:8787`, which adds the headers. Interactive docs: `/docs`.

## How it's built
```
client ──► api (CPU, FastAPI, cheap)  ──spawn/remote──►  QwenImage.generate (L40S GPU)
              │ validates, decodes images                    │ writes progress
              └──────────── reads ──── modal.Dict "qwen-image-21-progress" ◄┘
```
Bad requests, `/v1/options` and job polling never wake the GPU.

## Endpoints
| Method | Path | What |
|---|---|---|
| GET | `/health` | `{ok: true}` |
| GET | `/v1/options` | Size presets, limits, output formats, full JSON schema of the request. Build UI forms from this. |
| POST | `/v1/jobs` | Start a generation → `202 {id, status: "starting", status_url}` |
| GET | `/v1/jobs/{id}` | `202 {status: "starting"}` (waiting for GPU / loading model) · `202 {status: "running", progress: {image, images, step, steps}}` · `200 {status: "done", images: [...]}` · `200 {status: "failed", error}` |
| GET | `/v1/jobs/{id}/images/{k}` | Raw image bytes (usable as `<img src>` via the proxy). `409` if not done. |
| POST | `/v1/generate` | Waits, returns `{images: [{..., data_url}]}` inline. For scripts. |

Job results are kept 7 days. `max_containers=1` on the GPU, so jobs queue
one at a time.

## Request body (POST /v1/jobs and /v1/generate)
Only `prompt` is required.

| Field | Default | Notes |
|---|---|---|
| `prompt` | — | What to create, or the edit instruction |
| `images` | `[]` | 0–10 reference images, base64 or data URLs, ≤20 MB / ≤8192 px each. Order matters ("the first image"). Mark local edits by drawing on the image (no mask param). |
| `size` | `"1k"` | `"1k"` ≈1 MP, `"2k"` native (≈4x slower) |
| `aspect_ratio` | `null` | `1:1 4:3 3:4 3:2 2:3 16:9 9:16`. Null → 1:1, or first reference's shape when editing |
| `width`, `height` | `null` | Exact size, both or neither, multiple of 32, 256–2752. Can't combine with `aspect_ratio` |
| `num_images` | `1` | 1–4; image k uses `seed + k` |
| `steps` | `40` | 1–100 |
| `seed` | random | 0–2³²−1. Returned per image |
| `transparent` | `false` | RGBA output (adds Qwen's prefix). png/webp only |
| `negative_prompt` | `null` | Requires `guidance_scale > 1` |
| `guidance_scale` | `1.0` | `true_cfg_scale`. >1 needs `negative_prompt`, doubles time |
| `output_format` | `"png"` | `png`, `webp`, `jpeg` |
| `quality` | `95` | webp/jpeg only |
| `use_kv_cache` | `true` | Keep fixed when reproducing a seed |

Validation errors → `422` (FastAPI format, `detail[]`); bad image data → `400`.

## Response (done)
```json
{
  "id": "fc-...", "status": "done",
  "prompt_used": "...", "reference_images": 1, "seconds": 19.4,
  "images": [{"seed": 123, "width": 1024, "height": 1024, "format": "png",
              "seconds": 18.2, "url": "/v1/jobs/fc-.../images/0"}]
}
```
`/v1/generate` returns the same shape without `id`/`status`, and with
`data_url` instead of `url`.

## Examples
```bash
# start a job, then poll it
curl -X POST http://127.0.0.1:8787/v1/jobs -H "Content-Type: application/json" -d '{"prompt": "a red kite over a lake", "aspect_ratio": "16:9"}'
curl http://127.0.0.1:8787/v1/jobs/fc-XXXX
curl http://127.0.0.1:8787/v1/jobs/fc-XXXX/images/0 --output kite.png
```
```js
// edit an image from the browser (File from <input type="file">)
const dataUrl = await new Promise(r => { const f = new FileReader(); f.onload = () => r(f.result); f.readAsDataURL(file) })
const { id } = await (await fetch('/v1/jobs', { method: 'POST', headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ prompt: 'make it a rainy night', images: [dataUrl] }) })).json()
```
