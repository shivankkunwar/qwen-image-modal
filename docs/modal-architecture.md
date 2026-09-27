# Modal architecture notes (SDK 1.5.x, 2026) and why the app is built this way

## Two ways to serve
1. **Functions** – `@app.function` / `@app.cls` + `@modal.method`.
   Requests go into an input queue. If no container is running, the request
   waits while one boots. Supports `.remote()`, `.map()`, web endpoints.
2. **Servers** – `@app.server(port=..., target_concurrency=...)` (newer).
   A stateless reverse proxy talks straight to your HTTP server process.
   Lower latency, sticky sessions (`Modal-Session-ID` header), but **no
   queueing**: with zero containers, requests get **503**. Built for
   always-warm LLM serving (vLLM etc.).

**Choice: Functions.** Free-tier budget needs scale-to-zero. A Server would
either 503 the first request or need `min_containers>=1` (GPU billed 24/7).

## Building blocks used
- **Image**: `modal.Image.debian_slim().uv_pip_install(...)`; `with image.imports():`
  keeps container-only imports from running locally.
- **Volume**: `modal.Volume.from_name(..., create_if_missing=True)`; weights
  downloaded once by a CPU function, then `volume.commit()`.
- **Secret**: `modal.Secret.from_name("huggingface")` exposes `HF_TOKEN`.
- **Lifecycle**: `@modal.enter()` loads the model once per container.
- **Autoscaler knobs**: `scaledown_window` (2 s–20 min, default 60 s),
  `min_containers`, `buffer_containers`, `max_containers`.
  We use `scaledown_window=120`, `max_containers=1` (caps burn rate).
- **Web endpoint**: `@modal.fastapi_endpoint(method="POST", requires_proxy_auth=True)`
  – callers need Modal-Key / Modal-Secret headers.

## Deliberately NOT used
- **GPU memory snapshots** (`enable_memory_snapshot=True` +
  `experimental_options={"enable_gpu_snapshot": True}`): alpha feature. Modal's
  docs say that when cold start is dominated by loading weights, GPU snapshots
  generally don't help and can make it worse. Our cold start IS weight loading.
  Also reported restore failures on H100 (modal-client issue #4132).
- **routing_region="ap-south"** (Mumbai): regionally routed Functions only
  support `.remote()`/`.map()`, which breaks the web endpoint; latency is
  irrelevant next to ~8 s generation anyway.
- **compute_region** pinning: adds a price multiplier.
- **torch.compile**: compile time on every cold start eats credits; only worth
  it if a container stays warm for long batches.

## Newer features worth knowing
- Named images: `Image.publish("name:tag")` / `Image.from_name()` (1.5.0).
- `Function.with_options()`, `.with_concurrency()`, `.with_batching()` for
  runtime overrides; `update_autoscaler()` on deployed Functions/Servers.
- Logs APIs (`.stream()/.fetch()/.tail()`) and `modal billing rates` CLI.

## Sources
- https://modal.com/docs/guide/cold-start
- https://modal.com/docs/guide/memory-snapshots
- https://modal.com/docs/guide/servers
- https://modal.com/docs/guide/region-selection
- https://modal.com/docs/sdk/py/changelog
