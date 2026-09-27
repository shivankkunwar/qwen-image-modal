# Cost math (Modal Starter plan)

- Plan: $0/month, includes $30/month compute credits, up to 10 concurrent GPUs.
- Billing: per second for GPU + CPU + memory; nothing while scaled to zero.
- GPU hourly (approx, 2026): L4 $0.80 · L40S $1.95 · A100-80GB $2.50 · H100 $3.95.

## Our setup: L40S, bf16, 40 steps
- $30 / $1.95 ≈ 15.4 GPU-hours ≈ 55,000 s.
- 1024×1024 ≈ 7–8 s → ~7,000 images theoretical, ~$0.004–0.006 each.
  **Measured 2026-09-27 (first image, fresh container): 18.2 s generation +
  27 s model load, 64 s total wall time.** At 18 s that's ~$0.01/image and
  ~3,000 images theoretical. Re-measure a warm batch before trusting either number.
- 2048×2048 ≈ 30 s → ~1,800 theoretical.
- Realistic: ½–⅔ of theoretical because of cold starts (~30 GB load),
  idle scaledown time, CPU/memory charges.

## Rules of thumb
- Batch prompts (`--prompts-file`) → one cold start for many images.
- Short `scaledown_window` for occasional use; longer only during sessions.
- Never leave `min_containers>0` on (L40S 24/7 ≈ $1,400/month).
- Step-distilled model (8 steps vs 40) ≈ 5x more images per dollar.

## Sources
- https://modal.com/pricing
- https://www.budgetforge.dev/tools/modal-pricing-2026
