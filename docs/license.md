# License notes (as of 2026-09-27) – not legal advice

- Qwen-Image-2.1 ships under the **Qwen Research License** (non-commercial),
  unlike earlier Qwen-Image v1 models (Apache 2.0).
- LICENSE text: commercial use of the "Materials" (weights, code, docs) needs a
  separate commercial license (contact: model-business@notice.qwencloud.com).
- Official clarification from @QwenDevs on X (~2026-09-21): outputs are NOT
  part of the licensed Materials; users retain rights to what they generate.
  The LICENSE file itself had not been updated as of 2026-09-27.

## Practical reading
- Using/selling images you generated yourself: OK per Qwen's clarification.
- Running the model as a paid service for others (charging per generation,
  paid API, SaaS): commercial use of the Materials → needs a grant.
- Derivatives (LoRAs, quants, distilled checkpoints) inherit the research
  license.
- Fully commercial-safe alternative: Qwen-Image v1 family (Apache 2.0, 20B,
  needs ~80 GB GPU, ~20–40 s/image).

## Sources
- https://huggingface.co/Qwen/Qwen-Image-2.1/discussions/40
- https://x.com/QwenDevs/status/2101917379785838660
