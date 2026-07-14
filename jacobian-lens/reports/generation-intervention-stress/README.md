# Generation intervention stress benchmark

The `qwen3_8b_apostate_pre_alignment` artifact records the first complete real-model
stress run on 2026-07-13. It deliberately preserves the failing 6/22 baseline that
identified two scheduling defects:

- a fixed 12-token causal window rejected coherent longer responses;
- Replace and Suppress were applied at the first generated token rather than the
  generated source phrase.

After sentence/source alignment, targeted real-model reruns produced:

- `hi + banana` → `Hello! I like banana. What's your favorite fruit?`
- `12 + 30 + banana` → `42. I like banana.`
- `France capital + saffron` → `Paris. I like saffron.`
- `cold → warm` → `The room is warm.` at minimum strength 2
- `red → blue` → `I see a blue car.` at minimum strength 2

Pure suppression of strongly instructed literal outputs still failed closed at the
configured maximum strength. No logit, prompt, or output fallback was used in either
run. A complete post-alignment matrix was intentionally deferred to avoid sustained
GPU load; the focused regression and package suites cover the scheduling changes.
