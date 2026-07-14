# Changelog

## Unreleased

### Interface

- Added the modern graphite/violet J Studio workbench theme.
- Preserved the Cheat Engine-inspired read/results/intervention structure while
  moving advanced scan fields behind progressive disclosure.
- Clarified model, backend, lens, and fit state in the session identity bar.
- Modernized Chat and Rules layout hierarchy and reduced always-visible actions.
- Themed the native J-Lens workspace and its exported interactive HTML surface,
  including heatmaps, rank plots, pins, tooltips, and selected coordinates.
- Added right-click intervention actions for edit, duplicate, preview, ordering,
  enable/disable, and removal.
- Made Trace Influence open the trace tool with the clicked concept as its seed.

### J-space interventions

- Added ordered multi-token inject and replace operators.
- Added generated-token causal probes that choose the minimum effective strength.
- Made intervention duration control residual-hook lifetime, eliminating repeated
  target-token collapse for default Next Token edits.
- Rejects target-only, target-repeating, leading-target, and trajectory-derailing
  probes; accepted edits retain ordered token anchors from the baseline response.
- Compiles default bare-concept injection into a trace-visible friendly semantic
  carrier and schedules it at the first measured sentence boundary.
- Preserves the baseline punctuation inside the carrier, so short answers can become
  `42. I like banana.` without increasing the strength budget.
- Aligns default Replace and Suppress hooks to the generated source phrase and uses
  contextual leading-space token directions instead of editing the first token.
- Expands causal probes according to carrier length and delay so coherent replies are
  not rejected merely because a fixed 12-token probe ended too early.
- Keeps explicit Steps/Generation phrase transport literal and ordered.
- Kept failed bounded searches fail-closed with no logit-steering fallback.

### Runtime

- Improved Stable lens compatibility checks and progressive fitting status.
- Kept decoder generation and J-Lens rendering compatible with local ROCm BF16
  Qwen-family models.
