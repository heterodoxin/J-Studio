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

### J-space interventions

- Added ordered multi-token inject and replace operators.
- Added generated-token causal probes that choose the minimum effective strength.
- Made intervention duration control residual-hook lifetime, eliminating repeated
  target-token collapse for default Next Token edits.
- Kept failed bounded searches fail-closed with no logit-steering fallback.

### Runtime

- Improved Stable lens compatibility checks and progressive fitting status.
- Kept decoder generation and J-Lens rendering compatible with local ROCm BF16
  Qwen-family models.
