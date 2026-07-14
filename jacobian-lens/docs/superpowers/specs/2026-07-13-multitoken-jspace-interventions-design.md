# Multi-token J-space Interventions

## Goal

Make inject, suppress, and replace accept arbitrary tokenizer sequences while
remaining true residual-stream/J-space operations. No operation may force an
output token, modify next-token logits, or fall back to a logit-based method.
Generation must continue when one rule cannot be resolved: the failed rule is
reported and omitted while other valid rules remain active.

## Phrase representation

A phrase is encoded without special tokens in both its bare and leading-space
forms. Empty encodings are rejected and duplicate token sequences are removed.
Each token id `t` has an unembedding covector `u_t`. At fitted layer `l`, its
residual direction is

`d(l,t) = normalize(T_l^T u_t)`,

where `T_l` is the lens's effective, shrinkage-aware J transport. This uses the
same geometry as the active lens rather than a direct unembedding fallback.

For a source phrase, the token directions are orthonormalized with a compact
SVD to form `Q_A`. This avoids removing the same direction repeatedly when a
phrase contains duplicated or highly correlated subtokens. Target directions
form `B`; their ordering is retained.

## Residual transforms

At a selected layer, source coefficients are `c = Q_A^T h`. Suppression uses

`h' = h + alpha Q_A c`,

with `alpha = -min(g, 1)`. The bound prevents concept inversion at excessive
strength.

Replacement uses

`h' = h + alpha Q_A c + beta B M c`.

`M` is a deterministic linear-position alignment from source subtokens to
target subtokens. It uses interpolation when the sequence lengths differ and
normalizes every source column, so replacing a two-token phrase with a
four-token phrase does not quadruple the residual norm. `beta = g f`, where
`f` is the rule's replacement factor. Applying multiple rules composes their
transforms in visible stack order.

Injection has no source coefficient. It uses the phrase target's normalized
J-space centroid and an amplitude relative to the measured residual RMS. This
extends the existing additive J-space injection to all target subtokens; it
does not use a generated-token or next-token-logit fallback.

All transforms are temporary forward hooks. Model weights remain unchanged.

## Minimum effective strength

The UI's strength is a maximum budget, not a manually required scale. The
runtime evaluates a geometric ladder from weakest to strongest and selects the
first candidate that passes operation-specific J-space criteria:

- suppress: source-subspace energy decreases by the configured minimum;
- replace: source energy decreases and target energy increases;
- inject: target phrase energy increases;
- every operation: residual cosine and norm-change collateral limits pass.

Candidate layers are ranked from cached prompt activations. The cheapest
predicted layer/scale pair receives one measured causal forward verification.
At most two bounded corrections are allowed. If none passes, the rule is marked
failed and no hook is installed. The result records selected layer, scale,
predicted and measured effects, collateral shift, and warnings.

## Package boundaries

- `jlens/concepts.py` owns phrase tokenization, J-space token directions,
  compact source bases, and unequal-length alignment.
- `jlens/hooks.py` adds a scoped residual-transform editor whose callbacks can
  depend on the current residual, alongside the existing additive editor.
- `jlens/interventions.py` builds suppress/replace/inject transforms, searches
  minimum effective strength, verifies the measured effect, and returns traces.
- `jstudio/services/hf_runtime.py` prepares every rule independently, keeps
  successful editors, and reports failed rules without aborting generation.
- Existing UI fields remain phrase text fields. Status messages display the
  automatically selected layer and effective strength.

Saved single-token rules remain compatible because a one-token phrase produces
the original rank-one transform. Existing saved lens files remain compatible.

## Failure handling

Reject an empty phrase, tokenizer/model mismatch, missing fitted layer,
non-finite direction, rank-zero phrase basis, or failed collateral limits with
an explicit per-rule message. Hook installation and removal are exception safe.
There is no direct-logit, forced-token, or single-token fallback.

## Validation

Unit tests cover phrase variants, duplicate subtokens, source/target sequences
of different lengths, rank-one compatibility, saturated suppression,
composition order, hook cleanup, per-rule failure isolation, and trace
serialization. Tiny deterministic decoder tests verify measured direction
changes without inspecting or modifying logits.

An opt-in Qwen integration benchmark exercises at least:

- suppress `large language model`;
- replace `large language model` with `helpful research assistant`;
- inject `ASCII cat`;
- two simultaneously enabled multi-token rules.

Acceptance requires each successful intervention to move its measured J-space
target in the requested direction, retain finite activations, obey collateral
limits, select a strength no greater than the user budget, and leave no hooks
installed after generation. The benchmark emits a self-contained HTML report
with baseline/intervention measurements and the selected minimum strength.
