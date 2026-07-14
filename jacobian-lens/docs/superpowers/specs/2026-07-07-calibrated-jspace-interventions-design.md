# Calibrated J-Space Interventions

## Objective

Extend `jlens` from a read-only Jacobian-lens reference implementation into a
model-agnostic toolkit that can read, inject, suppress, and replace concepts in
decoder-only causal language models. Interventions must use the smallest
measured perturbation that satisfies an explicit success criterion and must
produce inspectable traces suitable for a future UI.

The first release targets Hugging Face decoder-only causal language models.
Single-token concepts are the reliable path. Arbitrary words and short phrases
are accepted through a multi-token experimental path. Multimodal,
encoder-decoder, and remote inference runtimes are outside this release.

## Mathematical model

For layer `l`, let `J_l` be the fitted average Jacobian and let `u_t(z)` be the
token `t` logit produced by the model's complete unembedding path, including
its final normalization. At activation `h`, the local J-lens score covector is

```text
a_(l,t)(h) = J_l^T grad_z u_t(z) evaluated at z = J_l h.
```

This is the exact first-order derivative of the score actually returned by
`model.unembed(J_l h)`. It improves on treating normalization as an unknown
scalar and automatically handles RMSNorm, LayerNorm, tied embeddings, and
other adapter-supported unembedding paths. The current implementation only
evaluates `J_l h`; the extension exposes these local covectors and treats an
intervention as a residual perturbation `delta` at a selected layer and
position. Covectors are recomputed after materially changing `h`, so the
constrained solve is a sequential local approximation rather than a single
stale linearization.

### Calibrated geometry

Lens fitting will optionally accumulate the residual mean and a regularized
covariance model at each fitted layer. To keep storage and fitting practical,
the covariance is represented as a diagonal plus a bounded-rank randomized
sketch. Shrinkage toward the isotropic covariance guarantees positive
definiteness:

```text
Sigma_l = (1 - lambda) Sigma_hat_l
          + lambda trace(Sigma_hat_l) / d * I.
```

The intervention cost is the Mahalanobis norm
`delta^T Sigma_l^-1 delta`, not raw Euclidean length. This penalizes movement
in low-variance, off-distribution directions and makes strengths more
comparable between layers and model families. Existing lens files remain
loadable and use identity covariance with an explicit uncalibrated status.

### Injection

For a target token `t`, competitors `C`, and requested score margin `m`, solve

```text
minimize    delta^T Sigma_l^-1 delta
subject to  (a_(l,t) - a_(l,c))^T (h + delta) >= m, for c in C.
```

The active constraints are solved in their small Gram system rather than in
the full residual dimension. Tikhonov regularization handles nearly collinear
token directions. Competitors default to the strongest baseline J-lens tokens,
which directly asks for the target to enter a requested top-k set.

### Replacement

Replacing source token `s` with target token `t` uses joint constraints:

- raise `t` to at least the source's baseline calibrated score;
- lower `s` by the corresponding margin;
- bound changes to a configurable preservation set of unrelated top concepts.

This is a minimum-cost constrained projection, not a fixed pseudoinverse swap.
It reduces to Anthropic's two-coordinate swap in the unregularized Euclidean
two-vector case, while behaving predictably for non-orthogonal directions.

### Multi-token concepts

Text is tokenized using both bare and leading-space variants. A single-token
variant is preferred. Otherwise, the intervention solves joint constraints for
the ordered token set, with weights normalized by direction scale. Success is
verified against the sequence log-probability under teacher forcing and, when
requested, actual generation. Because the original J-lens is token-indexed,
multi-token results are labeled experimental in every result and trace.

### Minimum effective strength

The constrained solution gives a direction and nominal magnitude under the
linearized lens, but downstream transformer behavior is nonlinear. Therefore,
the runtime verifies candidate interventions with real forward passes:

1. Test strength zero and the nominal constrained solution.
2. Exponentially bracket the first passing strength when the nominal solution
   is insufficient.
3. Bisect the passing interval to a configured relative tolerance.
4. Search a bounded set of eligible workspace layers and positions.
5. Return the passing candidate with minimum normalized intervention cost.

Default success for injection is target entry into the requested J-lens top-k
at the intervention site plus a positive downstream target-logit shift. Default
success for replacement additionally requires the target to outrank the source.
Callers can select stricter output-token or generated-sequence criteria.

## Components and interfaces

### Lens geometry

`LensGeometry` owns covariance statistics, conditioning diagnostics, and small
constrained solves. A thin score-linearization function obtains exact local
token covectors from the model's unembedding path; it does not require a full
transformer forward pass. The solver itself has no model hooks and is
unit-testable using synthetic matrices and score functions.

`JacobianLens` gains backward-compatible optional calibration metadata and
delegates geometry construction to `LensGeometry`. Save files receive an
explicit format version and model/tokenizer identity metadata. Loading rejects
shape mismatches and warns, rather than silently pretending, when calibration
or identity checks are unavailable.

### Concept resolution

`ConceptResolver` converts user text or token IDs into a `ConceptSpec`.
Resolution records tokenization variants, selected IDs, display strings, and
whether the concept is single-token or experimental multi-token. Ambiguous
tokenizations are visible in results instead of being silently collapsed.

### Residual mutation

The generic model protocol gains a scoped residual-mutation context. The hook
supports tensor outputs and tuple-style Hugging Face block outputs, preserves
all non-residual return values, validates layer and position bounds, and always
removes handles on exit. Intervention code depends on this protocol rather
than concrete Qwen, Gemma, or Llama classes.

### Intervention engine

`InterventionEngine` provides `inject(...)`, `suppress(...)`, and
`replace(...)`. Requests specify concepts and may override candidate layers,
positions, top-k, margin, preservation set, and success criterion. Defaults are
derived from fitted layers, skip noisy early layers, use the final prompt
position, and enforce bounded search budgets.

Every call returns an `InterventionResult`; unsuccessful searches return
diagnostics and the best observed candidate rather than an arbitrary strong
perturbation. User cancellation and numerical failures remove hooks and leave
the model unchanged.

### Inspection data

`InterventionTrace` is a serializable, UI-neutral record containing:

- model, lens, tokenizer, prompt, and concept metadata;
- baseline and intervened J-lens rankings;
- candidate layer, position, strength, margin, and normalized cost;
- calibration search points and pass/fail reasons;
- source/target score trajectories and preservation drift;
- downstream logit or sequence-probability changes;
- warnings for uncalibrated geometry or multi-token concepts.

The existing slice renderer receives trace overlays and JSON export. The first
release improves inspection rather than introducing a new application UI.

## Generalization strategy

Architecture handling stays in `HFLensModel` layout discovery and the generic
protocol. Core math only assumes a residual width, layer sequence, tokenizer,
forward pass, unembedding, and residual mutation capability.

Compatibility is validated at three levels:

1. deterministic tiny-model tests for exact constrained-solver properties;
2. mocked Hugging Face layouts for hook and output-shape compatibility;
3. optional slow smoke tests on one small Llama-family, Qwen-family, and
   Gemma-family decoder model.

Unsupported layouts fail during adapter construction with a capability report.
They never fail halfway through an intervention search.

## Testing and acceptance criteria

Mathematical tests verify:

- closed-form single-constraint solutions;
- KKT feasibility and complementary slackness for active-set solutions;
- Mahalanobis cost is no worse than fixed-direction baselines at equal margin;
- replacement preserves designated unrelated coordinates within tolerance;
- regularization remains finite for duplicate and collinear token directions;
- bisection returns the lowest passing strength within tolerance.

Integration tests verify:

- hooks alter only selected layers, positions, and batch elements;
- hooks are removed after success, failure, and exceptions;
- legacy lens files load and are explicitly marked uncalibrated;
- calibrated lens save/load preserves numerical results;
- single-token injection and replacement work end-to-end on the tiny decoder;
- traces round-trip through JSON and the existing viewer can consume them;
- all original fitting, application, and visualization tests continue to pass.

The implementation is accepted when the full CPU test suite passes and the
optional available-model smoke test demonstrates that calibrated search finds
a weaker passing intervention than a coarse fixed-strength sweep on at least
one real decoder model. A real-model miss is reported as an empirical result,
not hidden by increasing strength without bound.

## Delivery sequence

1. Add mathematical geometry and solver tests.
2. Add safe residual-mutation hooks and protocol changes.
3. Implement concept resolution and intervention requests/results.
4. Implement nonlinear calibration search and traces.
5. Extend fitting and lens serialization with covariance calibration.
6. Add viewer trace export/overlays and documentation.
7. Run CPU regression tests and available real-model smoke tests.
