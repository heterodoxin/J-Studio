# Progressive Fast Jacobian-Lens Fitting

**Date:** 2026-07-08
**Status:** Approved direction

## Objective

Produce a useful Jacobian lens for a 7B-class decoder in under 90 seconds on
the local 32 GiB ROCm GPU, while exposing a preliminary readout within 15
seconds. Quality, not elapsed time alone, determines when the lens is labeled
stable. Manual fitting commands must not be required by J Studio.

## Correct estimator

For prompt Jacobians `J_p`, define `K_p = J_p - I`. Probe blocks and prompt
samples are independent estimator axes:

```text
K_hat = (1 / B) sum_b (1 / r) sum_i z_bi
          [ (1 / N) sum_p K_p^T z_bi ]^T.
```

Each probe block is evaluated on every prompt assigned to that refinement
stage. This corrects the existing implementation, which assigns different
probe blocks to different prompts and therefore mixes prompt variance with
projection variance. Orthogonal signed DCT probes provide deterministic,
well-conditioned blocks; independent permutations provide additional blocks.

The lens targets the penultimate transformer block by default, matching the
paper's cleaner recommended readout. The final model layer remains the `J=I`
reference in visualizations.

## Progressive stages

### Preview

- Eight pretraining-like prompts, 32 probes, 64-token sequences.
- All requested source layers captured in the same backward passes.
- Target: available within 15 seconds on the reference 7B/8B ROCm host.
- Clearly labeled `Preview`; interventions remain disarmed.

### Stable

- At least 32 prompts and two independent 64-probe blocks.
- 128-token sequences when memory permits.
- Target: pass quality gates and complete within 90 seconds.
- J-Lens inspection and interventions become available only after validation.

### Refined

- Additional prompts/probe blocks run at low priority while J Studio is idle.
- Refinement stops when held-out metrics and successive transports converge.
- A user may stop refinement without invalidating the last stable lens.

## Performance architecture

The fitter uses one model forward graph per prompt and batched vector-Jacobian
products through `torch.autograd.grad(..., is_grads_batched=True)`. This avoids
replicating the prompt batch for every cotangent. A compatibility fallback uses
the existing replicated-batch method if an operator lacks a vmap rule.

An autotuner probes VJP batch sizes `1, 2, 4, 8, 16` against available VRAM,
selecting the fastest size below a 90% allocation ceiling. It caches the choice
by model fingerprint and sequence-length bucket. Prompt tokenization is done
once. CPU transfers use pinned buffers, and accumulated sufficient statistics
remain in fp32.

Checkpoints store probe definitions, per-probe prompt means, counts, validation
metrics, and model/tokenizer fingerprints. They do not duplicate per-prompt
correction tensors. Writes are atomic and resumable.

## Quality gates

A stage is `Stable` only when all applicable checks pass:

1. J-Lens top-k intermediate recovery beats the vanilla logit-lens baseline on
   held-out synthetic multi-hop and structured-input prompts.
2. Successive refinement stages meet a configured top-10 rank-overlap target.
3. Transport outputs remain finite and normalization diagnostics stay bounded.
4. Fitted layers show no catastrophic prompt outlier or correction-norm spike.
5. A small held-out causal check confirms that a top J-Lens direction has a
   measurable downstream effect.

The UI reports failed gates and continues refinement. It never labels a lens
stable merely because a timer expired.

## J Studio integration

When a compatible cached stable lens exists, it loads immediately. Otherwise,
model loading starts Preview fitting automatically and reports prompt/probe
progress. Preview results may be inspected, but their provenance and quality
state remain visible. Stable completion atomically replaces the active lens.

Cache identity includes model repository/path, revision or weight fingerprint,
tokenizer fingerprint, residual width, layer layout, target layer, estimator
version, and probe seed. Any mismatch starts a new fit rather than silently
reusing incompatible geometry.

## Resource and cancellation behavior

Fitting yields GPU ownership to active generation and can resume between prompt
boundaries. Cancellation writes the current sufficient statistics and removes
all hooks. Out-of-memory during autotuning lowers the VJP batch size; OOM at
batch size one fails with an explicit diagnostic rather than moving weights to
CPU silently.

J Studio will not terminate unrelated GPU workloads. If insufficient VRAM is
available, fitting waits and displays the blocking allocation.

## Acceptance criteria

- Preview under 15 seconds and Stable under 90 seconds for the local Qwen3-8B
  checkpoint when the reference GPU is otherwise available.
- Stable passes every quality gate and materially outperforms logit lens on the
  held-out intermediate-recovery set.
- Batched VJP numerically matches the replicated reference on deterministic
  tiny decoders.
- Resume produces the same sufficient statistics as an uninterrupted run.
- Unsupported batched-autograd operators fall back without changing results.
- Existing dense fitting, slice visualization, and intervention tests pass.

