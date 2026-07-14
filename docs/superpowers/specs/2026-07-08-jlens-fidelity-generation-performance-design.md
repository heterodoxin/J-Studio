# J-Lens Fidelity and Generation Performance Design

**Date:** 2026-07-08
**Status:** Approved

## Objective

Replace J Studio's approximate native J-Lens workspace with the repository's
original interactive slice visualization, remove score saturation, and raise
Qwen3-8B chat generation from the measured 20.5 tokens/s toward a required
40 tokens/s and preferred 100 tokens/s on the local 32 GiB ROCm GPU.

The visualization must remain an analysis surface for actual Jacobian-lens
readouts. It must not substitute next-token likelihoods or synthetic concepts.

## Evidence and Constraints

The current runtime was benchmarked with 128 generated tokens:

| Runtime | Result | Decision |
| --- | ---: | --- |
| Transformers, eager attention | 20.5 decode tok/s | Too slow |
| Transformers, SDPA | 22.0 decode tok/s | Insufficient improvement |
| vLLM BF16, eager ROCm mode | 36.0 overall tok/s | Stable fallback, below target |
| vLLM BF16, compiled graph mode | HIP graph replay crash | Do not use |
| vLLM bitsandbytes 4-bit | 25.5 overall tok/s | ROCm kernels are slower here |
| vLLM with Qwen2.5-1.5B draft | 24.9 overall tok/s | Low 8.6% draft acceptance |

The current J-Lens presentation also applies a sigmoid to vocabulary z-scores.
Large positive z-scores therefore collapse to `+0.99` or `+1.00`, destroying
useful differences. The native matrix is only a rough recreation of
`jlens.vis` and omits much of its information density and interaction model.

## Selected Architecture

### 1. Exact repository visualization

J Studio will embed the HTML produced by `jlens.vis.build_page` in a
`QWebEngineView`. The existing `slice_vis.html` implementation remains the
authoritative renderer. J Studio will not independently recreate its matrix,
heatmap, rank charts, tooltips, or scrubbing behavior in Qt.

The embedded surface includes:

- the layer-by-position top-token matrix;
- whitespace display controls and keyboard/mouse scrubbing;
- spatially preserved prompt and output text;
- synchronized By Layer and By Position readouts;
- pinned-token rank heatmap;
- rank-by-layer and rank-by-position plots;
- full-vocabulary rank tooltips and selected-cell details.

A narrow native toolbar surrounds the page for run selection, refresh,
inspection scope, export, and intervention actions. It must not compete with
or duplicate controls already supplied by the original page.

### 2. Slice data service

`HFModelRuntime` will expose an asynchronous slice operation backed by
`jlens.vis.compute_slice`. It receives a prompt or completed run plus options
such as layer stride, token window, top-K, and meaningful-token filtering. It
returns the self-contained HTML page and structured selection metadata.

Slice computation runs outside the GUI thread. A request has a stable run ID
and generation number; stale results are discarded when the user changes runs.
The last successful page stays visible under a non-destructive loading state.

The default view uses `mask_display=True`, keeps full-vocabulary ranks, and
includes the final model layer as the `J = I` reference row. No sigmoid score
is used in the visualization. Native summaries show rank, raw logit when
available, and an explicitly labeled within-cell standardized score only when
that statistic is useful.

### 3. Web-to-native bridge

A small `QWebChannel` bridge carries selection and action events from the
embedded page into J Studio:

- selected position and layer;
- pinned or unpinned token;
- inspect token details;
- open Inject, Replace, or Suppress editor at the selected coordinate;
- export the current selection.

The original page remains usable if the bridge is unavailable. Bridge messages
are schema-validated and cannot execute arbitrary Python or filesystem actions.

### 4. Dual generation and analysis backends

The BF16 Transformers model remains authoritative for J-space computation,
because hooks and residual activations are required by the Jacobian lens.

Chat generation uses a separate llama.cpp ROCm backend loaded from a local
Q4_K_M GGUF conversion of the same Qwen3-8B checkpoint. This reduces generation
weight bandwidth enough to make 40--100 tokens/s plausible while keeping the
BF16 analysis model available on the 32 GiB GPU. The generated text can then be
passed to the BF16 model for J-Lens inspection.

Two explicit generation modes are exposed:

- **Fast** (default): llama.cpp ROCm Q4_K_M generation.
- **Exact BF16:** the existing Transformers path for output comparisons and
  cases where quantization variance is unacceptable.

The UI displays backend, quantization, time to first token, and measured
tokens/s for every run. It never implies that quantized generation is
bit-identical to BF16 generation.

The GGUF is produced locally from the cached checkpoint. It is a generated
artifact outside Git and is reused on later launches. Startup probes the
llama.cpp backend and falls back to Exact BF16 with a visible reason if the
binary, model artifact, or ROCm device is unavailable.

## Data Flow

### Chat generation

1. Chat submits the rendered conversation and selected generation mode.
2. Fast mode streams tokens from the llama.cpp worker; Exact mode streams from
   `HFModelRuntime`.
3. The generation service records token timestamps and publishes live and final
   throughput metrics.
4. The immutable run stores prompt, output, backend provenance, and timing.
5. Inspect with J-Lens sends the relevant text to the BF16 slice service.

### J-Lens inspection

1. The workspace selects a run or prompt and requests a slice.
2. The slice worker records residual activations and applies the fitted lens.
3. `compute_slice` creates top-token and tracked-rank arrays.
4. `build_page` creates the original interactive HTML page.
5. The GUI loads the page and reconnects native actions through the bridge.

## Resource Management

The generation worker and BF16 analysis runtime report allocated VRAM at
startup. Fast generation uses a bounded context and KV cache sized so the BF16
model, lens working buffers, and the Q4 model fit concurrently. If a real slice
request would exceed the configured safety margin, J Studio temporarily pauses
the generation worker's GPU activity rather than allowing an out-of-memory
failure.

Workers have explicit startup, ready, failed, and stopped states. Closing J
Studio terminates the generation subprocess and releases both model runtimes.

## Error Handling

- A failed slice leaves the previous visualization visible and shows a retry
  action with the precise backend error.
- A stale slice result cannot replace a newer run selection.
- Web content is generated locally; navigation to arbitrary remote URLs is
  blocked.
- Malformed bridge events are ignored and logged.
- Fast-backend failure falls back to Exact BF16 without losing the prompt.
- Throughput below 40 tokens/s is reported as a performance warning, not hidden.
- ROCm graph execution remains disabled because it crashed during validation.

## Testing and Acceptance

### Visualization

- Contract tests verify that the J-Lens tab hosts the original renderer and no
  longer instantiates the approximate native matrix as its primary surface.
- Slice fixtures verify matrix dimensions, final-layer inclusion, tracked ranks,
  meaningful-token masking, and preserved ASCII whitespace.
- Browser interaction tests cover selection, scrubbing, pinning, heatmap/plot
  synchronization, and native intervention bridge events.
- Score tests prove that distinct large z-scores do not collapse to identical
  displayed `+0.99/+1.00` values.
- A visual regression capture is compared against the repository's
  `assets/slice_vis.png` structure at the supported desktop size.

### Generation

- Protocol tests cover streaming, cancellation, process failure, and fallback.
- Provenance tests distinguish Fast Q4 and Exact BF16 runs.
- A local benchmark generates at least 256 tokens after warmup and records TTFT
  and decode throughput.
- Acceptance target: Fast mode sustains at least 40 tokens/s on the current ROCm
  host. The preferred target is 100 tokens/s, but correctness and stability take
  precedence over reaching it through unsafe graph execution.
- Exact BF16 remains behaviorally compatible with the current runtime.

## Non-Goals

- Making one fitted Jacobian artifact valid for unrelated model checkpoints.
- Replacing the repository's visualization design with another dashboard.
- Calling vanilla next-token predictions J-space.
- Shipping generated GGUF or lens artifacts in Git.
- Enabling unstable HIP graph replay to inflate benchmark results.
