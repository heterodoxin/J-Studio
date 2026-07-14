# Progressive Fast Jacobian-Lens Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Produce a quality-gated Qwen3-8B preview lens in under 15 seconds and a stable lens in under 90 seconds on the reference ROCm GPU.

**Architecture:** Replace the prompt/probe-confounded sketch with a two-axis estimator that averages every fixed probe block across prompts before combining independent blocks. Use memory-efficient batched VJPs, progressive stages, compact resumable statistics, and held-out quality gates.

**Tech Stack:** Python 3.11, PyTorch 2.12 ROCm, Transformers, pytest

## Global Constraints

- Preview target: under 15 seconds; Stable target: under 90 seconds on an otherwise available reference GPU.
- Stable is quality-gated and must beat vanilla logit lens on held-out intermediate recovery.
- Penultimate transformer block is the default Jacobian target.
- Accumulation and validation statistics remain fp32.
- Existing dense fitting, lens loading, visualization, and interventions remain compatible.
- Unrelated GPU processes are never terminated.

---

### Task 1: Correct Two-Axis Sketch Statistics

**Files:**
- Modify: `jlens/fitting.py`
- Modify: `tests/test_fitting.py`

**Interfaces:**
- Produces: `ProbeBlock(probes: Tensor, correction_sum: dict[int, Tensor], n_prompts: int)`
- Produces: `fit_sketch(..., probe_blocks: int = 1)` with checkpoint format `prompt-averaged-probe-blocks-v2`

- [ ] **Step 1: Write the failing estimator test**

```python
def test_probe_blocks_average_every_prompt_before_combining(monkeypatch):
    model = TinyDecoder(n_layers=4, d_model=8)
    seen = []
    real = fitting.sketched_jacobian_for_prompt

    def record(model, prompt, layers, **kwargs):
        seen.append((prompt, kwargs["_probes"].clone()))
        return real(model, prompt, layers, **kwargs)

    monkeypatch.setattr(fitting, "sketched_jacobian_for_prompt", record)
    fit_sketch(model, ["abcdefghij " * 5, "klmnopqrst " * 5],
               source_layers=[2], sketch_rank=4, probe_blocks=2,
               dim_batch=2, max_seq_len=64)
    assert len(seen) == 4
    assert torch.equal(seen[0][1], seen[1][1])
    assert not torch.equal(seen[1][1], seen[2][1])
    assert torch.equal(seen[2][1], seen[3][1])
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_fitting.py::test_probe_blocks_average_every_prompt_before_combining -q`

Expected: FAIL because `probe_blocks` is unsupported and the current implementation assigns one block per prompt.

- [ ] **Step 3: Replace the confounded accumulator**

For each block index, generate one signed-DCT probe matrix, run all prompts with that same matrix, sum corrections by layer, and divide by the successful prompt count. Concatenate block probe matrices and their prompt-mean corrections; `SketchedJacobian.apply` already divides by total rank, yielding the required `1/(B*r)` factor.

```python
for block_index in range(probe_blocks):
    probes = _orthogonal_probes(d_model, sketch_rank, seed, block_index=block_index)
    correction_sum = zeros_by_layer()
    for prompt in prompts:
        estimate = sketched_jacobian_for_prompt(..., _probes=probes)
        correction_sum += estimate.corrections
    blocks.append(ProbeBlock(probes, correction_sum, n_prompts))
```

- [ ] **Step 4: Add compact v2 checkpoint state**

Persist one `correction_sum` and prompt count per probe block, plus current block/prompt indices. Reject the existing `independent-probe-blocks-v1` checkpoint with an actionable incompatibility error rather than misreading it.

- [ ] **Step 5: Run fitting regression tests**

Run: `python -m pytest tests/test_fitting.py -q`

Expected: PASS, including full-rank equivalence and resume determinism.

- [ ] **Step 6: Commit**

```bash
git add jlens/fitting.py tests/test_fitting.py
git commit -m "fix: separate prompt averaging from probe refinement"
```

### Task 2: Memory-Efficient Batched VJPs

**Files:**
- Modify: `jlens/fitting.py`
- Modify: `tests/test_fitting.py`

**Interfaces:**
- Adds: `vjp_backend: Literal["auto", "batched", "replicated"] = "auto"`
- Produces: `_batched_vjp(target, sources, cotangents, retain_graph) -> tuple[Tensor, ...]`

- [ ] **Step 1: Write numerical-equivalence tests**

```python
@pytest.mark.parametrize("dim_batch", [1, 2, 4])
def test_batched_vjp_matches_replicated(dim_batch):
    model = TinyDecoder(n_layers=4, d_model=8)
    kwargs = dict(source_layers=[0, 2], sketch_rank=8,
                  dim_batch=dim_batch, max_seq_len=64, seed=3)
    batched, _, _ = sketched_jacobian_for_prompt(
        model, "abcdefghij " * 5, vjp_backend="batched", **kwargs)
    replicated, _, _ = sketched_jacobian_for_prompt(
        model, "abcdefghij " * 5, vjp_backend="replicated", **kwargs)
    for layer in kwargs["source_layers"]:
        torch.testing.assert_close(batched[layer].corrections,
                                   replicated[layer].corrections,
                                   rtol=2e-5, atol=2e-5)
```

- [ ] **Step 2: Run the tests and verify RED**

Run: `python -m pytest tests/test_fitting.py -k batched_vjp -q`

- [ ] **Step 3: Implement one-forward batched reverse mode**

Run the model once with batch size one. Build cotangents with shape `[vjp_batch, 1, seq, d]` and call:

```python
torch.autograd.grad(
    outputs=target_activation,
    inputs=source_activations,
    grad_outputs=cotangents,
    is_grads_batched=True,
    retain_graph=retain_graph,
)
```

Reduce the returned `[vjp_batch, 1, seq, d]` tensors over valid source positions. Keep the existing replicated implementation in a private fallback function.

- [ ] **Step 4: Add deterministic fallback coverage**

```python
def test_auto_vjp_falls_back_to_replicated(monkeypatch):
    monkeypatch.setattr(fitting, "_batched_vjp",
                        lambda *a, **k: (_ for _ in ()).throw(
                            RuntimeError("Batching rule not implemented")))
    result, _, _ = sketched_jacobian_for_prompt(
        TinyDecoder(), "abcdefghij " * 5, [1], sketch_rank=4,
        dim_batch=2, max_seq_len=64, vjp_backend="auto")
    assert result[1].rank == 4
```

Auto fallback must restart the prompt computation through the replicated path because a failed autograd call may invalidate the retained graph.

- [ ] **Step 5: Run full CPU tests and commit**

Run: `python -m pytest -q`

```bash
git add jlens/fitting.py tests/test_fitting.py
git commit -m "perf: vectorize Jacobian probe VJPs"
```

### Task 3: Quality Evaluation and Progressive Stages

**Files:**
- Create: `jlens/evaluation.py`
- Create: `jlens/progressive.py`
- Modify: `jlens/__init__.py`
- Create: `tests/test_progressive.py`

**Interfaces:**
- Produces: `FitStage(name, prompts, sketch_rank, probe_blocks, max_seq_len)`
- Produces: `FitQuality(pass_at_10, baseline_pass_at_10, rank_overlap, finite, stable, reasons)`
- Produces: `fit_progressive(model, prompts, validation_items, stages, ..., on_stage=None) -> ProgressiveFitResult`

- [ ] **Step 1: Write failing quality-gate tests**

```python
def test_stable_requires_beating_logit_baseline():
    quality = FitQuality(pass_at_10=.4, baseline_pass_at_10=.5,
                         rank_overlap=.9, finite=True)
    assert not quality.stable
    assert "logit" in " ".join(quality.reasons).lower()

def test_progressive_callback_receives_preview_then_stable(tiny_progressive_inputs):
    seen = []
    result = fit_progressive(*tiny_progressive_inputs,
                             stages=(FitStage("Preview", 2, 4, 1, 32),
                                     FitStage("Stable", 3, 4, 2, 64)),
                             evaluator=lambda lens, previous: FitQuality(
                                 .8, .2, 1.0, True),
                             on_stage=lambda stage: seen.append(stage.name))
    assert seen == ["Preview", "Stable"]
    assert result.active.name == "Stable"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_progressive.py -q`

- [ ] **Step 3: Implement held-out rank metrics**

Resolve each intermediate using bare and leading-space tokenizer variants. At the requested position, compute its minimum full-vocabulary rank over fitted workspace layers for both J-Lens and `use_jacobian=False`. Report pass@10 and top-10 overlap against the prior stage.

- [ ] **Step 4: Implement progressive orchestration**

Default stages are Preview `(8, 32, 1, 64)` and Stable `(32, 64, 2, 128)`. Emit immutable stage results containing elapsed time, lens, quality, and checkpoint path. Stable failure retains Preview as active and reports reasons.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_progressive.py tests/test_fitting.py -q`

```bash
git add jlens/evaluation.py jlens/progressive.py jlens/__init__.py tests/test_progressive.py
git commit -m "feat: add quality-gated progressive lens fitting"
```

### Task 4: ROCm Autotuning and Production CLI

**Files:**
- Create: `jlens/autotune.py`
- Modify: `scripts/fit_decoder_lens.py`
- Create: `tests/test_autotune.py`
- Create: `tests/test_fit_cli.py`

**Interfaces:**
- Produces: `autotune_vjp_batch(model, prompt, layers, candidates=(1,2,4,8,16), memory_fraction=.9) -> AutotuneResult`
- CLI modes: `--stage preview|stable|refined`, `--quality-json PATH`, `--progress-json PATH`

- [ ] **Step 1: Write failing autotuner tests**

```python
def test_autotuner_selects_fastest_safe_candidate():
    runner = FakeRunner({1: (4.0, .4), 2: (2.5, .6), 4: (1.8, .85), 8: (1.2, .96)})
    result = choose_batch((1, 2, 4, 8), runner, memory_fraction=.9)
    assert result.batch_size == 4

def test_autotuner_recovers_from_oom():
    runner = FakeRunner({1: (4.0, .4), 2: torch.OutOfMemoryError()})
    assert choose_batch((1, 2), runner, memory_fraction=.9).batch_size == 1
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_autotune.py tests/test_fit_cli.py -q`

- [ ] **Step 3: Implement isolated tuning**

Benchmark a four-probe slice per candidate, synchronize ROCm around timing, reset peak-memory stats, and stop after the first OOM. Cache results in `~/.cache/jlens/autotune.json` under a model/layout/sequence fingerprint.

- [ ] **Step 4: Rewrite CLI around progressive fitting**

Default target is `n_layers - 2`; choose up to 25 evenly spaced source layers. Write atomic progress JSON after every prompt and quality JSON after every stage. Existing explicit `--rank`, `--prompts`, and `--dim-batch` flags remain available as overrides.

- [ ] **Step 5: Run CLI tests and commit**

Run: `python -m pytest tests/test_autotune.py tests/test_fit_cli.py -q`

```bash
git add jlens/autotune.py scripts/fit_decoder_lens.py tests/test_autotune.py tests/test_fit_cli.py
git commit -m "feat: autotune progressive ROCm lens fitting"
```

### Task 5: Real Qwen Benchmark and Acceptance

**Files:**
- Create: `scripts/benchmark_progressive_fit.py`
- Create: `tests/test_benchmark_progressive_fit.py`
- Modify: `README.md`

**Interfaces:**
- Produces JSON with stage elapsed time, VJPs/s, peak VRAM, pass@10, baseline pass@10, overlap, and stable status.

- [ ] **Step 1: Add benchmark argument and output smoke test**

```python
def test_benchmark_writes_required_metrics(tmp_path, monkeypatch):
    monkeypatch.setattr(benchmark, "run_benchmark", lambda args: {
        "schema": 1, "preview_seconds": 10.0, "stable_seconds": 70.0,
        "vjps_per_second": 100.0, "peak_vram_gib": 18.0,
        "pass_at_10": .7, "baseline_pass_at_10": .3,
        "rank_overlap": .9, "stable": True,
    })
    output = tmp_path / "result.json"
    assert benchmark.main(["--model", "test/model", "--stage", "stable",
                           "--output", str(output)]) == 0
    assert set(json.loads(output.read_text())) >= {
        "schema", "preview_seconds", "stable_seconds", "vjps_per_second",
        "peak_vram_gib", "pass_at_10", "baseline_pass_at_10",
        "rank_overlap", "stable",
    }
```

- [ ] **Step 2: Run full automated verification**

Run: `python -m pytest -q`

Run: `python -m ruff check jlens tests scripts`

Expected: all tests and lint pass.

- [ ] **Step 3: Wait for sufficient free VRAM without terminating other work**

Require at least 20 GiB free on the model's ROCm device. If unavailable, print the owning PIDs and exit with status `75` so the benchmark can be retried safely.

- [ ] **Step 4: Run the real acceptance benchmark**

```bash
python scripts/benchmark_progressive_fit.py \
  --model heterodoxin/qwen3-8b-apostate \
  --stage stable \
  --output /tmp/jstudio-progressive-fit.json
```

Acceptance requires Preview `<15s`, Stable `<90s`, `stable=true`, finite outputs, and J-Lens pass@10 above logit-lens pass@10. If time passes but quality fails, optimize or refine; never weaken the gate silently.

- [ ] **Step 5: Document measured results and commit**

```bash
git add scripts/benchmark_progressive_fit.py tests/test_benchmark_progressive_fit.py README.md
git commit -m "test: benchmark progressive fitting on ROCm"
```
