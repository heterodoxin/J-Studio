# Calibrated J-Space Interventions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model-agnostic, minimum-disturbance J-space concept injection, suppression, and replacement with calibrated geometry and inspectable traces.

**Architecture:** A standalone geometry module solves covariance-aware constrained projections. Scoped residual-edit hooks apply those projections through the existing `LensModel` abstraction, while an intervention engine resolves concepts, searches the smallest passing strength, and records a serializable trace. `JacobianLens` serialization remains backward compatible and optionally stores calibrated residual geometry.

**Tech Stack:** Python 3.10+, PyTorch, Hugging Face Transformers, pytest, dataclasses, JSON

## Global Constraints

- First-release model scope is Hugging Face decoder-only causal language models.
- Existing lens files must load and be explicitly marked uncalibrated.
- Single-token concepts are reliable; multi-token concepts are experimental and visibly labeled.
- Intervention success must be verified by real forward passes and bounded search.
- Existing fitting, application, and visualization behavior must remain compatible.
- No model-family-specific intervention math is allowed outside the Hugging Face adapter.

---

## File structure

- Create `jlens/geometry.py`: covariance representation, online covariance sketch, local score linearization, constrained minimum-cost solver.
- Create `jlens/interventions.py`: concept resolution, residual-edit requests, intervention search, result and trace types.
- Modify `jlens/hooks.py`: safe scoped residual mutation hook.
- Modify `jlens/lens.py`: optional geometry metadata and backward-compatible serialization.
- Modify `jlens/fitting.py`: optional post-fit residual covariance calibration.
- Modify `jlens/protocol.py`: document mutation-compatible residual block contract.
- Modify `jlens/__init__.py`: export the public intervention API.
- Modify `jlens/vis.py`: write intervention traces as viewer sidecars.
- Modify `README.md`: document injection/replacement examples and limitations.
- Create `tests/test_geometry.py`, `tests/test_intervention_hooks.py`, `tests/test_interventions.py`, and `tests/test_lens_geometry_io.py`.

### Task 1: Covariance-aware constrained geometry

**Files:**
- Create: `jlens/geometry.py`
- Create: `tests/test_geometry.py`

**Interfaces:**
- Produces: `CovarianceMetric(diagonal, factors=None, calibrated=True)`
- Produces: `CovarianceMetric.apply(vector) -> Tensor`
- Produces: `minimum_cost_perturbation(constraints, deficits, metric, *, tolerance=1e-6, max_iterations=2000) -> ProjectionSolution`
- Produces: `minimum_passing_scale(predicate, *, initial=1.0, maximum=16.0, relative_tolerance=0.01) -> ScaleSearchResult`

- [ ] **Step 1: Write solver and scale-search tests**

```python
def test_single_constraint_matches_closed_form():
    sigma = torch.tensor([[2.0, 0.3], [0.3, 1.0]])
    metric = CovarianceMetric.from_dense(sigma)
    c = torch.tensor([[1.0, -0.5]])
    b = torch.tensor([0.7])
    got = minimum_cost_perturbation(c, b, metric).delta
    expected = (b[0] / (c @ sigma @ c.T)[0, 0]) * (sigma @ c[0])
    torch.testing.assert_close(got, expected, atol=1e-5, rtol=1e-5)

def test_collinear_constraints_stay_finite_and_feasible():
    c = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    result = minimum_cost_perturbation(c, torch.tensor([1.0, 2.0]), CovarianceMetric.identity(2))
    assert torch.isfinite(result.delta).all()
    assert torch.all(c @ result.delta >= torch.tensor([1.0, 2.0]) - 1e-5)

def test_scale_search_returns_lower_boundary():
    result = minimum_passing_scale(lambda x: x >= 0.37, relative_tolerance=1e-3)
    assert result.passed and 0.37 <= result.scale <= 0.371
```

- [ ] **Step 2: Run tests and verify they fail because `jlens.geometry` is absent**

Run: `.venv/bin/python -m pytest tests/test_geometry.py -q`
Expected: collection error containing `No module named 'jlens.geometry'`.

- [ ] **Step 3: Implement covariance operators and the constrained dual solve**

```python
@dataclass(frozen=True)
class CovarianceMetric:
    diagonal: torch.Tensor
    factors: torch.Tensor | None = None
    calibrated: bool = True

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        result = value * self.diagonal
        if self.factors is not None and self.factors.numel():
            result = result + (value @ self.factors.T) @ self.factors
        return result

def minimum_cost_perturbation(constraints, deficits, metric, *, tolerance=1e-6, max_iterations=2000):
    sigma_ct = metric.apply(constraints)
    gram = constraints @ sigma_ct.T
    dual = torch.zeros_like(deficits)
    for _ in range(max_iterations):
        previous = dual.clone()
        for i in range(len(dual)):
            residual = deficits[i] - gram[i] @ dual
            dual[i] = torch.clamp(dual[i] + residual / gram[i, i].clamp_min(1e-12), min=0)
        if (dual - previous).abs().max() <= tolerance:
            break
    delta = dual @ sigma_ct
    cost = float(dual @ (gram @ dual))
    violation = torch.clamp(deficits - constraints @ delta, min=0)
    return ProjectionSolution(
        delta=delta, dual=dual, cost=cost,
        feasible=bool(violation.max() <= tolerance), iterations=iterations,
        max_violation=float(violation.max()),
    )
```

Use fp64 internally for the Gram solve, validate shapes/finiteness, retain
negative-deficit preservation constraints because they can become active after
other coordinates move, and report feasibility, iterations, maximum violation,
and Mahalanobis cost. Return
`ProjectionSolution(delta=delta, dual=dual, cost=cost, feasible=feasible,
iterations=iterations, max_violation=max_violation)`.

- [ ] **Step 4: Implement bounded bracketing and bisection**

Test zero first, bracket by doubling `initial`, stop at `maximum`, then bisect until `(upper-lower)/max(upper, 1e-12) <= relative_tolerance`. Record every evaluated `(scale, passed)` pair.

- [ ] **Step 5: Run geometry tests**

Run: `.venv/bin/python -m pytest tests/test_geometry.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add jlens/geometry.py tests/test_geometry.py
git commit -m "feat: add calibrated intervention geometry"
```

### Task 2: Safe residual mutation hooks

**Files:**
- Modify: `jlens/hooks.py`
- Modify: `jlens/protocol.py`
- Create: `tests/test_intervention_hooks.py`

**Interfaces:**
- Produces: `ResidualEdit(layer: int, positions: tuple[int, ...], delta: Tensor, batch_indices: tuple[int, ...] | None = None)`
- Produces: `ActivationEditor(blocks, edits)` context manager
- Preserves: `ActivationRecorder` behavior

- [ ] **Step 1: Write tests for tensor, tuple, position, batch, and cleanup behavior**

```python
def test_editor_changes_only_selected_position():
    block = nn.Identity()
    x = torch.zeros(2, 4, 3)
    edit = ResidualEdit(layer=0, positions=(1,), delta=torch.tensor([1., 2., 3.]), batch_indices=(0,))
    with ActivationEditor([block], [edit]):
        out = block(x)
    torch.testing.assert_close(out[0, 1], edit.delta)
    assert torch.count_nonzero(out[1]) == 0

def test_editor_restores_hooks_after_exception():
    block = nn.Identity()
    with pytest.raises(RuntimeError):
        with ActivationEditor([block], [ResidualEdit(0, (-1,), torch.ones(3))]):
            raise RuntimeError("boom")
    assert len(block._forward_hooks) == 0
```

- [ ] **Step 2: Run tests and verify missing-symbol failures**

Run: `.venv/bin/python -m pytest tests/test_intervention_hooks.py -q`
Expected: import failure for `ActivationEditor`.

- [ ] **Step 3: Implement immutable edit specifications and output reconstruction**

The hook must clone the residual, resolve negative positions, validate batch and residual dimensions, cast `delta` to the activation device/dtype, edit without mutating the original tensor, and reconstruct tensor, tuple, or list outputs without changing auxiliary values.

- [ ] **Step 4: Run hook tests plus existing hook-dependent tests**

Run: `.venv/bin/python -m pytest tests/test_intervention_hooks.py tests/test_fitting.py -q`
Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add jlens/hooks.py jlens/protocol.py tests/test_intervention_hooks.py
git commit -m "feat: add scoped residual editing hooks"
```

### Task 3: Calibrated lens serialization and fitting

**Files:**
- Modify: `jlens/lens.py`
- Modify: `jlens/fitting.py`
- Create: `tests/test_lens_geometry_io.py`

**Interfaces:**
- Extends: `JacobianLens(jacobians: dict[int, Tensor], *, n_prompts: int, d_model: int, geometry: dict[int, CovarianceMetric] | None = None, metadata: dict[str, str] | None = None)`
- Produces: `JacobianLens.metric(layer) -> CovarianceMetric`
- Produces: `calibrate_geometry(model, lens, prompts, *, max_seq_len=128, rank=16, shrinkage=0.05) -> JacobianLens`

- [ ] **Step 1: Write legacy and calibrated round-trip tests**

```python
def test_legacy_checkpoint_loads_uncalibrated(tmp_path):
    torch.save({"J": {0: torch.eye(3)}, "n_prompts": 1, "d_model": 3}, tmp_path / "legacy.pt")
    lens = JacobianLens.load(str(tmp_path / "legacy.pt"))
    assert not lens.metric(0).calibrated

def test_calibrated_geometry_round_trip(tmp_path):
    metric = CovarianceMetric(torch.tensor([1., 2., 3.]), torch.ones(1, 3) * 0.1)
    lens = JacobianLens({0: torch.eye(3)}, n_prompts=2, d_model=3, geometry={0: metric})
    lens.save(str(tmp_path / "lens.pt"))
    loaded = JacobianLens.load(str(tmp_path / "lens.pt"))
    torch.testing.assert_close(loaded.metric(0).diagonal, metric.diagonal, atol=2e-3, rtol=0)
```

- [ ] **Step 2: Run tests and verify constructor failure**

Run: `.venv/bin/python -m pytest tests/test_lens_geometry_io.py -q`
Expected: `JacobianLens.__init__()` rejects `geometry`.

- [ ] **Step 3: Add versioned, backward-compatible serialization**

Store `format_version=2`, per-layer diagonal/factor tensors, and metadata. Keep existing `J`, `n_prompts`, `source_layers`, and `d_model` keys. `metric(layer)` returns an uncalibrated identity metric when no geometry exists.

- [ ] **Step 4: Implement streaming covariance calibration**

Use `ActivationRecorder` to gather fitted-layer residuals with gradients disabled. Maintain Welford mean/diagonal variance and a bounded Frequent-Directions sketch per layer. At finalization, create low-rank factors from the sketch, subtract their represented diagonal variance from the total diagonal, clamp to `1e-6 * mean_variance`, and apply configured shrinkage.

- [ ] **Step 5: Run IO, fitting, and apply tests**

Run: `.venv/bin/python -m pytest tests/test_lens_geometry_io.py tests/test_fitting.py -q`
Expected: all tests pass.

- [ ] **Step 6: Commit**

```bash
git add jlens/lens.py jlens/fitting.py tests/test_lens_geometry_io.py
git commit -m "feat: calibrate and persist residual geometry"
```

### Task 4: Concepts, score linearization, and traces

**Files:**
- Create: `jlens/interventions.py`
- Create: `tests/test_interventions.py`

**Interfaces:**
- Produces: `ConceptSpec`, `ConceptResolver.resolve(text_or_ids) -> ConceptSpec`
- Produces: `local_score_covectors(model, lens, residual, layer, token_ids) -> Tensor`
- Produces: `InterventionTrace.to_dict()/from_dict()`
- Produces: `InterventionResult`

- [ ] **Step 1: Write concept, exact-gradient, and trace tests**

```python
def test_local_covector_matches_finite_difference(tiny_model, tiny_lens):
    h = torch.randn(tiny_model.d_model)
    token = 3
    grad = local_score_covectors(tiny_model, tiny_lens, h, 1, [token])[0]
    direction = torch.randn_like(h)
    eps = 1e-3
    score = lambda x: tiny_model.unembed(tiny_lens.transport(x, 1))[token]
    fd = (score(h + eps * direction) - score(h - eps * direction)) / (2 * eps)
    torch.testing.assert_close(grad @ direction, fd, atol=2e-3, rtol=2e-2)

def test_trace_json_round_trip():
    trace = InterventionTrace(
        operation="inject", target_ids=(7,), source_ids=(), experimental=False,
        selected_layer=2, selected_positions=(-1,), selected_scale=0.5,
        normalized_cost=0.25, baseline_scores={"7": 0.1},
        after_scores={"7": 0.5},
        search_points=(SearchPoint(0.5, True, 0.4),), warnings=(),
    )
    assert InterventionTrace.from_dict(trace.to_dict()) == trace
```

- [ ] **Step 2: Run tests and verify missing-symbol failures**

Run: `.venv/bin/python -m pytest tests/test_interventions.py -q`
Expected: import failures for intervention types.

- [ ] **Step 3: Implement concept resolution**

Try bare and leading-space tokenizations, discard special-only results, prefer a unique single-token variant, preserve all alternatives in the result, and set `experimental=True` for selected multi-token variants.

- [ ] **Step 4: Implement exact local score derivatives**

Create a detached fp32 residual with gradients enabled, call `lens.transport` and `model.unembed`, and use `torch.autograd.grad` once per requested scalar token score. Return `[n_tokens, d_model]` gradients and never retain the graph after return.

- [ ] **Step 5: Implement frozen trace/result dataclasses**

Use JSON-compatible primitives for trace storage. Tensor values must be converted to scalar/list fields at construction, not hidden inside `to_dict`.

- [ ] **Step 6: Run tests and commit**

Run: `.venv/bin/python -m pytest tests/test_interventions.py -q`
Expected: all current tests pass.

```bash
git add jlens/interventions.py tests/test_interventions.py
git commit -m "feat: add concepts and intervention traces"
```

### Task 5: Minimum-effective injection and replacement engine

**Files:**
- Modify: `jlens/interventions.py`
- Modify: `tests/test_interventions.py`
- Modify: `jlens/__init__.py`

**Interfaces:**
- Produces: `InterventionEngine(model, lens)`
- Produces: `inject(prompt, target, *, layers=None, positions=(-1,), top_k=5, margin=0.0, maximum_scale=16.0, relative_tolerance=0.01) -> InterventionResult`
- Produces: `suppress(prompt, target, *, layers=None, positions=(-1,), top_k=5, margin=0.0, maximum_scale=16.0, relative_tolerance=0.01) -> InterventionResult`
- Produces: `replace(prompt, source, target, *, layers=None, positions=(-1,), margin=0.0, preserve_top_k=8, preservation_tolerance=0.05, maximum_scale=16.0, relative_tolerance=0.01) -> InterventionResult`

- [ ] **Step 1: Write end-to-end tiny-decoder tests**

```python
def test_inject_reaches_target_with_minimal_reported_scale(tiny_engine):
    result = tiny_engine.inject(PROMPT, 7, layers=[2], positions=(-1,), top_k=8)
    assert result.success
    assert result.trace.selected_scale >= 0
    assert result.trace.search_points[-1].passed

def test_replace_target_outranks_source(tiny_engine):
    baseline = tiny_engine.read(PROMPT, layer=2, position=-1)
    source = baseline.top_ids[0]
    target = baseline.top_ids[-1]
    result = tiny_engine.replace(PROMPT, source, target, layers=[2], positions=(-1,))
    assert result.success
    assert result.trace.after_scores[str(target)] >= result.trace.after_scores[str(source)]

def test_suppress_removes_target_from_top_k(tiny_engine):
    baseline = tiny_engine.read(PROMPT, layer=2, position=-1)
    target = baseline.top_ids[0]
    result = tiny_engine.suppress(PROMPT, target, layers=[2], positions=(-1,), top_k=8)
    assert result.success and target not in result.trace.after_top_ids[:8]

def test_multitoken_result_is_experimental(tiny_engine):
    result = tiny_engine.inject(PROMPT, "ab", layers=[2], positions=(-1,), top_k=16)
    assert result.trace.experimental
    assert result.trace.sequence_logprob_after >= result.trace.sequence_logprob_before
```

- [ ] **Step 2: Run focused tests and verify engine import failure**

Run: `.venv/bin/python -m pytest tests/test_interventions.py -q`
Expected: import failure for `InterventionEngine`.

- [ ] **Step 3: Implement baseline capture and candidate construction**

Capture selected and final residuals in one forward pass. Compute baseline lens scores, select the strongest competitors, compute exact local covectors, and form linearized deficit constraints:

```python
current_margin = score[target] - score[competitor]
constraint = covector[target] - covector[competitor]
deficit = requested_margin - current_margin
```

Use `minimum_cost_perturbation` with `lens.metric(layer)`.

For replacement, add both signs of each preservation covector as constraints,
with bounds equal to `preservation_tolerance` times that concept's baseline
absolute score. For suppression, reverse the target/competitor constraints so
the target falls below the current top-k boundary.

- [ ] **Step 4: Implement real-forward verification and scale calibration**

Apply `ActivationEditor`, capture final residuals, recompute nonlinear lens scores from `h + scale * delta`, and compute downstream logits from the real final residual. Injection passes when the target is in lens top-k and its downstream logit exceeds baseline. Replacement passes when target score is at least source score and downstream target-minus-source margin improves.

For a multi-token target, build one normalized constraint group per selected
token. Verify the phrase by appending it to the prompt under teacher forcing,
summing the corresponding continuation log-probabilities, and requiring this
sequence log-probability to improve over baseline. Record both values in the
trace and retain the experimental warning.

- [ ] **Step 5: Implement layer/position selection and failure diagnostics**

Search only requested or fitted workspace layers, reject invalid positions before any forward pass, cap scales at `maximum_scale`, and choose the successful candidate with minimum `scale**2 * solution.cost`. When none passes, return the candidate with best target-margin improvement and `success=False`.

- [ ] **Step 6: Export public API and run regression tests**

Run: `.venv/bin/python -m pytest tests/test_geometry.py tests/test_intervention_hooks.py tests/test_interventions.py tests/test_fitting.py -q`
Expected: all tests pass.

- [ ] **Step 7: Commit**

```bash
git add jlens/interventions.py jlens/__init__.py tests/test_interventions.py
git commit -m "feat: add minimum-effective J-space interventions"
```

### Task 6: Inspection export, documentation, and final verification

**Files:**
- Modify: `jlens/vis.py`
- Modify: `README.md`
- Modify: `tests/test_vis_modes.py`
- Modify: `tests/test_compute_slice.py`

**Interfaces:**
- Produces: `write_intervention_trace(trace, out_dir) -> Path`
- Extends: `build_page(slice_data, prompt, *, title, description, pinned_token_ids=None, mode="embed", out_dir=None, alt_token=None, intervention_trace: InterventionTrace | None = None)`
- Documents: `InterventionEngine.inject`, `.suppress`, `.replace`, calibration, and limitations

- [ ] **Step 1: Write trace-sidecar test**

```python
def test_write_intervention_trace(tmp_path, trace):
    path = write_intervention_trace(trace, tmp_path)
    assert path.name == "intervention.json"
    assert json.loads(path.read_text())["operation"] == trace.operation

def test_embed_page_contains_intervention_summary(trace):
    page, _, _ = build_page(_synthetic_slice(), "p", title="T", description="d", intervention_trace=trace)
    assert '"intervention"' in page and '"selected_scale"' in page
```

- [ ] **Step 2: Run the test and verify missing-symbol failure**

Run: `.venv/bin/python -m pytest tests/test_vis_modes.py::test_write_intervention_trace -q`
Expected: import failure for `write_intervention_trace`.

- [ ] **Step 3: Implement atomic JSON sidecar writing**

Serialize to `intervention.json.tmp`, flush and close it, then replace
`intervention.json`. In embed mode add the same JSON-compatible dictionary to
the bootstrap payload under `intervention`; in fetch mode add
`intervention.json` to the sidecar manifest. Render a compact summary block
above the existing slice grid showing operation, concepts, selected
layer/position, scale, normalized cost, and warnings. Do not embed model tensors
or arbitrary Python objects.

- [ ] **Step 4: Add README examples and limitations**

Document a complete single-token injection example, replacement example, geometry calibration, legacy uncalibrated fallback, bounded-strength failure behavior, experimental multi-token status, and Hugging Face decoder-only scope.

- [ ] **Step 5: Run formatting and the complete suite**

Run: `.venv/bin/python -m ruff check jlens tests`
Expected: no diagnostics.

Run: `.venv/bin/python -m pytest -q`
Expected: all tests pass.

- [ ] **Step 6: Run a direct CPU smoke test**

Fit the tiny decoder, calibrate geometry, inject a token, replace a baseline top token, save/load the lens, serialize both traces, and print the selected layers, scales, and normalized costs. Expected output ends with `INTERVENTION_SMOKE_OK`.

- [ ] **Step 7: Commit**

```bash
git add jlens/vis.py README.md tests/test_vis_modes.py tests/test_compute_slice.py
git commit -m "docs: expose calibrated intervention workflow"
```
