# Multi-token J-space Interventions Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add multi-token inject, suppress, and replace using bounded J-space residual transforms with automatically selected minimum effective strength.

**Architecture:** Phrase geometry is isolated in `jlens/concepts.py`; state-dependent forward hooks live in `jlens/hooks.py`; `InterventionEngine` builds and measures transforms without a downstream-logit fallback. J Studio prepares each enabled rule independently so an invalid rule cannot abort generation.

**Tech Stack:** Python 3.11, PyTorch, pytest, PySide6, Hugging Face Transformers, ROCm.

## Global Constraints

- Never force a token or use a direct/next-token-logit fallback.
- Preserve existing saved lenses and single-token intervention records.
- Treat UI strength as a maximum budget and report the selected minimum strength.
- Temporary hooks must always be removed, including after exceptions.
- A failed rule must not abort generation or disable other successful rules.

---

### Task 1: Phrase geometry

**Files:**
- Create: `jlens/concepts.py`
- Create: `tests/test_concepts.py`

**Interfaces:**
- Produces: `PhraseGeometry`, `compact_basis(directions)`, `sequence_alignment(source_count, target_count)`, and `phrase_geometry(covectors, target_covectors=None)`.

- [ ] **Step 1: Write failing geometry tests**

```python
def test_compact_basis_removes_duplicate_directions():
    basis = compact_basis(torch.tensor([[1., 0.], [1., 0.], [0., 1.]]))
    torch.testing.assert_close(basis.T @ basis, torch.eye(2))

def test_alignment_normalizes_unequal_length_columns():
    alignment = sequence_alignment(2, 4)
    torch.testing.assert_close(alignment.sum(0), torch.ones(2))
```

- [ ] **Step 2: Run tests and verify they fail because the module is absent**

Run: `python3.11 -m pytest -q tests/test_concepts.py`
Expected: FAIL importing `jlens.concepts`.

- [ ] **Step 3: Implement compact SVD geometry**

```python
@dataclass(frozen=True)
class PhraseGeometry:
    source_basis: torch.Tensor
    target_directions: torch.Tensor | None
    alignment: torch.Tensor | None

def compact_basis(directions, relative_tolerance=1e-5):
    _, singular, vh = torch.linalg.svd(normalize_rows(directions), full_matrices=False)
    keep = singular > singular.max() * relative_tolerance
    return vh[keep].T.contiguous()
```

`sequence_alignment` must use linear interpolation across normalized token positions and normalize every source column.

- [ ] **Step 4: Run geometry tests**

Run: `python3.11 -m pytest -q tests/test_concepts.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jlens/concepts.py tests/test_concepts.py
git commit -m "feat: add multi-token phrase geometry"
```

### Task 2: State-dependent residual transforms

**Files:**
- Modify: `jlens/hooks.py`
- Modify: `tests/test_hooks.py`

**Interfaces:**
- Produces: `ResidualTransform(layer, positions, transform, batch_indices=None, max_applications=None)` and `ResidualTransformEditor(blocks, transforms)`.
- `transform` consumes and returns `[..., d_model]` tensors without mutating its input.

- [ ] **Step 1: Write failing tests for composition, cached `-1` positions, and cleanup**

```python
def test_transform_editor_recomputes_last_position_and_cleans_up():
    edit = ResidualTransform(0, (-1,), lambda h: h * 0)
    with ResidualTransformEditor(blocks, [edit]):
        assert blocks[0](torch.ones(1, 3, 4))[0, -1].count_nonzero() == 0
        assert blocks[0](torch.ones(1, 1, 4))[0, -1].count_nonzero() == 0
    torch.testing.assert_close(blocks[0](torch.ones(1, 1, 4)), torch.ones(1, 1, 4))
```

- [ ] **Step 2: Verify RED**

Run: `python3.11 -m pytest -q tests/test_hooks.py -k transform`
Expected: FAIL importing the new types.

- [ ] **Step 3: Implement the scoped transform editor**

Reuse `ActivationEditor._residual` and `_replace_residual`, clone once per layer, apply transforms in input order, validate shape/finite output, and remove every handle in `__exit__`.

- [ ] **Step 4: Verify GREEN**

Run: `python3.11 -m pytest -q tests/test_hooks.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jlens/hooks.py tests/test_hooks.py
git commit -m "feat: add scoped residual transforms"
```

### Task 3: Multi-token projection engine

**Files:**
- Modify: `jlens/interventions.py`
- Modify: `jlens/__init__.py`
- Modify: `tests/test_interventions.py`

**Interfaces:**
- Produces: `PhraseResidualOperator.apply(residual)`, `InterventionEngine.phrase_inject`, `phrase_suppress`, and `phrase_replace`.
- Extends `InterventionResult` with optional `operator`; `apply()` selects a `ResidualTransformEditor` when present.

- [ ] **Step 1: Write failing multi-token behavior tests**

```python
def test_multitoken_suppress_uses_all_source_tokens(tiny_engine):
    result = tiny_engine.phrase_suppress(PROMPT, "ab", layers=[2], maximum_scale=16)
    assert result.success
    assert result.trace.source_ids == (8, 9)
    assert result.trace.selected_scale < 16

def test_unequal_multitoken_replace_builds_measured_operator(tiny_engine):
    result = tiny_engine.phrase_replace(PROMPT, "ab", "cdef", layers=[2], maximum_scale=16)
    assert result.success
    assert result.operator is not None
    assert result.trace.target_ids == (10, 11, 12, 13)
```

- [ ] **Step 2: Verify RED**

Run: `python3.11 -m pytest -q tests/test_interventions.py -k phrase`
Expected: FAIL because phrase methods are absent.

- [ ] **Step 3: Implement bounded phrase operators**

```python
@dataclass(frozen=True)
class PhraseResidualOperator:
    source_basis: torch.Tensor | None
    target_directions: torch.Tensor
    alignment: torch.Tensor | None
    alpha: float
    beta: float
    injection_scale: float

    def apply(self, residual):
        if self.source_basis is None:
            rms = residual.float().pow(2).mean(-1, keepdim=True).sqrt()
            return residual + self.injection_scale * rms * self.target_directions.mean(1)
        coefficients = residual @ self.source_basis
        removed = coefficients @ self.source_basis.T
        redirected = (coefficients @ self.alignment.T) @ self.target_directions.T
        return residual + self.alpha * removed + self.beta * redirected
```

Search the normalized scale ladder `(1/16, 1/8, 1/4, 1/2, 3/4, 1)` capped by `maximum_scale / 16`. Select the first operator meeting source/target movement, cosine, norm, and finite-value checks. Store UI scale as normalized scale times 16.

- [ ] **Step 4: Verify phrase and regression tests**

Run: `python3.11 -m pytest -q tests/test_concepts.py tests/test_hooks.py tests/test_interventions.py`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jlens/interventions.py jlens/__init__.py tests/test_interventions.py
git commit -m "feat: add minimum-strength phrase interventions"
```

### Task 4: J Studio integration and rule isolation

**Files:**
- Modify: `../app/jstudio/services/hf_runtime.py`
- Modify: `../app/tests/test_hf_runtime.py`

**Interfaces:**
- Consumes: `InterventionEngine.phrase_inject`, `phrase_suppress`, `phrase_replace`, and `apply`.
- Produces: `HFModelRuntime.prepare_interventions(...)` returning one editor/result pair per draft without raising for a draft-local validation failure.

- [ ] **Step 1: Write failing runtime tests**

```python
def test_prepare_interventions_keeps_valid_rules_after_multitoken_failure():
    editors, results = runtime.prepare_interventions(prompt, (invalid, valid))
    assert not results[0].success
    assert results[1].success
    assert len(editors) == 2
```

Also assert multi-token replace calls `phrase_replace` and no generation-path next-token verifier is called.

- [ ] **Step 2: Verify RED**

Run: `python3.11 -m pytest -q ../app/tests/test_hf_runtime.py -k 'multitoken or valid_rules'`
Expected: FAIL because the existing loop aborts and uses the next-token verifier.

- [ ] **Step 3: Wire phrase methods and isolate failures**

Create one engine per preparation, dispatch all operations to phrase methods, wrap each draft in `try/except`, convert failures into an unsuccessful `InterventionResult`, and append `nullcontext()` for failed rules. Remove `_editor_targets_next_token` and `_next_token_logits` from the intervention path.

- [ ] **Step 4: Verify app integration**

Run: `python3.11 -m pytest -q ../app/tests/test_hf_runtime.py`
Expected: PASS.

- [ ] **Step 5: Commit in the app repository**

```bash
cd ../app
git add jstudio/services/hf_runtime.py tests/test_hf_runtime.py
git commit -m "feat: enable multi-token J-space rules"
```

### Task 5: Verification and Qwen smoke benchmark

**Files:**
- Create: `scripts/benchmark_phrase_interventions.py`
- Create at runtime: `reports/intervention-benchmark/phrase-qwen/index.html`

**Interfaces:**
- Produces a self-contained HTML table containing operation, source, target, selected layer/strength, measured source movement, target movement, residual cosine, norm ratio, and pass/fail.

- [ ] **Step 1: Add the opt-in benchmark script**

Use the active local Qwen model and stable lens. Run the three approved phrase cases plus two composed rules, enter/exit every editor with `ExitStack`, and assert no layer retains a forward hook.

- [ ] **Step 2: Run static and unit verification**

```bash
python3.11 -m ruff check jlens tests scripts
python3.11 -m pytest -q
cd ../app && python3.11 -m ruff check jstudio tests && python3.11 -m pytest -q
```

Expected: both Ruff runs clean and every test passes.

- [ ] **Step 3: Run the Qwen benchmark**

```bash
python3.11 scripts/benchmark_phrase_interventions.py \
  --model heterodoxin/qwen3-8b-apostate \
  --lens ../lenses/heterodoxin--qwen3-8b-apostate/lens.pt \
  --output reports/intervention-benchmark/phrase-qwen
```

Expected: finite measurements, selected strengths within budget, no leaked hooks, and HTML report created. A scientifically failed case remains marked failed; it is not converted to a fallback.

- [ ] **Step 4: Restart J Studio**

Run: `cd ../app && python3.11 -m jstudio`
Expected: model and stable lens load, and a multi-token rule no longer aborts generation.
