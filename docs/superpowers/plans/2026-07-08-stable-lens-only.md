# Stable Lens Only Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make J Studio refuse Preview, uncalibrated, legacy, or failed lenses for J-space viewing and interventions.

**Architecture:** The local runtime is the authority for whether a lens is usable. UI capabilities mirror only `LensFitState.STABLE`; Preview fit artifacts may be saved for resume/debugging but are never activated as the runtime lens or shown as J-space.

**Tech Stack:** Python 3.11, PySide6, pytest, Ruff, local ROCm/Hugging Face runtime, `jlens.JacobianLens`.

## Global Constraints

- Stable lens is required for J-Lens viewing.
- Stable calibrated lens is required for interventions.
- Legacy `identity-plus-*` lenses and Preview artifacts must not be loaded for viewing.
- Failed Stable quality gates must leave the UI in “Stable required,” not “Preview active.”
- Use TDD: write failing tests, verify red, implement, verify green.

---

### Task 1: Runtime rejects non-Stable lenses

**Files:**
- Modify: `jstudio/services/hf_runtime.py`
- Test: `tests/test_hf_runtime.py`

**Interfaces:**
- Consumes: `HFModelRuntime._validate_lens(lens) -> None`
- Produces: `_candidate_lens_paths(selected: Path) -> tuple[Path, ...]` containing Stable candidates only.

- [ ] **Step 1: Write failing tests**

Add tests asserting Preview metadata, uncalibrated Stable metadata, and legacy estimators are rejected.

- [ ] **Step 2: Run tests to verify red**

Run: `pytest -q tests/test_hf_runtime.py::test_runtime_rejects_preview_lens_for_viewing tests/test_hf_runtime.py::test_runtime_rejects_uncalibrated_stable_lens_for_viewing`

- [ ] **Step 3: Implement Stable-only validation**

Change `_candidate_lens_paths` to exclude `preview.lens.pt`. Change `_validate_lens` to require `quality_stage == "Stable"` and calibrated geometry for every source layer.

- [ ] **Step 4: Verify green**

Run: `pytest -q tests/test_hf_runtime.py`

### Task 2: Progressive fitting activates only Stable

**Files:**
- Modify: `jstudio/services/lens_fitting.py`
- Test: `tests/test_lens_fitting.py`

**Interfaces:**
- Consumes: `ProgressiveLensController._on_stage(result)`
- Produces: Preview status as non-active progress only; Stable status activates the runtime lens only when quality gates pass.

- [ ] **Step 1: Write failing tests**

Update controller tests so Preview is not activated and failed Stable gates end in `FAILED`.

- [ ] **Step 2: Run tests to verify red**

Run: `pytest -q tests/test_lens_fitting.py`

- [ ] **Step 3: Implement Stable-only activation**

Set `accepted = result.name == "Stable" and result.quality.stable`. Do not call `runtime.activate_lens` for Preview. After fitting completes without Stable, publish `FAILED`.

- [ ] **Step 4: Verify green**

Run: `pytest -q tests/test_lens_fitting.py`

### Task 3: UI exposes only Stable inspect capability

**Files:**
- Modify: `jstudio/ui/shell/main_window.py`
- Modify: `jstudio/ui/jlens/workspace.py`
- Test: `tests/ui/test_lens_fit_status.py`

**Interfaces:**
- Consumes: `LensFitStatus.state`
- Produces: `SessionCapabilities.inspect == True` only for `LensFitState.STABLE`.

- [ ] **Step 1: Write failing UI tests**

Assert Preview disables the J-Lens web view and session inspect capability; Stable enables both.

- [ ] **Step 2: Run tests to verify red**

Run: `QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu pytest -q tests/ui/test_lens_fit_status.py`

- [ ] **Step 3: Implement UI gate**

Change J-Lens workspace web enablement to `status.state is LensFitState.STABLE`. Change shell capability refresh so Preview does not set `inspect=True` or a fake lens id.

- [ ] **Step 4: Verify green**

Run: `QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu pytest -q tests/ui/test_lens_fit_status.py`

### Task 4: Full verification and restart

**Files:**
- No additional files.

**Interfaces:**
- Consumes: all previous tasks.
- Produces: committed stable-lens-only behavior.

- [ ] **Step 1: Run full verification**

Run: `QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu pytest -q && ruff check jstudio tests && git diff --check`

- [ ] **Step 2: Commit**

Run: `git add ... && git commit -m "fix: require stable lens for j-space"`

- [ ] **Step 3: Restart app**

Stop the old `python -m jstudio` process and start `python -m jstudio` from `/var/home/Heterodoxin/Desktop/J-Studio/app`.
