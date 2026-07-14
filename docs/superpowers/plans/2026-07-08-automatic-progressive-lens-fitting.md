# Automatic Progressive Lens Fitting Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make J Studio automatically create, validate, activate, and cache a fast progressive lens when a compatible cached lens is absent.

**Architecture:** The existing HF runtime owns one model and serializes generation, fitting, and slice work through a GPU coordinator. A progressive controller runs `jlens.fit_progressive` on a worker, publishes immutable stage updates, and atomically activates only validated stages.

**Tech Stack:** Python 3.11, PySide6, existing `jlens` backend, PyTorch ROCm, pytest-qt

## Global Constraints

- Never load a second 7B/8B model copy for fitting.
- Never terminate unrelated GPU workloads.
- Preview is inspectable but cannot arm interventions; Stable may enable them.
- Model/tokenizer/estimator fingerprints control cache compatibility.
- Existing compatible lenses load immediately.

---

### Task 1: Fitting Status Protocol and Fake Service

**Files:**
- Modify: `jstudio/domain.py`
- Modify: `jstudio/services/protocols.py`
- Modify: `jstudio/services/fake.py`
- Test: `tests/test_fake_services.py`

**Interfaces:**
- Produces: `LensFitState(StrEnum): MISSING, WAITING, PREVIEW, REFINING, STABLE, FAILED`
- Produces: `LensFitStatus(state, stage, completed, total, elapsed_seconds, quality, detail)`
- Produces: `LensService.fit_status()`, `LensService.start_fit()`, `LensService.cancel_fit()`

- [ ] **Step 1: Write the failing lifecycle test**

```python
def test_fake_lens_fit_publishes_preview_then_stable():
    services = create_fake_services(token_delay=0)
    seen = []
    services.lens.fit_changed.connect(seen.append)
    services.lens.start_fit()
    assert [status.state for status in seen] == [
        LensFitState.PREVIEW, LensFitState.REFINING, LensFitState.STABLE]
    assert services.lens.fit_status().quality == "passed"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_fake_services.py -k fit_publishes -q`

- [ ] **Step 3: Add immutable records and backend-neutral callbacks**

Use a small callback subscription API rather than Qt signals in service code: `subscribe_fit(callback) -> unsubscribe`. Fake fitting publishes the three deterministic states synchronously.

- [ ] **Step 4: Run GREEN and commit**

Run: `python -m pytest tests/test_fake_services.py tests/test_domain.py -q`

```bash
git add jstudio/domain.py jstudio/services/protocols.py jstudio/services/fake.py tests/test_fake_services.py
git commit -m "feat: define progressive lens fitting status"
```

### Task 2: Shared-Model Progressive Controller

**Files:**
- Create: `jstudio/services/lens_fitting.py`
- Modify: `jstudio/services/hf_runtime.py`
- Test: `tests/test_lens_fitting.py`
- Modify: `tests/test_hf_runtime.py`

**Interfaces:**
- Produces: `GPUCoordinator.exclusive(operation)`
- Produces: `ProgressiveLensController.start()`, `cancel()`, `status()`, `subscribe(callback)`
- Produces atomic `HFModelRuntime.activate_lens(lens, path, quality)`

- [ ] **Step 1: Write failing controller tests**

```python
def test_controller_activates_preview_then_stable(tmp_path):
    runtime = RuntimeDouble()
    fitter = ProgressiveFitterDouble((preview_result(), stable_result()))
    controller = ProgressiveLensController(runtime, fitter, tmp_path)
    controller.start(); controller.join(timeout=1)
    assert runtime.activated == ["Preview", "Stable"]
    assert controller.status().state is LensFitState.STABLE

def test_cancel_preserves_last_valid_stage(tmp_path):
    controller = cancellable_controller(tmp_path)
    controller.start(); controller.wait_for_stage("Preview"); controller.cancel()
    assert controller.runtime.active_stage == "Preview"
```

- [ ] **Step 2: Run RED**

Run: `python -m pytest tests/test_lens_fitting.py -q`

- [ ] **Step 3: Implement single-model GPU coordination**

All model generation, slice computation, and fitting acquire the same reentrant coordinator. Fitting checks cancellation between prompts. Interactive work gets the lock at the next prompt boundary; no operation interrupts an active autograd graph.

- [ ] **Step 4: Permit startup without a lens**

`HFModelRuntime` must stop raising `FileNotFoundError`. It reports `lens=None`, starts the controller, rejects slice/intervention requests with the current fit status, and continues supporting exact generation. A compatible cached lens still loads synchronously.

- [ ] **Step 5: Run tests and commit**

Run: `python -m pytest tests/test_lens_fitting.py tests/test_hf_runtime.py -q`

```bash
git add jstudio/services/lens_fitting.py jstudio/services/hf_runtime.py tests/test_lens_fitting.py tests/test_hf_runtime.py
git commit -m "feat: fit and activate lenses on the shared model"
```

### Task 3: Progress UI and Capability Transitions

**Files:**
- Modify: `jstudio/ui/shell/session_bar.py`
- Modify: `jstudio/ui/jlens/workspace.py`
- Modify: `jstudio/ui/shell/main_window.py`
- Create: `tests/ui/test_lens_fit_status.py`

**Interfaces:**
- Consumes `LensFitStatus` updates.
- Produces visible progress, Cancel/Resume actions, quality diagnostics, and automatic J-Lens refresh when a stage activates.

- [ ] **Step 1: Write failing UI transition test**

```python
def test_lens_fit_status_enables_inspection_before_intervention(window, services):
    services.lens.publish_fit(LensFitState.PREVIEW, 8, 8, "unchecked")
    assert window.jlens_workspace.isEnabled()
    assert not window.main_workspace.arm_button.isEnabled()
    services.lens.publish_fit(LensFitState.STABLE, 32, 32, "passed")
    assert window.main_workspace.arm_button.isEnabled()
    assert "Stable" in window.session_bar.lens_status.text()
```

- [ ] **Step 2: Run RED**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_lens_fit_status.py -q`

- [ ] **Step 3: Implement progress and recovery controls**

Show stage, prompt/probe progress, elapsed time, and quality state in the session bar and J-Lens empty state. Cancel preserves the last stage; Resume calls `start_fit`. Stable activation refreshes the current prompt slice without changing selection.

- [ ] **Step 4: Run all app tests and commit**

Run: `QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu python -m pytest -q`

Run: `python -m ruff check jstudio tests`

```bash
git add jstudio/ui/shell/session_bar.py jstudio/ui/jlens/workspace.py jstudio/ui/shell/main_window.py tests/ui/test_lens_fit_status.py
git commit -m "feat: show automatic lens fitting progress"
```

### Task 4: Real End-to-End Acceptance

**Files:**
- Modify: `README.md`

**Interfaces:**
- Verifies automatic fitting from a deliberately empty model cache directory.

- [ ] **Step 1: Run with a fresh lens cache after backend acceptance passes**

```bash
JSTUDIO_LENS_CACHE=/tmp/jstudio-lens-acceptance python -m jstudio
```

Verify Preview appears under 15 seconds, Stable appears under 90 seconds, generation remains cancellable, and the stable lens survives restart.

- [ ] **Step 2: Run full regression and document measured behavior**

Run both repository test suites. Record actual Preview/Stable time, peak VRAM, and quality metrics in README without rounding a failed target into a pass.

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: document automatic progressive fitting"
```

