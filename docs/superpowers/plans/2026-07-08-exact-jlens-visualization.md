# Exact J-Lens Visualization Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the approximate Qt J-Lens matrix with the repository's original interactive `jlens.vis` renderer and preserve native intervention entry points.

**Architecture:** A backend-neutral slice protocol returns self-contained HTML from an asynchronous lens service. The real service delegates to `jlens.vis.compute_slice` and `build_page`; the UI hosts the page in a locked-down `QWebEngineView` with a schema-limited `QWebChannel` bridge.

**Tech Stack:** Python 3.11, PySide6/Qt WebEngine/Qt WebChannel, `jlens.vis`, pytest, pytest-qt

## Global Constraints

- The original `jlens.vis` HTML is the authoritative renderer; do not recreate its matrix, heatmap, or rank plots in Qt.
- Slice computation must run outside the GUI thread and stale results must be discarded.
- Default slices use `mask_display=True`, full-vocabulary ranks, and the final `J = I` model layer.
- Local web content may not navigate to arbitrary remote URLs.
- Bridge messages may select coordinates and request native actions, but may not execute arbitrary Python or filesystem operations.
- Do not display sigmoid-compressed activation scores as the primary value.

---

### Task 1: Slice Service Contract and Deterministic Fake

**Files:**
- Modify: `jstudio/services/protocols.py`
- Modify: `jstudio/services/fake.py`
- Test: `tests/test_fake_services.py`

**Interfaces:**
- Produces: `SliceRequest(run_id: str, text: str, title: str, layer_stride: int = 1, last_n_tokens: int | None = None, top_n: int = 10, mask_display: bool = True)`
- Produces: `SlicePage(run_id: str, generation: int, html: str)`
- Produces: `LensService.request_slice(request: SliceRequest) -> Future[SlicePage]`

- [ ] **Step 1: Write the failing fake-service test**

```python
def test_fake_lens_service_returns_self_contained_slice_page():
    services = create_fake_services(token_delay=0)
    request = SliceRequest("run-1", "( ^ )", "ASCII face")
    page = services.lens.request_slice(request).result(timeout=1)
    assert page.run_id == "run-1"
    assert page.generation == 1
    assert "J-lens" in page.html
    assert "ASCII face" in page.html
    services.generation.close()
```

- [ ] **Step 2: Run the test and verify RED**

Run: `python -m pytest tests/test_fake_services.py::test_fake_lens_service_returns_self_contained_slice_page -q`

Expected: FAIL because `SliceRequest` and `request_slice` do not exist.

- [ ] **Step 3: Add immutable protocol records and fake implementation**

```python
@dataclass(frozen=True, slots=True)
class SliceRequest:
    run_id: str
    text: str
    title: str
    layer_stride: int = 1
    last_n_tokens: int | None = None
    top_n: int = 10
    mask_display: bool = True

@dataclass(frozen=True, slots=True)
class SlicePage:
    run_id: str
    generation: int
    html: str
```

Implement `FakeLensService.request_slice` with a completed `Future` and an incrementing generation counter. Validate positive stride/top-K and non-empty text in `SliceRequest.__post_init__`.

- [ ] **Step 4: Run focused and protocol tests**

Run: `python -m pytest tests/test_fake_services.py tests/test_domain.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jstudio/services/protocols.py jstudio/services/fake.py tests/test_fake_services.py
git commit -m "feat: define asynchronous J-Lens slice service"
```

### Task 2: Real `jlens.vis` Slice Adapter

**Files:**
- Create: `jstudio/services/slice_runtime.py`
- Modify: `jstudio/services/hf_runtime.py`
- Test: `tests/test_slice_runtime.py`

**Interfaces:**
- Consumes: `SliceRequest`, `SlicePage`, `HFModelRuntime.lens_model`, `HFModelRuntime.lens`
- Produces: `render_slice(runtime, request, generation) -> SlicePage`
- Produces: `HFLensService.request_slice(request) -> Future[SlicePage]`

- [ ] **Step 1: Write failing adapter tests using a fake `jlens.vis` module**

```python
def test_render_slice_uses_original_renderer(monkeypatch):
    calls = []
    monkeypatch.setattr(slice_runtime, "compute_slice", lambda *a, **kw: calls.append(kw) or object())
    monkeypatch.setattr(slice_runtime, "build_page", lambda *a, **kw: ("<html>original</html>", None, None))
    page = render_slice(RuntimeDouble(), SliceRequest("r", "ascii", "ASCII"), 7)
    assert page.html == "<html>original</html>"
    assert calls == [{"layer_stride": 1, "last_n_tokens": None, "top_n": 10, "mask_display": True}]
```

```python
def test_real_lens_service_orders_slice_generations(runtime):
    service = HFLensService(runtime)
    one = service.request_slice(SliceRequest("r1", "one", "One")).result(1)
    two = service.request_slice(SliceRequest("r2", "two", "Two")).result(1)
    assert (one.generation, two.generation) == (1, 2)
    service.close()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_slice_runtime.py -q`

Expected: FAIL because `slice_runtime` does not exist.

- [ ] **Step 3: Implement the thin renderer adapter**

```python
def render_slice(runtime, request: SliceRequest, generation: int) -> SlicePage:
    data = compute_slice(
        runtime.lens_model,
        runtime.lens,
        request.text,
        layer_stride=request.layer_stride,
        last_n_tokens=request.last_n_tokens,
        top_n=request.top_n,
        mask_display=request.mask_display,
    )
    page, _, _ = build_page(data, request.text, title=request.title, mode="embed")
    return SlicePage(request.run_id, generation, page)
```

Construct `HFLensService(runtime)` with a single-worker executor. `request_slice` increments its generation under lock and submits `render_slice`. Shut the executor down from `HFGenerationService.close` before releasing the runtime.

- [ ] **Step 4: Run adapter and existing runtime tests**

Run: `python -m pytest tests/test_slice_runtime.py tests/test_hf_runtime.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jstudio/services/slice_runtime.py jstudio/services/hf_runtime.py tests/test_slice_runtime.py tests/test_hf_runtime.py
git commit -m "feat: render original J-Lens slices asynchronously"
```

### Task 3: Locked-Down Web View and Native Bridge

**Files:**
- Create: `jstudio/ui/jlens/web_view.py`
- Test: `tests/ui/test_jlens_web_view.py`

**Interfaces:**
- Produces: `JLensWebView.set_page(page: SlicePage) -> None`
- Produces signals: `coordinate_selected(int, int)`, `term_pinned(str)`, `intervention_requested(str, int, int)`
- Produces bridge slots: `select(position, layer)`, `pin(term)`, `intervene(term, layer, position)`

- [ ] **Step 1: Write failing security and bridge tests**

```python
def test_web_view_loads_html_and_rejects_remote_navigation(qtbot):
    view = JLensWebView()
    qtbot.addWidget(view)
    view.set_page(SlicePage("r", 1, "<html><body>slice</body></html>"))
    assert view.current_generation == 1
    assert not view.page().acceptNavigationRequest(QUrl("https://example.com"), 0, True)

def test_bridge_validates_intervention_payload(qtbot):
    bridge = JLensBridge()
    with qtbot.assertNotEmitted(bridge.intervention_requested):
        bridge.intervene("", -1, -1)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_jlens_web_view.py -q`

Expected: FAIL because the web view and bridge do not exist.

- [ ] **Step 3: Implement the bridge and navigation policy**

Use `QWebEnginePage`, `QWebEngineView`, `QWebChannel`, and `@Slot`. Allow only initial `data:`/`about:blank` loads and the renderer's already-embedded assets. Reject `http`, `https`, `file`, and custom schemes. Register exactly one object named `jstudioBridge`; validate non-empty terms and non-negative integer coordinates before emitting.

- [ ] **Step 4: Add the minimal page bootstrap**

After `setHtml`, inject a small script that forwards the renderer's selection and pin custom events when present. The original page must remain fully functional if no event is emitted.

```javascript
window.addEventListener("jlens-select", e =>
  window.jstudioBridge?.select(e.detail.position, e.detail.layer));
```

- [ ] **Step 5: Run WebEngine tests**

Run: `QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu python -m pytest tests/ui/test_jlens_web_view.py -q`

Expected: PASS without remote navigation.

- [ ] **Step 6: Commit**

```bash
git add jstudio/ui/jlens/web_view.py tests/ui/test_jlens_web_view.py
git commit -m "feat: host J-Lens renderer in a secure web view"
```

### Task 4: Replace the Approximate Workspace

**Files:**
- Rewrite: `jstudio/ui/jlens/workspace.py`
- Modify: `jstudio/ui/shell/main_window.py`
- Modify: `tests/ui/test_jlens_workspace.py`
- Modify: `tests/ui/test_end_to_end.py`

**Interfaces:**
- Consumes: `JStudioServices.lens.request_slice`, `JLensWebView`
- Produces: `JLensWorkspace.inspect(run_id, text, title, position=0) -> None`
- Preserves: `selection`, `intervention_requested`

- [ ] **Step 1: Replace old workspace assertions with failing renderer assertions**

```python
def test_workspace_uses_original_slice_renderer(qtbot, services):
    workspace = JLensWorkspace(services)
    qtbot.addWidget(workspace)
    workspace.inspect("run-1", "( ^ )", "ASCII face", position=1)
    qtbot.waitUntil(lambda: workspace.web.current_generation == 1)
    assert workspace.web.isVisibleTo(workspace)
    assert workspace.status.text() == "Ready"
    assert not hasattr(workspace, "matrix")
```

```python
def test_workspace_discards_stale_slice_result(qtbot, deferred_slice_service):
    workspace = JLensWorkspace(deferred_slice_service.services)
    qtbot.addWidget(workspace)
    workspace.inspect("old", "old", "Old")
    workspace.inspect("new", "new", "New")
    deferred_slice_service.complete("new", generation=2, html="new-page")
    deferred_slice_service.complete("old", generation=1, html="old-page")
    qtbot.waitUntil(lambda: workspace.web.current_generation == 2)
    assert workspace.web.last_html == "new-page"
```

- [ ] **Step 2: Run tests and verify RED**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_jlens_workspace.py -q`

Expected: FAIL against the native matrix workspace.

- [ ] **Step 3: Implement the focused workspace**

Keep a compact native toolbar with title, run summary, Refresh, Export, and intervention actions. The main area is only `JLensWebView`. `inspect` stores the request, shows a non-destructive `Loading slice…` state, attaches a done callback through a Qt signal, and accepts a result only when its generation matches the latest request.

- [ ] **Step 4: Route run text from the shell**

In `_inspect_run`, find the immutable project run, combine prompt and output without losing whitespace, and call:

```python
self.jlens_workspace.inspect(
    run_id,
    run.prompt + ("\n\n" + run.output_text if run.output_text else ""),
    "Chat inspection",
    position=position,
)
```

Construct the workspace as `JLensWorkspace(services, self)`.

- [ ] **Step 5: Run workspace and end-to-end tests**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_jlens_workspace.py tests/ui/test_end_to_end.py tests/ui/test_chat.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jstudio/ui/jlens/workspace.py jstudio/ui/shell/main_window.py tests/ui/test_jlens_workspace.py tests/ui/test_end_to_end.py
git commit -m "feat: replace J-Lens tab with original slice renderer"
```

### Task 5: Visual and Full-Suite Verification

**Files:**
- Modify: `tests/ui/test_visual_smoke.py`
- Modify: `README.md`

**Interfaces:**
- Verifies all prior tasks; produces no new runtime API.

- [ ] **Step 1: Add a failing visual contract**

```python
def test_jlens_visual_contract(window):
    workspace = window.jlens_workspace
    assert isinstance(workspace.web, JLensWebView)
    assert workspace.sizeHint().width() >= 1200
    assert workspace.sizeHint().height() >= 780
    assert workspace.web.accessibleName() == "Interactive J-Lens slice"
    assert workspace.refresh_button.accessibleName()
    assert workspace.export_button.accessibleName()
```

- [ ] **Step 2: Run the visual contract and verify RED if coverage is missing**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_visual_smoke.py -q`

- [ ] **Step 3: Make the smallest accessibility/layout corrections and document the renderer**

README must state that J-Lens uses the repository's original interactive slice page and full-vocabulary ranks, not compressed confidence scores.

- [ ] **Step 4: Run complete verification**

Run: `QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu python -m pytest -q`

Expected: all tests PASS.

Run: `python -m ruff check jstudio tests`

Expected: `All checks passed!`

- [ ] **Step 5: Run a real ASCII-face smoke capture**

Launch J Studio with the fitted Qwen lens, inspect the repository's ASCII-face prompt, and save a screenshot. Verify the screenshot contains the layer×position matrix, spatial prompt, By Layer, By Position, rank heatmap, and both rank plots.

- [ ] **Step 6: Commit**

```bash
git add tests/ui/test_visual_smoke.py README.md
git commit -m "test: verify exact J-Lens visualization"
```
