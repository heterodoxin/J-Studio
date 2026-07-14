# J Studio Desktop UI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the complete J Studio PySide6 desktop workbench specified in `docs/superpowers/specs/2026-07-07-j-studio-ui-design.md`, with service-driven model/lens workflows, a Cheat Engine-shaped main window, dense J-lens analysis, intervention controls, projects, secondary research tools, and a fail-closed spawned QuickJS rules sandbox.

**Architecture:** `jstudio` is a standalone Python package whose immutable domain records and service protocols do not import Qt. PySide6 views consume service signals and item models; they never perform inference, lens computation, hidden-state access, or intervention math. Rules run in a new spawned worker for every evaluation and return validated declarative actions through a bounded JSON protocol.

**Tech Stack:** Python 3.12+, PySide6 / Qt 6, quickjs, multiprocessing `spawn`, pytest, pytest-qt, Ruff, setuptools

## Global Constraints

- Initial main-window geometry is 734 × 592 logical pixels and startup tab is Main.
- Permanent native document tabs are Main, Chat, J-Lens, and Rules.
- The main UI uses native Qt widgets and the operating-system palette; no dashboard navigation or custom default skin.
- Baseline runs are immutable and intervention runs are distinct records.
- No view performs inference, lens computation, hidden-state access, or intervention math.
- Large activation, trace, rule, and experiment data uses Qt item models or custom painting, never widget-per-row.
- UI progress updates are capped at 10 Hz and all service work stays off the GUI thread.
- Rule execution uses pip-installed `quickjs`, a new spawned process per evaluation, fixed resource limits, bounded JSON, declarative output, and fail-closed behavior.
- Imported interventions are disarmed and imported rules are disabled/untrusted.
- Default generation mode is Baseline.
- Strings shown to users are wrapped with Qt translation calls and identifiers use monospace presentation.

---

## File structure

- Create `pyproject.toml`: package metadata, Python/dependency floors, `j-studio` entry point, pytest/Ruff configuration.
- Create `jstudio/domain.py`: immutable UI-facing records and enums.
- Create `jstudio/services/protocols.py`: replaceable model/lens/generation/project/sandbox protocols and event callbacks.
- Create `jstudio/services/fake.py`: deterministic async fake services used by the app demo and UI tests.
- Create `jstudio/project.py`: safe JSON project schema, import hardening, load/save.
- Create `jstudio/rules/protocol.py`, `validation.py`, `worker.py`, `sandbox.py`: bounded request/action records, source/action validation, spawned QuickJS worker, parent timeout/failure policy.
- Create `jstudio/ui/models.py`: virtualized table/tree models for activations, interventions, sessions, rules, traces, and experiments.
- Create `jstudio/ui/shell/main_window.py`, `commands.py`, `session_bar.py`, `activity.py`: window, menus, state gating, identity strip, activity reporting.
- Create `jstudio/ui/main_workspace.py`: found concepts/read controls/scanner/intervention list.
- Create `jstudio/ui/interventions/editor.py`, `stack.py`: validated modeless editor, ordered stack, preview/arm/apply states.
- Create `jstudio/ui/chat.py`: transcript, streaming composer, inspection actions.
- Create `jstudio/ui/jlens/workspace.py`, `matrix.py`, `plots.py`: synchronized six-region J-lens analysis surface.
- Create `jstudio/ui/rules/workspace.py`, `editor.py`: rule list/editor/configuration/API/Test Bench/logs.
- Create `jstudio/ui/secondary.py`: Model View, Layer Explorer, Influence Trace, Generation Trace, Experiments, Snapshot Manager.
- Create `jstudio/ui/sessions/picker.py`: local/worker/offline session picker and capability preview.
- Create `jstudio/ui/settings.py`, `theme.py`, `accessibility.py`: settings categories, palette modes, scaling/focus/accessibility helpers.
- Create `jstudio/ui/app.py`, `jstudio/__main__.py`: application construction and command-line launch.
- Create `tests/`: domain, project, sandbox, Qt model, widget, workflow, accessibility, and visual smoke tests.

### Task 1: Package, domain records, and service boundary

**Files:**
- Create: `pyproject.toml`
- Create: `jstudio/__init__.py`
- Create: `jstudio/domain.py`
- Create: `jstudio/services/__init__.py`
- Create: `jstudio/services/protocols.py`
- Create: `tests/test_domain.py`

**Interfaces:**
- Produces: `ModelSessionSummary`, `ConceptActivation`, `InterventionDraft`, `InterventionEntry`, `RunRecord`, `JLensFrame`, `RuleRecord`, `ActivityRecord`, `ExperimentRecord`.
- Produces: `JStudioServices` aggregate containing `sessions`, `generation`, `lens`, `interventions`, and `rules` protocols.
- Enforces immutable baseline run records and capability/state literals.

- [ ] **Step 1: Write failing tests for immutable records and baseline separation**

```python
def test_run_records_are_immutable_and_modes_are_distinct():
    baseline = RunRecord.create(prompt="hello", mode=RunMode.BASELINE)
    controlled = baseline.derive(mode=RunMode.WITH_STACK, intervention_ids=("i1",))
    assert baseline.run_id != controlled.run_id
    assert baseline.intervention_ids == ()
    with pytest.raises(FrozenInstanceError):
        baseline.prompt = "changed"

def test_offline_session_capabilities_disable_mutation():
    session = ModelSessionSummary.offline_trace("trace-1", layers=64)
    assert session.capabilities.inspect
    assert not session.capabilities.generate
    assert not session.capabilities.intervene
```

- [ ] **Step 2: Run tests and verify imports fail**

Run: `python -m pytest tests/test_domain.py -q`
Expected: collection failure for missing `jstudio.domain`.

- [ ] **Step 3: Implement validated frozen domain records**

Use `Enum` subclasses for session/run/intervention/rule states, `@dataclass(frozen=True, slots=True)` records, UUID factory methods, finite numeric validation, layer-bound validation, and tuples/mapping proxies rather than mutable containers. `RunRecord.derive()` must always allocate a new ID and preserve `baseline_run_id`.

- [ ] **Step 4: Implement UI-facing protocols without Qt imports**

```python
class GenerationService(Protocol):
    def start(self, request: GenerationRequest, sink: GenerationEventSink) -> str:
        raise NotImplementedError
    def pause(self, run_id: str) -> None:
        raise NotImplementedError
    def resume(self, run_id: str) -> None:
        raise NotImplementedError
    def next_token(self, run_id: str) -> None:
        raise NotImplementedError
    def stop(self, run_id: str) -> None:
        raise NotImplementedError

@dataclass(frozen=True, slots=True)
class JStudioServices:
    sessions: SessionService
    generation: GenerationService
    lens: LensService
    interventions: InterventionService
    rules: RuleSandboxProtocol
```

- [ ] **Step 5: Run domain tests and commit**

Run: `python -m pytest tests/test_domain.py -q`
Expected: all tests pass.

```bash
git add pyproject.toml jstudio tests/test_domain.py
git commit -m "feat: scaffold J Studio domain and service contracts"
```

### Task 2: Safe project model and deterministic fake services

**Files:**
- Create: `jstudio/project.py`
- Create: `jstudio/services/fake.py`
- Create: `tests/test_project.py`
- Create: `tests/test_fake_services.py`

**Interfaces:**
- Produces: `ProjectDocument.new()`, `.load(path)`, `.save(path)`, `.to_dict()`.
- Produces: `FakeSessionService`, `FakeGenerationService`, `FakeLensService`, `FakeInterventionService`.
- Consumes domain records and protocol callback interfaces from Task 1.

- [ ] **Step 1: Write project-import hardening tests**

```python
def test_import_disarms_interventions_and_distrusts_rules(tmp_path):
    path = tmp_path / "imported.jstudio.json"
    path.write_text(json.dumps(IMPORTED_PROJECT))
    project = ProjectDocument.load(path, imported=True)
    assert all(not entry.enabled and entry.state is InterventionState.DRAFT for entry in project.interventions)
    assert all(not rule.enabled and not rule.trusted for rule in project.rules)

def test_project_rejects_secrets_and_unknown_schema(tmp_path):
    with pytest.raises(ProjectFormatError):
        ProjectDocument.from_dict({"schema": 99, "access_token": "secret"})
```

- [ ] **Step 2: Run tests and verify missing implementation**

Run: `python -m pytest tests/test_project.py tests/test_fake_services.py -q`
Expected: missing-module failures.

- [ ] **Step 3: Implement versioned atomic project JSON**

Schema version `1` stores only session descriptors without credentials, prompt library, sweep configs, intervention/rule records, experiment definitions, trace references, and layout preferences. Save through a sibling `.tmp` followed by `Path.replace`. Reject unknown top-level keys, absolute credential fields, non-finite JSON values, and payloads above 16 MiB.

- [ ] **Step 4: Implement deterministic fake async services**

Use a single `ThreadPoolExecutor(max_workers=2)` owned by `FakeGenerationService`. Stream token/frame events at configurable intervals through the sink, implement pause/resume/next/stop with `threading.Event`, and provide deterministic 64-layer concept frames. Fake services must never import Qt and expose `close()` for test cleanup.

- [ ] **Step 5: Run service/project tests and commit**

Run: `python -m pytest tests/test_project.py tests/test_fake_services.py -q`
Expected: all tests pass.

```bash
git add jstudio/project.py jstudio/services/fake.py tests/test_project.py tests/test_fake_services.py
git commit -m "feat: add safe projects and fake J Studio services"
```

### Task 3: Qt item models, command state, and native application shell

**Files:**
- Create: `jstudio/ui/__init__.py`
- Create: `jstudio/ui/models.py`
- Create: `jstudio/ui/shell/__init__.py`
- Create: `jstudio/ui/shell/commands.py`
- Create: `jstudio/ui/shell/session_bar.py`
- Create: `jstudio/ui/shell/activity.py`
- Create: `jstudio/ui/shell/main_window.py`
- Create: `tests/ui/test_models.py`
- Create: `tests/ui/test_shell.py`

**Interfaces:**
- Produces: virtualized `ActivationTableModel`, `InterventionTreeModel`, `SessionTableModel`, `RuleTableModel`, `TraceEventModel`, `ExperimentRunModel`.
- Produces: `CommandRegistry.refresh(AppState)` and `JStudioMainWindow(services, project)`.

- [ ] **Step 1: Write failing model and shell tests**

```python
def test_main_window_geometry_tabs_and_native_structure(qtbot, services):
    window = JStudioMainWindow(services, ProjectDocument.new())
    qtbot.addWidget(window)
    assert window.size() == QSize(734, 592)
    assert [window.tabs.tabText(i) for i in range(4)] == ["Main", "Chat", "J-Lens", "Rules"]
    assert window.tabs.currentWidget() is window.main_workspace
    assert window.findChild(QDockWidget) is None

def test_activation_model_updates_without_row_widgets(qtbot):
    model = ActivationTableModel()
    model.replace_rows(FAKE_ACTIVATIONS)
    assert model.rowCount() == len(FAKE_ACTIVATIONS)
    assert model.data(model.index(0, 0), Qt.DisplayRole) == "injection"
```

- [ ] **Step 2: Run tests and verify missing Qt classes**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_models.py tests/ui/test_shell.py -q`
Expected: collection failures for missing UI modules.

- [ ] **Step 3: Implement virtualized models and roles**

Use `QAbstractTableModel`/`QAbstractItemModel`, `beginResetModel` for batch replacement, precise `dataChanged` for streaming score updates, check-state roles for intervention/rule enablement, accessible descriptions, stable IDs in `Qt.UserRole`, and no child widget creation.

- [ ] **Step 4: Implement command registry and menus**

Create the exact File/Edit/Model/Table/Tools/Help menus and shortcuts from Sections 7 and 21. `CommandRegistry.refresh()` applies session/lens/offline/running/armed/dirty-rule prerequisites and sets explanatory status tips on disabled actions.

- [ ] **Step 5: Implement shell and identity strip**

`JStudioMainWindow` uses a native `QTabWidget`, 734×592 initial size, a 36 px `SessionBar`, permanent tabs, status bar, and activity button. Store per-tab remembered geometry; J-Lens may widen to 1200×780 and returning to Main restores the compact geometry until Main is explicitly resized.

- [ ] **Step 6: Run shell tests and commit**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_models.py tests/ui/test_shell.py -q`
Expected: all tests pass.

```bash
git add jstudio/ui tests/ui/test_models.py tests/ui/test_shell.py
git commit -m "feat: build native J Studio shell and item models"
```

### Task 4: Session picker, Main scanner, and intervention workflow

**Files:**
- Create: `jstudio/ui/sessions/__init__.py`
- Create: `jstudio/ui/sessions/picker.py`
- Create: `jstudio/ui/main_workspace.py`
- Create: `jstudio/ui/interventions/__init__.py`
- Create: `jstudio/ui/interventions/editor.py`
- Create: `jstudio/ui/interventions/stack.py`
- Create: `tests/ui/test_main_workflow.py`
- Create: `tests/ui/test_intervention_editor.py`
- Create: `tests/ui/test_session_picker.py`

**Interfaces:**
- Produces: `SessionPickerDialog`, `MainReadWorkspace`, `InterventionEditor`, `InterventionStackView`.
- Emits validated generation/read/intervention requests through service protocols.

- [ ] **Step 1: Write first-read lifecycle and editor tests**

```python
def test_first_read_pause_next_resume_stop(qtbot, window, fake_generation):
    qtbot.keyClicks(window.main_workspace.prompt, "Inspect this prompt")
    qtbot.mouseClick(window.main_workspace.first_read, Qt.LeftButton)
    qtbot.waitUntil(lambda: window.main_workspace.run_state == RunState.RUNNING)
    assert window.main_workspace.button_labels() == ("Pause", "Next Token", "Stop")
    qtbot.mouseClick(window.main_workspace.first_read, Qt.LeftButton)
    qtbot.mouseClick(window.main_workspace.next_read, Qt.LeftButton)
    qtbot.mouseClick(window.main_workspace.undo_read, Qt.LeftButton)
    assert window.main_workspace.button_labels() == ("First Read", "Next Read", "Undo Read")

def test_replace_editor_validates_and_adds_draft(qtbot, window):
    editor = window.main_workspace.open_intervention_editor("replace", "injection")
    editor.target_term.setText("trusted")
    editor.strength.setValue(0.8)
    qtbot.mouseClick(editor.add_button, Qt.LeftButton)
    entry = window.project.interventions[-1]
    assert entry.operation is InterventionOperation.REPLACE
    assert entry.source_term == "injection" and entry.target_term == "trusted"
```

- [ ] **Step 2: Run focused tests and verify failures**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_main_workflow.py tests/ui/test_intervention_editor.py tests/ui/test_session_picker.py -q`
Expected: missing-widget failures.

- [ ] **Step 3: Implement the 46/54 scanner layout**

Use a vertical splitter with a 7 px handle. Upper `QSplitter` defaults to 46% Found Concepts and 54% Read Controls. Implement Term/Score/Previous, signed score delegate with zero line, exact First/Next/Undo relabeling, two-line prompt plus ellipsis editor, read/concept types, layer bounds, Positive/Negative/Injected, Fast Read/Layer Step, Pause Generation, Model View, and manual add.

- [ ] **Step 4: Implement result and stack interactions**

Double-click adds an intervention draft. Context menu provides Inject/Replace/Suppress/Model View/Trace/Copy. Lower tree-table has Enabled/Operation/Match/Result/Strength/Layers/Duration/Trigger/Status, reordering/grouping, Preview/Arm/Clear, queued/applied acknowledgement, and immutable baseline-vs-controlled run creation.

- [ ] **Step 5: Implement modeless intervention editor**

Build the 620×620 editor with operation-dependent fields, regex complexity validation, finite strength bounds from session capabilities, All/Range/Selected Layers, durations, triggers, plain-language preview, Save Draft/Preview/Add. Offline sessions allow drafts and preview but disable Apply/Arm with explicit repair text.

- [ ] **Step 6: Implement session picker**

Build 820×580 dialog with search, Local/Remote/Offline tabs, item models, detail/capability preview, F5 refresh, Enter/double-click open, and persistent error/retry state. No credential values enter the session summary.

- [ ] **Step 7: Run workflow tests and commit**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_main_workflow.py tests/ui/test_intervention_editor.py tests/ui/test_session_picker.py -q`
Expected: all tests pass.

```bash
git add jstudio/ui/main_workspace.py jstudio/ui/interventions jstudio/ui/sessions tests/ui
git commit -m "feat: implement J Studio read and intervention workflow"
```

### Task 5: Chat and synchronized J-Lens analysis surface

**Files:**
- Create: `jstudio/ui/chat.py`
- Create: `jstudio/ui/jlens/__init__.py`
- Create: `jstudio/ui/jlens/workspace.py`
- Create: `jstudio/ui/jlens/matrix.py`
- Create: `jstudio/ui/jlens/plots.py`
- Create: `tests/ui/test_chat.py`
- Create: `tests/ui/test_jlens_workspace.py`

**Interfaces:**
- Produces: `ChatWorkspace`, `JLensWorkspace`, shared `JLensSelectionModel(run_id, position, layer, pinned_terms)`.
- Consumes immutable run/frame data and generation service events.

- [ ] **Step 1: Write streaming chat and six-region synchronization tests**

```python
def test_chat_send_stop_and_inspect(qtbot, window):
    window.tabs.setCurrentWidget(window.chat_workspace)
    qtbot.keyClicks(window.chat_workspace.composer, "Explain the diagram")
    qtbot.mouseClick(window.chat_workspace.send_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: window.chat_workspace.transcript_model.rowCount() >= 2)
    window.chat_workspace.inspect_message(1)
    assert window.tabs.currentWidget() is window.jlens_workspace

def test_matrix_selection_updates_all_regions(qtbot, jlens_workspace):
    jlens_workspace.set_run(FAKE_RUN_WITH_MATRIX)
    jlens_workspace.selection.select(position=28, layer=42)
    assert jlens_workspace.by_layer.selected_layer == 42
    assert jlens_workspace.by_position.selected_position == 28
    assert jlens_workspace.matrix.selected == (28, 42)
    assert jlens_workspace.layer_plot.crosshair == 42
```

- [ ] **Step 2: Run tests and verify missing workspaces**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_chat.py tests/ui/test_jlens_workspace.py -q`
Expected: missing-class failures.

- [ ] **Step 3: Implement Chat workspace**

Use a virtualized transcript model/delegate, fixed multiline composer, Send/Stop, generation settings, per-message Copy/Regenerate/Continue/Compare/Add-to-Prompt/Inspect actions, intervention/rule event markers, and active-control status linked to Main. Never overlay J-space terms on visible response text.

- [ ] **Step 4: Implement custom-painted matrix and plots**

`JLensMatrixView` subclasses `QAbstractScrollArea`, paints visible cells only, supports click/hover/shift-pin/double-click, keyboard selection, Ctrl/Shift wheel scrubbing, and accessible coordinate descriptions. `RankPlot` and `PinnedHeatmap` use `QPainter`, not per-point widgets.

- [ ] **Step 5: Implement synchronized six-region layout**

Create matrix, structured input/output text, By Layer, By Position, pinned-term heatmap, and two rank plots around one selection model. Add pin chips, context intervention actions, Single/Baseline-vs-Intervention/Run-vs-Run/Layer-vs-Layer modes, labeled subtraction direction, navigation history, Go To controls, export selection, and provenance details.

- [ ] **Step 6: Run Chat/J-Lens tests and commit**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_chat.py tests/ui/test_jlens_workspace.py -q`
Expected: all tests pass.

```bash
git add jstudio/ui/chat.py jstudio/ui/jlens tests/ui/test_chat.py tests/ui/test_jlens_workspace.py
git commit -m "feat: add Chat and synchronized J-Lens analysis"
```

### Task 6: Fail-closed QuickJS rules protocol and sandbox

**Files:**
- Create: `jstudio/rules/__init__.py`
- Create: `jstudio/rules/protocol.py`
- Create: `jstudio/rules/validation.py`
- Create: `jstudio/rules/worker.py`
- Create: `jstudio/rules/sandbox.py`
- Create: `tests/rules/test_validation.py`
- Create: `tests/rules/test_sandbox.py`

**Interfaces:**
- Produces: `SandboxLimits`, `RuleEvaluationRequest`, `RuleEvaluationResult`, `ValidatedRuleAction`.
- Produces: `QuickJSSandbox.evaluate(request, *, cancel_event=None) -> RuleEvaluationResult`.
- Enforces source/input/output/action/log/time/heap/stack/address-space bounds from Section 15.

- [ ] **Step 1: Write source and action validation tests**

```python
@pytest.mark.parametrize("forbidden", ["eval(", "Function(", "import(", "async function", "function*", "WebAssembly", "Date.now", "Math.random"])
def test_forbidden_source_fails_closed(forbidden):
    result = validate_rule_source(f"function run(ctx) {{ {forbidden}; return []; }}")
    assert not result.valid

def test_action_validator_rejects_nonfinite_strength_and_extra_keys():
    rejected = validate_actions([{"type": "inject", "term": "x", "strength": float("inf"), "escape": True}], layer_count=64)
    assert rejected.validated == () and len(rejected.rejected) == 1
```

- [ ] **Step 2: Write spawned-worker failure tests**

Cover valid inject/replace/suppress/log/tag/stop, malformed JSON, source/input/output limits, infinite loop, recursion, heap pressure, forbidden globals, dynamic import/eval, 33 actions, invalid layers/strength, worker crash, 50 ms parent timeout, and parent cancellation. Every failure asserts `actions == ()`.

- [ ] **Step 3: Run sandbox tests and verify missing implementation**

Run: `python -m pytest tests/rules -q`
Expected: missing-module failures.

- [ ] **Step 4: Implement bounded protocol and validators**

Use strict dataclass-to-JSON conversion with `allow_nan=False`, exact-key schemas, UTF-8 byte counts, maximum nesting depth 32, maximum string length 16 KiB, output/action/log caps, layer/duration/match-mode validation, and deterministic conflict ordering Stop→Replace→Suppress→Inject→Tag/Log.

- [ ] **Step 5: Implement one-shot spawned QuickJS worker**

Use `multiprocessing.get_context("spawn")`, a unidirectional pipe, and one request/response per child. In the child, apply `resource` limits where supported, clear environment values, set QuickJS memory/stack/time limits, expose only JSON-frozen `ctx`, `jspace`, `generation`, and `rule`, remove dynamic code/clock/randomness capabilities, evaluate validated source, JSON-stringify `run(ctx)`, send one bounded response, and exit.

- [ ] **Step 6: Implement parent timeout/cancellation/failure policy**

Terminate and join on 50 ms timeout or cancellation, reject malformed/oversized responses, never reuse workers, track consecutive rule failures, and return a sandbox-paused protocol error after a worker crash. No partial action survives a protocol failure.

- [ ] **Step 7: Run sandbox tests and commit**

Run: `python -m pytest tests/rules -q`
Expected: all tests pass.

```bash
git add jstudio/rules tests/rules
git commit -m "feat: add isolated fail-closed QuickJS rules sandbox"
```

### Task 7: Rules UI and conflict/test workflows

**Files:**
- Create: `jstudio/ui/rules/__init__.py`
- Create: `jstudio/ui/rules/workspace.py`
- Create: `jstudio/ui/rules/editor.py`
- Create: `tests/ui/test_rules_workspace.py`

**Interfaces:**
- Produces: `RulesWorkspace`, `RuleSourceEditor`, `RuleTestBench`.
- Consumes `RuleSandboxProtocol` asynchronously and never imports `quickjs`.

- [ ] **Step 1: Write current-test gating and auto-disable tests**

```python
def test_edited_rule_requires_current_successful_test_before_enable(qtbot, rules_workspace):
    rules_workspace.new_rule("Guard")
    rules_workspace.editor.setPlainText(VALID_RULE)
    assert not rules_workspace.enable_action.isEnabled()
    qtbot.mouseClick(rules_workspace.test_button, Qt.LeftButton)
    qtbot.waitUntil(lambda: rules_workspace.last_test_passed)
    assert rules_workspace.enable_action.isEnabled()
    rules_workspace.editor.insertPlainText("\n// changed")
    assert not rules_workspace.enable_action.isEnabled()

def test_three_failures_disable_rule(rules_workspace):
    rule = rules_workspace.add_rule(VALID_RULE, enabled=True)
    for _ in range(3):
        rules_workspace.record_execution_failure(rule.rule_id, "timeout")
    assert not rules_workspace.rule(rule.rule_id).enabled
```

- [ ] **Step 2: Run tests and verify missing workspace**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_rules_workspace.py -q`
Expected: missing-class failure.

- [ ] **Step 3: Implement three-pane Rules workspace**

Build 260 px rule list, flexible source editor with line-number gutter/highlighting/bracket matching/find-format/diagnostics, 340 px Configuration/API/Test tabs, and 200 px Problems/Returned Actions/Execution Log. Implement New/Folder/Save/Enable/Disable/Duplicate/Export/Import/Delete, Ctrl+S and Ctrl+Enter.

- [ ] **Step 4: Implement Test Bench and conflict preview**

Run sandbox work in a non-GUI executor, show raw JSON, validated/rejected actions, stack conflicts, time/memory/input/output/log metrics, require a successful test of the exact source/config hash before enablement, and import rules as disabled/untrusted.

- [ ] **Step 5: Run Rules UI tests and commit**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_rules_workspace.py -q`
Expected: all tests pass.

```bash
git add jstudio/ui/rules tests/ui/test_rules_workspace.py
git commit -m "feat: build J Studio rules workspace"
```

### Task 8: Secondary research tools, settings, activity, and persistence

**Files:**
- Create: `jstudio/ui/secondary.py`
- Create: `jstudio/ui/settings.py`
- Create: `jstudio/ui/theme.py`
- Create: `jstudio/ui/accessibility.py`
- Create: `tests/ui/test_secondary_tools.py`
- Create: `tests/ui/test_settings_accessibility.py`
- Create: `tests/ui/test_project_workflow.py`

**Interfaces:**
- Produces Model View, Layer Explorer, Influence Trace, Generation Trace, Experiments, Snapshot Manager, Settings, Activity views.
- Persists geometry/splitters/columns and project documents without secrets.

- [ ] **Step 1: Write secondary-window and imported-project tests**

```python
def test_tools_open_focused_secondary_windows(qtbot, window):
    for command in ("model_view", "layer_explorer", "influence_trace", "generation_trace", "experiments", "snapshot_manager"):
        window.commands[command].trigger()
        tool = window.tool_window(command)
        assert tool.isVisible() and tool.isWindow()

def test_imported_project_starts_safe(qtbot, window, imported_project_path):
    window.open_project(imported_project_path, imported=True)
    assert all(not row.enabled for row in window.project.interventions)
    assert all(not rule.enabled and not rule.trusted for rule in window.project.rules)
```

- [ ] **Step 2: Run tests and verify missing windows**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_secondary_tools.py tests/ui/test_settings_accessibility.py tests/ui/test_project_workflow.py -q`
Expected: missing-class failures.

- [ ] **Step 3: Implement secondary tools**

Layer Explorer shows selected run/layer/token provenance and intervention entry. Influence Trace provides bounded graph custom painting plus virtualized result table and estimated-influence wording. Generation Trace aligns tokens/frames/manual actions/rules/warnings. Experiments provides Setup/Runs/Compare/Report with immutable baseline pairing. Model View, Advanced Sweep, Snapshot Manager, and report export use service data only.

- [ ] **Step 4: Implement settings, activity, and safe confirmation**

Create 940×720 searchable settings with all eleven categories, read-only rule security limits, Baseline default, light/dark/system palettes, 90–160% scaling, reduced motion, shortcuts, and sandbox self-test. Activity throttles progress to 10 Hz. Confirmation dialogs name operation/terms/strength/layers/duration/rules/mode and default-focus the safest action.

- [ ] **Step 5: Implement layout and project persistence**

Use `QSettings` for window/splitter/column geometry only and `ProjectDocument` for portable content. Add consolidated dirty-document close dialog, atomic save, recent projects, trace/report import/export, and secret redaction in expandable error details.

- [ ] **Step 6: Run secondary/persistence tests and commit**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_secondary_tools.py tests/ui/test_settings_accessibility.py tests/ui/test_project_workflow.py -q`
Expected: all tests pass.

```bash
git add jstudio/ui/secondary.py jstudio/ui/settings.py jstudio/ui/theme.py jstudio/ui/accessibility.py tests/ui
git commit -m "feat: add J Studio research tools and settings"
```

### Task 9: Application entry point, backend adapter seam, and end-to-end workflows

**Files:**
- Create: `jstudio/ui/app.py`
- Create: `jstudio/__main__.py`
- Create: `jstudio/services/jlens_adapter.py`
- Create: `tests/ui/test_end_to_end.py`
- Create: `tests/test_jlens_adapter.py`

**Interfaces:**
- Produces: `create_application(argv, services=None)`, `main()` and optional `JLENS_AVAILABLE` adapter.
- The adapter dynamically imports `jlens`; UI modules remain backend-independent.

- [ ] **Step 1: Write entry-point and canonical-path tests**

Cover: first launch/generate/stream; pause-next-resume-stop; select concept/replace/queued/applied/undo; missing lens recovery; sweep cancellation retained Partial; offline inspection with mutation disabled; baseline vs stack comparison; keyboard-only prompt/pause/next/replace/rule-test; and clean shutdown of executors/workers.

- [ ] **Step 2: Run end-to-end tests and verify failures**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_end_to_end.py tests/test_jlens_adapter.py -q`
Expected: missing entry-point/adapter failures.

- [ ] **Step 3: Implement app construction and optional adapter**

`create_application()` configures high-DPI rounding, organization/application names, translators, system palette, services, project, and main window without blocking. `main()` calls `app.exec()`. `jlens_adapter.py` maps J Studio requests/results to the calibrated intervention API when importable and exposes a precise unavailable capability otherwise; it performs work through executors, never the GUI thread.

- [ ] **Step 4: Run end-to-end tests and commit**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_end_to_end.py tests/test_jlens_adapter.py -q`
Expected: all tests pass.

```bash
git add jstudio/ui/app.py jstudio/__main__.py jstudio/services/jlens_adapter.py tests
git commit -m "feat: complete J Studio application workflows"
```

### Task 10: Visual fidelity, accessibility, packaging, and final verification

**Files:**
- Modify: `README.md`
- Create: `tests/ui/test_visual_smoke.py`
- Create: `tests/ui/test_accessibility.py`
- Create: `tests/test_package.py`

**Interfaces:**
- Produces reproducible offscreen screenshots and installable `j-studio` command.

- [ ] **Step 1: Write visual/accessibility/package assertions**

Assert 734×592 default, 46/54 upper split within 3%, 7 px horizontal splitter, 108 px lower pane, 32 px bottom strip, native palette roles, all icon-only controls have accessible names/tooltips, tab order reaches prompt/read/concept/intervention/rule controls, 160% scaling has no clipped key labels, and wheel/keyboard navigation updates J-Lens selection.

- [ ] **Step 2: Run tests and verify any fidelity failures**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_visual_smoke.py tests/ui/test_accessibility.py tests/test_package.py -q`
Expected: failures identify missing fidelity/accessibility metadata before fixes.

- [ ] **Step 3: Complete README and package metadata**

Document Python 3.12+, `pip install -e .[dev]`, `python -m jstudio`, fake/demo mode, optional calibrated `jlens` adapter, QuickJS security boundary, no Node requirement, test commands, and future repository migration. Include no Anthropic/Neuronpedia affiliation claim.

- [ ] **Step 4: Run format, lint, complete tests, and visual smoke**

Run: `python -m ruff format --check jstudio tests`
Expected: no files need formatting.

Run: `python -m ruff check jstudio tests`
Expected: no diagnostics.

Run: `QT_QPA_PLATFORM=offscreen python -m pytest -q`
Expected: all tests pass.

Run: `QT_QPA_PLATFORM=offscreen python -m jstudio --screenshot /tmp/j-studio-main.png --quit-after 300`
Expected: exit 0 and a non-empty PNG matching the compact Cheat Engine-shaped shell.

- [ ] **Step 5: Verify no forbidden architectural imports**

Run: `rg -n "import (torch|transformers|quickjs)|from (torch|transformers|quickjs)" jstudio/ui`
Expected: no matches.

Run: `rg -n "(open\(|requests\.|urllib|subprocess|socket\.)" jstudio/rules/worker.py`
Expected: no matches.

- [ ] **Step 6: Commit**

```bash
git add README.md pyproject.toml jstudio tests
git commit -m "docs: package and verify J Studio desktop UI"
```
