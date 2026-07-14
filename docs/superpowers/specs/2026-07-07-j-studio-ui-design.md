# J Studio Python UI Design Specification

**Status:** Product and interaction design  
**Target stack:** Python 3.12+, PySide6 / Qt 6  
**Rules runtime:** pip-installed `quickjs`, isolated through Python `multiprocessing`  
**Scope:** User interface, UI-facing contracts, and rule-sandbox boundaries only

## 1. Product Definition

J Studio is a desktop workbench for inspecting and intervening in language-model J-space. It uses a Jacobian lens (J-lens) to show verbalizable concepts present across model layers and allows researchers to inject, replace, suppress, and compare those concepts during controlled generation runs.

The interaction model is inspired by Cheat Engine's productive loop—attach, scan, narrow, save, and manipulate—but the product language and information architecture are native to model interpretability:

| Cheat-tool concept | J Studio concept |
|---|---|
| Process | Model Session |
| Memory scan | J-lens Sweep |
| Found address | Concept Activation |
| Address table | Intervention Stack |
| Memory viewer | Layer Explorer |
| Pointer scan | Influence Trace |
| Debugger | Generation Trace |
| Change value | Inject / Replace / Suppress Concept |
| Script table | Rules |

J-space is not presented as chain-of-thought, consciousness, or a complete record of model reasoning. The UI consistently calls it a set of verbalizable internal representations surfaced by J-lens. Readouts are estimates with provenance and confidence, not ground truth.

## 2. UI-Only Boundary

This document specifies:

- Window structure, widgets, navigation, visual states, interactions, shortcuts, and accessibility.
- UI models for loaded models, prompts, layer readouts, concepts, interventions, rules, traces, errors, and progress.
- Validation and isolation requirements visible at the JavaScript Rules boundary.
- Request/response contracts between the UI and model/lens services.

This document does not specify:

- How Jacobian lenses are fitted, stored, computed, or evaluated.
- How hidden states, gradients, token probabilities, or model weights are obtained.
- The mathematics or implementation of injection, replacement, suppression, or tracing.
- Model inference engines, GPU kernels, quantization, remote protocols, or training.
- Claims that a J-lens readout is a model's literal thought or complete reasoning process.

The UI emits validated requests and renders reported states. Model and interpretability services remain replaceable.

## 3. Product Principles

### 3.1 Inspect before intervening

Read-only exploration is always available before modification. Intervention controls clearly distinguish Preview, Arm, and Apply. No loaded model is modified merely by selecting a concept or opening an editor.

### 3.2 Provenance everywhere

Every readout displays model, revision, lens, prompt run, token position, layer, timestamp, and whether interventions were active. Comparisons never silently combine incompatible runs.

### 3.3 Dense but legible

J Studio is an expert workbench. It favors compact tables, timelines, and plots while maintaining predictable spacing, large enough pointer targets, complete keyboard access, and a comfortable density option.

### 3.4 One workspace, purposeful windows

The application is one `QMainWindow` with dockable panes and central documents. Layer Explorer and Rule Editor may detach as independent windows for multi-monitor work. Short data-entry tasks use dialogs.

### 3.5 Automation is declarative and contained

JavaScript rules can inspect an immutable event snapshot and return a bounded list of intervention declarations. They cannot call Python, Qt, inference services, the filesystem, network, environment, clock, randomness, subprocesses, or one another.

### 3.6 Familiar by direct translation

The main window must feel and behave like Cheat Engine rather than a modern dashboard. Existing Cheat Engine users should recognize target selection, First/Next/Undo, found-results handling, double-click-to-add, the lower table, context menus, the splitter, Advanced Options, and secondary tool windows without learning a new navigation system.

## 4. Cheat Engine Fidelity Contract

### 4.1 Main window

`JStudioMainWindow` is a `QMainWindow` whose default outer size is 734 x 592 logical pixels and default client area is 734 x 572, matching the source `MainUnit.lfm`. It may be resized, but the initial geometry and relative proportions must remain recognizably Cheat Engine.

| Cheat Engine region | J Studio translation | Default geometry |
|---|---|---|
| Main menu | File, Edit, Model, Table, Tools, Help | Native menu bar |
| Process label and toolbar | Selected model label plus Select Model, Open, and Save icons | Top 36 px |
| Found list | Concept results table | Left 46%, upper region |
| Scan controls | Prompt/read controls and J-Space Scan Options | Right 54%, upper region |
| Horizontal splitter | Resizes upper results and lower interventions | 7 px |
| Address list | Intervention List tree-table | Lower region, 108 px initially |
| Bottom command strip | Advanced Options at left, Rules at right | 32 px |

There is no navigation rail, dashboard, persistent right Inspector, card grid, or default tab strip in the main window. Secondary tools open in focused windows or dialogs, mirroring Cheat Engine.

### 4.2 Exact main-window translation

| Cheat Engine control | J Studio control |
|---|---|
| No Process Selected | No Model Selected |
| Select a process | Select a Model Session |
| Found: 0 | Found: 0 Concepts |
| Address / Value / Previous | Term / Score / Previous |
| Scan Value | Prompt |
| Scan Type | Read Type |
| Value Type | Concept Type |
| First Scan | First Read |
| Next Scan | Next Read |
| Undo Scan | Undo Read |
| Memory Scan Options | J-Space Scan Options |
| Start / Stop address | Start / Stop layer |
| Writable / Executable / CopyOnWrite | Positive / Negative / Injected |
| Fast Scan / Alignment | Fast Read / Layer Step |
| Pause the game while scanning | Pause generation while reading |
| Memory View | Model View |
| Add Address Manually | Add Intervention Manually |
| Address list | Intervention List |
| Advanced Options | Advanced Options |
| Table Extras | Rules |

The implementation uses native Qt widgets and the operating-system palette. It does not apply a custom dark dashboard skin by default. Dark mode may follow the OS, but control density, borders, menu behavior, button sizes, and table styling remain native.

### 4.3 Document tabs

A native `QTabWidget` directly below the menu bar contains four permanent tabs:

1. **Main:** the Cheat Engine-shaped found-concepts scanner and Intervention List specified above.
2. **Chat:** normal multi-turn conversation with the selected model, including prompt composer, streamed output, Stop, Regenerate, and per-message “Inspect with J-Lens.”
3. **J-Lens:** the dense layer-by-position visualization specified in Section 11.
4. **Rules:** the sandboxed JavaScript rule list, editor, Test Bench, and execution log.

Main remains the startup tab and restores the compact 734 x 592 geometry. Chat may use the compact size or a user-resized window. J-Lens may automatically request a larger remembered geometry because its analysis surface is information-dense; returning to Main restores the compact geometry unless the user explicitly resized Main.

The tabs share one model session, prompt/run history, Intervention List, and rule set. Sending a Chat message creates a run visible to Main and J-Lens. Selecting “Inspect with J-Lens” focuses the corresponding run, token, and frame in the J-Lens tab.

### 4.4 Model toolbar

The top model label is the permanent identity strip for the active target:

- Model icon and display name.
- Model revision.
- Lens name and compatibility status.
- Backend badge: Local, Remote Worker, or Offline Trace.
- Device and precision summary when reported.
- Status: Disconnected, Loading, Ready, Generating, Paused, Failed.
- Select Session button when empty; overflow menu when populated.

Double-clicking the identity reopens the model picker, matching Cheat Engine's process-label behavior. The label never displays raw access tokens, filesystem secrets, or full remote credentials.

### 4.5 Window behavior

The main window remains a single compact tool window. Multiple read sessions may use a small tab strip only when the user explicitly adds one. Secondary tools such as Model View, Layer Explorer, Rules, Influence Trace, and Experiments open their own windows. Geometry, splitter position, table columns, and secondary-window positions persist when enabled.

### 4.6 Interactive visual reference

The implementation should use the [interactive J Studio HTML mockup](../../mockups/j-studio.html) as the relative visual and interaction reference for the main window, Chat tab, J-Lens tab, Rules tab, and Intervention Editor. The design document remains authoritative when mockup text and this specification conflict.

## 5. Visual System

### 5.1 Palette

The default follows the operating system. Light and dark are explicit alternatives.

| Token | Light | Dark | Use |
|---|---:|---:|---|
| `surface.window` | `#F4F6F8` | `#171A1F` | Application background |
| `surface.panel` | `#FFFFFF` | `#20242B` | Panels and dialogs |
| `surface.data` | `#EDF1F5` | `#14171C` | Editors and plots |
| `text.primary` | `#17202A` | `#F2F4F7` | Main text |
| `text.secondary` | `#5F6B78` | `#AAB2BD` | Metadata |
| `accent.read` | `#2563EB` | `#60A5FA` | Read-only inspection |
| `accent.inject` | `#7C3AED` | `#A78BFA` | Injection |
| `accent.replace` | `#C2410C` | `#FB923C` | Replacement |
| `accent.suppress` | `#B91C1C` | `#F87171` | Suppression |
| `state.success` | `#15803D` | `#4ADE80` | Valid/completed |
| `state.warning` | `#B45309` | `#FBBF24` | Warning/conflict |

Color is supplemented by icons and text. Inject uses a plus-in-circle, Replace uses bidirectional arrows, Suppress uses a struck-through concept, and Observe uses an eye.

### 5.2 Typography and spacing

- Interface: platform UI font, nominal 13 px.
- Tables and metadata: 12 px.
- Tokens, terms, JavaScript, layer IDs, and numeric values: platform monospace, 12 px.
- Page heading: 20 px semibold.
- Base spacing: 4 px; standard gap: 8 px; section gap: 16 px; page padding: 16 px.
- Standard control height: 30 px compact and 36 px comfortable.
- Text scaling: 90%–160% without clipping.

### 5.3 Data marks

Activation strength uses a signed horizontal bar with a numeric score. Positive and negative values extend in opposite directions from a visible zero line. Confidence is never encoded by opacity alone; it has a numeric value or Low/Medium/High label. Layers are numbered in model order and never reversed implicitly.

## 6. Python UI Architecture

```text
jstudio/ui/
  app.py
  shell/
    main_window.py
    commands.py
    session_bar.py
    activity_panel.py
  sessions/
    picker.py
    details.py
    models.py
  jlens/
    workspace.py
    sweep_setup.py
    activation_table.py
    timeline.py
  interventions/
    stack.py
    editor.py
    preview.py
    models.py
  layers/
    explorer.py
    heatmap.py
    compare.py
  influence/
    workspace.py
    graph.py
  generation/
    trace.py
    token_timeline.py
  rules/
    workspace.py
    editor.py
    test_bench.py
    api_reference.py
    worker.py
    protocol.py
  experiments/
    workspace.py
    comparison.py
  settings/
  dialogs/
  components/
  resources/
  accessibility.py
  theme.py
```

Windows compose widgets and bind commands. Qt item models are separate. `rules/worker.py` runs only inside a spawned process and has no Qt imports. No view directly calls model inference or reads hidden states.

### 6.1 UI-facing types

```python
@dataclass(frozen=True)
class ModelSessionSummary:
    session_id: str
    model_id: str
    revision: str
    lens_id: str | None
    layer_count: int
    backend_kind: Literal["local", "remote-worker", "offline-trace"]
    state: Literal["loading", "ready", "generating", "paused", "failed"]

@dataclass(frozen=True)
class ConceptActivation:
    term: str
    score: float
    confidence: float | None
    layer: int
    token_index: int
    source: Literal["observed", "injected", "replaced", "rule"]

@dataclass(frozen=True)
class InterventionDraft:
    operation: Literal["inject", "replace", "suppress"]
    source_term: str | None
    target_term: str | None
    strength: float
    layer_start: int
    layer_end: int
    duration: Literal["current-token", "next-token", "generation", "steps"]
    step_count: int | None
```

These are display and request types, not computational representations.

## 7. Global Commands

### 7.1 Menus

- **File:** New Project, Open Project, Open Recent, Save, Save As, Import Trace, Export Report, Close Project, Quit.
- **Edit:** Undo Read, Cut, Copy, Paste, Select All, Settings.
- **Model:** Select Model Session, Recent Models, Load Local Model, Connect Worker, Open Offline Trace, Load Lens, Model Information, Detach.
- **Table:** Add Intervention, Add Group, Add Rule, Activate Selected, Deactivate Selected, Table Files, Export Activations.
- **Tools:** Model View, Layer Explorer, J-Lens Sweep, Influence Trace, Generation Trace, Experiments, Snapshot Manager, Rules.
- **Help:** J-space Concepts, Rule API, Keyboard Reference, Research References, Report Issue, About.

### 7.2 State rules

| Condition | UI result |
|---|---|
| No session | Sweep, generation, layer, trace, and intervention application commands disabled |
| Session without compatible lens | J-Lens actions disabled; Load Lens recovery action shown |
| Offline trace | Inspection enabled; generation and intervention application disabled |
| Sweep running | Setup locked; Run becomes Stop; progress visible in workspace and Activity |
| No activation selected | Create intervention actions disabled |
| Stack contains invalid entries | Arm and Apply disabled; first error linked |
| Stack armed | Active Intervention List rows show checked state and an “Armed” status |
| Generation running | Mutable model/session settings locked; Pause and Stop enabled |
| Rule has unsaved changes | Enable disabled until saved and successfully tested |
| Rules sandbox unavailable | Rules remain editable; execution disabled with repair instructions |

Disabled controls explain their prerequisite: “Load a compatible J-lens to run a sweep,” not merely “Unavailable.”

## 8. Model Session Picker

### 8.1 Layout

`SessionPickerDialog` is 820 x 580 by default, minimum 660 x 440. It contains a search field, tabs for Local Models, Remote Workers, and Offline Traces, a virtualized table, detail preview, and footer actions.

Columns:

| Tab | Columns |
|---|---|
| Local Models | Model, Revision, Parameters, Precision, Device, Lens Status |
| Remote Workers | Worker, Model, Revision, Endpoint Alias, State, Lens Status |
| Offline Traces | Name, Model, Revision, Tokens, Layers, Captured |

The footer contains Manage Models…, Connect Worker…, Open Trace…, Cancel, and Open. Open is disabled until selection is valid. Search receives initial focus. Double-click and Enter open; `F5` refreshes.

### 8.2 Path

1. User selects a row.
2. The preview shows compatibility, expected capabilities, and missing lens warnings.
3. User activates Open.
4. The dialog shows “Opening <model>…” without closing.
5. On success it closes and the centered model label updates.
6. On failure it stays open, preserves selection, and shows Retry and Details.

Remote credentials are collected by a dedicated connection dialog and stored only through the platform credential store. The normal UI uses endpoint aliases.

## 9. Main Read Workspace

The Main Read Workspace is the default main window and directly translates Cheat Engine's scanner. It lets the user enter a prompt, perform a first read, refine or advance the read, inspect found concepts, and add selected concepts to the Intervention List.

### 9.1 Cheat Engine-shaped layout

The upper region preserves the original left-results/right-controls split:

1. **Found Concepts** at left is a report table with Term, Score, and Previous columns plus the `Found: N Concepts` label.
2. **Read Controls** at right contains First Read, Next Read, Undo Read, Prompt, Read Type, and Concept Type in the same vertical order as Cheat Engine's scan controls.
3. **J-Space Scan Options** occupies the lower-right group box with Start Layer, Stop Layer, Positive, Negative, Injected, Fast Read, Layer Step, and Pause Generation While Reading.
4. **Model View** sits below the found list at left and **Add Intervention Manually** sits below the scan options at right.
5. A horizontal splitter separates the upper region from the lower **Intervention List**.
6. The bottom strip contains **Advanced Options** at left and **Rules** at right.

The prompt field is two lines by default rather than Cheat Engine's single-line value field. A small ellipsis button opens a larger prompt editor. This is the only deliberate geometry expansion inside the scan-control column.

### 9.2 Prompt-and-control path

1. User selects a model session or accepts the current session.
2. User types a prompt and selects Generate.
3. First Read begins generation and the J-lens readout.
4. The scan buttons temporarily relabel in place: First Read becomes Pause, Next Read becomes Next Token, and Undo Read becomes Stop.
5. Found Concepts updates at a throttled rate while scores and previous scores remain visible.
6. Double-clicking a found concept adds it to the Intervention List, matching Cheat Engine's double-click-to-address-list behavior.
7. Right-clicking a concept exposes Inject, Replace, Suppress, Model View, Trace Influence, and Copy.
8. Intervention rows can be enabled, disabled, reordered, edited, frozen for a duration, or assigned a rule.
9. When generation stops, the three buttons revert to First Read, Next Read, and Undo Read.
10. The generated response is viewed through Model View or a compact collapsible response pane opened from the found-list context menu.

If the user activates an intervention while generation is running, its Intervention List row shows “Queued for next token” until the service reports it applied. It never labels a request Applied optimistically.

### 9.3 Live J-space panel

Each concept row contains the term, signed score, small trend arrow, and source marker. A selected row expands in place; it does not open a modal by default. Expanded actions are:

- Inject: pre-fills the selected term but permits a different term.
- Replace: uses the selected term as the match and focuses the replacement field.
- Suppress: shows strength and duration only.
- Details: opens the advanced Layer Explorer at the current token/layer.

The panel defaults to Top 8 and has a search field only after the user selects “All concepts.” Numeric layer and confidence data are in the row tooltip and Details view, not persistent columns.

### 9.4 Active Controls tray

The tray is a single-line summary above the Live Control Bar. It shows up to three chips such as `Replace injection → trusted`, `Inject caution`, and `Rule: Prompt guard`. Additional entries collapse behind `+N`. Clicking a chip opens a compact editor. Clear All requires confirmation only while generation is active.

### 9.5 Chat tab

Chat provides a conventional model conversation without requiring the user to understand read settings. The message transcript occupies the main area and the composer remains fixed at the bottom. The composer supports multiline prompts, Send, Stop, attachment/context selection, and a menu for generation settings.

Assistant messages expose Copy, Regenerate, Continue, Compare, Add Output to Prompt, and Inspect with J-Lens. User messages expose Edit and Resend. During streaming, Stop remains visible and the active token is available to J-Lens without overlaying internal data on the conversation text.

Rules and checked Intervention List rows apply to Chat runs. A compact status line above the composer states `Baseline`, `2 interventions active`, or `2 rules + 1 intervention active`; clicking it focuses Main and the corresponding rows. Rule/intervention events appear as collapsible markers between messages, not as hidden behavior.

### 9.6 Advanced sweep

The original full sweep interface remains available under Advanced Tools > J-Lens Sweep. It is intended for offline inspection and research configuration, not the default prompt flow.

#### Advanced sweep layout

The default workspace is a horizontal splitter:

- Left, flexible: activation timeline and activation table.
- Right, 380 px: prompt and sweep configuration.

Across the top is a sweep tab bar. Each tab owns prompt, token selection, layer range, filters, readout, comparison state, and provenance.

#### Advanced sweep setup

Controls appear in this exact order:

1. **Input**
   - Prompt editor, minimum four lines.
   - Optional system prompt disclosure.
   - Tokenize button and token-count label.
2. **Readout Position**
   - Position: Each Token, Selected Token, Last Prompt Token, Generated Tokens.
   - Token selector when Selected Token is active.
3. **Layers**
   - All Layers checkbox.
   - From and To layer spin boxes.
   - Layer sampling: Every Layer or Every N Layers.
4. **Concept Filter**
   - Minimum absolute score.
   - Top concepts per layer.
   - Include terms and Exclude terms chip fields.
   - Show positive, negative, or both.
5. **Run**
   - Primary Run Sweep button.
   - Live Readout toggle.
   - Compare Against selector.

Validation focuses the first invalid field and provides exact helper text. Layer bounds update from the selected session.

#### Advanced timeline

The timeline has token positions on the horizontal axis and layers on the vertical axis. Each cell can show the top concept, score, or compact heat intensity. A mode selector switches among Concepts, Strength, Difference, and Intervention Overlay.

Interactions:

- Hover shows term, signed score, confidence, layer, token, source, and run.
- Click selects a cell and filters the table.
- Drag selects a layer/token region.
- Mouse wheel scrolls; Ctrl+wheel zooms.
- Double-click opens the cell in Layer Explorer.
- Context menu offers Inject Term, Replace Term, Suppress Term, Pin Concept, Trace Influence, Copy Details.

#### Advanced activation table

Default columns are Term, Score, Confidence, Layer, Token, Trend, Source. It is a virtualized `QTableView` with extended selection. Double-click opens Layer Explorer. Enter opens the Intervention Editor with Inject selected.

The table footer shows displayed concepts, total concepts, selected region, and active filters. Sorting never changes the underlying sweep; a visible header marker identifies sort order.

#### Advanced sweep states

| State | Presentation |
|---|---|
| Empty session | Select Session call-to-action |
| Lens missing | Compatibility card with Load Lens |
| Ready | Run Sweep primary; empty readout guidance |
| Running | Setup locked; Stop primary; streaming cells marked provisional |
| Paused | Resume and Stop; provisional data retained |
| Complete | Provenance banner and exact duration |
| Cancelled | Completed portion retained and labeled Partial |
| Failed | Prior valid readout retained; Retry, Edit Setup, Details |

## 10. Intervention Stack

### 10.1 Operation vocabulary

- **Inject:** add a selected term/pattern to J-space with a requested strength, layer scope, and duration.
- **Replace:** match one term/pattern and substitute another within a requested scope.
- **Suppress:** reduce or remove a matched term/pattern within a requested scope.

The UI does not imply token insertion into the visible prompt. Labels always say “J-space Injection” or “J-space Replacement” in full on first use.

### 10.2 Stack layout

The stack is an ordered tree-table with columns Enabled, Operation, Match, Result, Strength, Layers, Duration, Trigger, Status. Groups may contain interventions and rules. A toolbar provides Inject, Replace, Suppress, Group, Duplicate, Move Up, Move Down, Preview, Arm, and Clear.

Rows use these states:

- Draft: gray pencil icon.
- Valid: check icon.
- Conflict: amber split-arrow icon.
- Armed: outlined shield icon.
- Applied: filled operation icon and run reference.
- Failed: error icon with Inspector details.
- Unsupported: disabled row with capability explanation.

### 10.3 Intervention Editor

The modeless editor is 620 x 620 and contains:

1. Operation segmented control: Inject, Replace, Suppress.
2. Match Term, hidden for Inject.
3. Target Term, hidden for Suppress and labeled Injected Term or Replacement Term.
4. Match mode: Exact Term, Case-Insensitive, Regular Expression, Concept ID.
5. Strength slider plus numeric input, valid range reported by session capability.
6. Layer scope: All, Range, Selected Layers.
7. Duration: Current Token, Next Token, N Steps, Entire Generation.
8. Trigger: Manual, Before Token, After Match, Rule.
9. Preview card describing the request in plain language.
10. Save Draft, Preview, and Add to Stack.

Regular expressions are validated before saving and have a visible complexity warning. A Replace entry cannot use an empty replacement. An Inject entry cannot use an empty target term.

### 10.4 Preview and apply path

1. User creates one or more entries.
2. Preview validates capabilities and displays expected affected layers/tokens.
3. Conflicts appear in an ordered list with Resolve links.
4. User selects Arm Stack.
5. A persistent amber Armed banner appears with Review and Disarm.
6. Generation commands explicitly offer Baseline or With Armed Stack.
7. After a run, each applied row links to the Generation Trace and records actual reported status.

Applying an intervention never overwrites the saved baseline. Baseline and intervention runs are distinct immutable run records.

## 11. J-Lens Visualization and Layer Explorer

The J-Lens tab follows the supplied layer/position analysis reference and uses a dense white research-canvas presentation inside the otherwise native desktop shell.

### 11.1 Header and navigation

The header shows run title, prompt summary, selected position, selected layer, whitespace toggle, help, and scrub instructions. Position and layer are keyboard-adjustable. Holding Shift while scrolling scrubs position; holding Ctrl scrubs layer.

### 11.2 Synchronized analytical regions

The tab contains six synchronized regions:

1. **Layer × Position Matrix:** rows are layers, columns are prompt/output positions, and each cell shows the highest-ranked verbalizable term plus compact rank. Selected cells receive a magenta outline.
2. **Input/Output Text:** rendered beneath the matrix with the selected character or token outlined, preserving spatial relationships for ASCII, code, tables, and other structured text.
3. **By Layer:** at the selected position, shows ranked concepts for every layer. The selected layer uses a full-row highlight.
4. **By Position:** at the selected layer, shows ranked concepts for every position. The selected position uses a full-row highlight.
5. **Pinned-Term Heatmap:** color encodes the rank of pinned terms at every `(position, layer)` coordinate. Axes are Position and Layer.
6. **Rank Plots:** separate line plots show each pinned term's rank by layer and by position with crosshairs at the current selection.

Hovering any matrix cell shows a ranked tooltip with term, score/rank, layer, position, token, and provenance. Clicking selects. Shift-click pins a term. Double-click opens the precise Intervention Editor. All panels update from the same `(run, position, layer)` selection model.

### 11.3 Pinned concepts and intervention entry

Pinned terms appear as removable color-coded chips shared by the matrix, heatmap, and plots. Context actions are Inject at Selection, Replace at Selection, Suppress at Selection, Add to Intervention List, Trace Influence, and Copy Coordinates. Inject/Replace never applies directly from the visualization; it opens the precise editor with position and layer scope prefilled.

Modes are Single Run, Baseline vs Intervention, Run vs Run, and Layer vs Layer. Difference mode always labels left/right operands and subtraction direction. The tab supports Back, Forward, Go to Layer, Go to Position, Pin, Export Selection, and Copy Citation.

## 12. Influence Trace

Influence Trace replaces the memory-style pointer scanner metaphor with a graph/timeline for following where a concept appears to propagate.

Setup fields are Seed Term, Seed Layer/Token, Direction, Layer Range, Token Range, Minimum Score, Maximum Nodes, and Compare Against. The result view combines a graph with a sortable table of Term, Source, Destination, Strength, Layer Delta, Token Delta, and Path Length.

Nodes open Layer Explorer. Context actions create Inject, Replace, or Suppress drafts, pin a path, and export selected paths. Large traces stream into a virtualized model and show a hard node cap. The UI labels inferred links as estimated influence rather than causation unless the service explicitly reports a causal intervention result.

## 13. Generation Trace

Generation Trace presents prompt tokens, generated tokens, J-space frames, interventions, rules, and output in one aligned timeline.

The top toolbar contains Generate, One Token, Pause, Resume, Stop, Baseline, With Stack, and Compare. The center token timeline marks:

- Prompt versus generated tokens.
- J-lens frames.
- Manual injections/replacements/suppressions.
- Rule-produced actions.
- Warnings and failures.

Selecting an event opens exact input snapshot, validated action, service result, and provenance in Inspector. Running state keeps prior data visible with a “Generation running” overlay on values that are not yet final.

## 14. Rules Workspace

### 14.1 Purpose

Rules are user-authored JavaScript functions that inspect J Studio event snapshots and return declarative actions. They support dynamic J-space Injection, J-space Replacement, suppression, stop requests, logs, and tags. Rules never receive a live model object or mutation handle.

### 14.2 Layout

The Rules workspace uses a three-pane layout:

- Left, 260 px: rule list and folders.
- Center, flexible: JavaScript editor.
- Right, 340 px: Configuration / API / Test tabs.
- Bottom, 200 px: Problems / Returned Actions / Execution Log tabs.

The rule list columns are Enabled, Name, Trigger, Priority, Last Result, Failures. Toolbar actions are New Rule, New Folder, Save, Enable, Disable, Duplicate, Export, Import, and Delete.

The editor provides line numbers, bracket matching, syntax highlighting, lint diagnostics, API autocomplete, find/replace, format, undo/redo, and a read-only generated type-definition view. Save is `Ctrl+S`; Test is `Ctrl+Enter`.

### 14.3 Triggers

Each rule selects exactly one trigger:

| Trigger | Invocation |
|---|---|
| `jspace.frame` | After a J-lens frame is available |
| `generation.beforeToken` | Immediately before a generation step |
| `generation.afterToken` | After a generated token and its readout are available |
| `intervention.beforeApply` | Before a manual stack is submitted |
| `sweep.afterComplete` | After a sweep completes; only log/tag actions permitted |

Rules cannot create arbitrary triggers, timers, background loops, or callbacks.

### 14.4 Rule module shape

Each rule is one source file with one global entry function:

```js
function run(ctx) {
  if (ctx.jspace.has("injection", { minScore: 0.70 })) {
    return [
      jspace.replace("injection", "trusted", {
        strength: 0.80,
        layers: { from: 18, to: 26 },
        duration: "next-token"
      }),
      rule.log("info", "Replaced injection signal")
    ];
  }
  return [];
}
```

The entry point must be exactly `function run(ctx)`. Top-level executable statements, dynamic imports, static imports, `eval`, `Function`, async functions, generators, and promises are rejected during validation. This avoids a module loader entirely.

### 14.5 Immutable context API

The only global input is frozen `ctx`:

```js
ctx.event.type                 // trigger string
ctx.event.sequence             // integer within run
ctx.model.id                   // display-safe model identifier
ctx.model.revision             // revision string
ctx.model.layerCount           // integer
ctx.lens.id                    // lens identifier
ctx.layer.index                // current layer or null
ctx.token.index                // current token index or null
ctx.token.text                 // current display token or null
ctx.generation.step            // generation step
ctx.generation.outputText      // bounded visible output prefix
ctx.jspace.has(term, options)  // boolean
ctx.jspace.score(term)         // number or null
ctx.jspace.top(limit)          // frozen array, limit 1..100
ctx.jspace.find(pattern)       // frozen bounded array
ctx.stack.active()             // frozen summaries of armed entries
ctx.tags.get(name)             // prior run tag value or null
```

`find` accepts a string or a validated regular-expression descriptor supplied by J Studio. It does not execute a user-created JavaScript `RegExp` against unbounded data. All strings and arrays are length-limited before entering the worker.

### 14.6 Action constructors

The sandbox exposes pure constructors that return plain frozen objects:

```js
jspace.inject(term, {
  strength,
  layers,       // "current", "all", or {from, to}
  duration,     // "current-token", "next-token", "generation", or {steps}
  label         // optional short display label
})

jspace.replace(matchTerm, replacementTerm, {
  strength,
  layers,
  duration,
  matchMode     // "exact" or "case-insensitive"
})

jspace.suppress(term, {
  strength,
  layers,
  duration
})

generation.stop(reason)
rule.log(level, message)       // level: "debug", "info", "warn", "error"
rule.tag(name, value)          // JSON scalar; visible in traces and filters
```

Constructors do not apply actions. The host validates returned JSON after the worker exits. Unknown keys, non-finite numbers, out-of-range layers/strength, oversized text, invalid duration, and unsupported capabilities reject the individual action. A rule may return at most 32 actions.

### 14.7 Conflict ordering

Rules run by ascending numeric priority, then stable rule ID. Returned actions are combined with the manual stack in this order:

1. Stop.
2. Replace.
3. Suppress.
4. Inject.
5. Tag and Log.

Two replacements matching the same exact term and overlapping scope are a hard conflict unless they have distinct priorities. The higher-priority rule wins and the lower action is marked Skipped. Suppress after Replace targets the replacement term only if it explicitly names that term. The Returned Actions panel shows this resolution before an edited rule can be enabled.

### 14.8 Test Bench

The Test tab lets users select a captured immutable snapshot, edit safe scalar fields, run the rule, and inspect:

- Returned raw JSON.
- Validated actions.
- Rejected actions and reasons.
- Conflicts against the active stack.
- Execution time, peak worker memory, input bytes, output bytes, and log bytes.

Testing never applies actions to a live session. A rule must pass its most recent test after every source or configuration edit before Enable becomes available.

## 15. Rules Sandbox Boundary

### 15.1 Python-only installation

The project depends on the pip-installable `quickjs` Python package. J Studio does not require Node.js, npm, Deno, Bun, a browser, or a separately installed JavaScript executable. The binding embeds QuickJS in the Python environment.

QuickJS is an implementation detail behind `RuleSandboxProtocol`; the UI does not import it directly. If the package is absent or incompatible, Rules remain editable and exportable, but Test and Enable are disabled with a repair message.

### 15.2 Process isolation

Every evaluation runs in a newly spawned Python worker process, never the Qt process and never a reusable shared rule process. Communication is one length-prefixed JSON request and one length-prefixed JSON response through a unidirectional pipe.

The worker receives only:

- Validated source text.
- Frozen JSON context.
- Trigger name.
- Numeric resource limits.

The worker environment excludes project objects, credentials, model handles, Qt objects, open files, inherited network clients, and user environment values. The parent terminates the worker on timeout and discards it after every evaluation.

### 15.3 Mandatory limits

| Resource | Default hard limit |
|---|---:|
| Wall time | 50 ms |
| QuickJS execution time | 25 ms |
| JavaScript heap | 16 MiB |
| Worker address space | 256 MiB where supported |
| Stack | 512 KiB |
| Source | 128 KiB |
| Input JSON | 512 KiB |
| Output JSON | 256 KiB |
| Actions | 32 |
| Log entries | 100 |
| Total log text | 8 KiB |
| `top()` or `find()` results | 100 |

Limits are not user-increasable from a rule. Administrators may lower them through application policy. A timeout, crash, malformed response, or limit breach fails closed: no actions from that evaluation are applied.

### 15.4 Forbidden capabilities

The JavaScript global object contains only frozen standard value types needed for deterministic expressions plus `ctx`, `jspace`, and `rule`. It contains no file, network, process, package, module, Python, FFI, WebAssembly, timer, clock, randomness, locale, clipboard, UI, or dynamic-code capability.

The host must not use Node's `vm` module as a security boundary. The UI documentation links to the sandbox rationale and shows the active limits in Rules > API > Security.

### 15.5 Failure policy

- First failure: rule action is skipped; warning appears in trace and Rules list.
- Three consecutive failures: rule is automatically disabled for the session.
- Worker crash or protocol violation: all rule execution pauses and an Activity error requests sandbox repair.
- Imported rules start Disabled and Untrusted, require source review, successful Test Bench execution, and explicit Enable.
- Project signatures may establish source provenance but never bypass sandboxing or validation.

## 16. Experiments

Experiments organize repeatable baseline/intervention comparisons. An experiment contains a session reference, prompt set, lens settings, intervention stack, enabled rules, generation settings, and run matrix.

The workspace has Setup, Runs, Compare, and Report tabs. The run table columns are Run, Prompt, Mode, Seed Alias, Stack, Rules, Status, Duration, Output, and Tags. Compare presents aligned outputs, activation deltas, concept timelines, and rule actions. Reports include exact provenance and never collapse repeated trials into one result without showing aggregation method.

## 17. Project Model

A J Studio project contains UI-owned references and documents:

- Session descriptor without secrets.
- Prompt library.
- Sweep configurations and imported readouts.
- Intervention stacks.
- JavaScript rules and rule metadata.
- Experiment definitions.
- Trace references.
- Layout and display preferences scoped to the project.

Saving never serializes live model objects, Python callables, worker processes, or credentials. Imported projects open with all interventions disarmed and all imported rules disabled until reviewed.

## 18. Settings

Settings is a modeless 940 x 720 window with search, left category navigation, scrolling content, Restore Page Defaults, Cancel, and Apply.

Categories:

1. General
2. Appearance
3. Sessions
4. J-Lens
5. Interventions
6. Rules
7. Generation
8. Storage
9. Shortcuts
10. Languages
11. Advanced

Rules settings show the fixed sandbox limits, worker status, QuickJS package version, auto-disable threshold, log retention, and “Run Sandbox Self-Test.” Security limits are read-only unless reduced by policy. There is no “Allow filesystem” or equivalent escape hatch.

Intervention settings include default duration, default strength, confirmation policy, auto-preview, and whether generation defaults to Baseline or With Armed Stack. It must default to Baseline.

## 19. Errors, Activity, and Confirmation

Long operations appear in Activity with title, session/run context, state, progress, elapsed time, detail, and Cancel when supported.

Error hierarchy:

1. Field validation beside the field.
2. Recoverable page failure in an inline banner.
3. Background failure in Activity plus a toast.
4. Modal error only when the user cannot safely continue.

Intervention confirmation names the operation, terms, strength, layer range, duration, active rules, and run mode. Buttons use verbs: Run Baseline, Run With Stack, Disarm, Cancel. The safest action receives default focus.

Raw exceptions, worker diagnostics, and service traces appear only under expandable Details with Copy. Secrets are redacted before reaching the UI.

## 20. Canonical Interaction Paths

### 20.1 Chat normally, then inspect

1. Open Chat.
2. Enter a prompt and select Send.
3. Read the streamed assistant response or select Stop.
4. Continue the conversation normally across multiple turns.
5. Open an assistant message's menu and choose Inspect with J-Lens.
6. J-Lens opens at the corresponding run and last generated position.
7. Select a matrix cell or term and open the Intervention Editor.
8. Add the validated intervention to the shared Intervention List.
9. Return to Chat and Regenerate or Continue with the checked intervention active.

### 20.2 Prompt and control a live generation

1. Open Live.
2. Enter a prompt and select Generate.
3. Read the streaming response while the eight strongest J-space concepts update at right.
4. Select Pause to inspect a stable frame or Next Token to advance deliberately.
5. Click a concept and choose Inject, Replace, or Suppress.
6. Confirm the compact action; it enters Active Controls as Queued.
7. Resume or select Next Token.
8. Verify the control changes to Applied only after acknowledgement.
9. Stop or allow generation to complete.
10. Optionally compare against a baseline or open the detailed trace.

### 20.3 Inspect a prompt in depth

1. Select Model Session.
2. Verify compatible J-lens in the centered model label.
3. Enter prompt in J-Lens workspace.
4. Select token position, layers, and concept threshold.
5. Run Sweep.
6. Inspect timeline and activation table.
7. Open a concept in Layer Explorer or pin it.
8. Export readout or save project.

### 20.4 Inject a J-space term

1. Select an activation or choose Intervention > New Injection.
2. Enter target term, strength, layers, and duration.
3. Preview.
4. Add to Intervention Stack.
5. Resolve conflicts.
6. Arm Stack.
7. Run With Stack.
8. Review applied event and resulting readout in Generation Trace.
9. Compare with baseline.

### 20.5 Replace a J-space term

1. Select source activation.
2. Choose Replace Term.
3. Confirm source term and enter replacement.
4. Set match mode, strength, layer range, and duration.
5. Preview expected scope.
6. Add, arm, and run.
7. Trace source/replacement scores across layers and tokens.

### 20.6 Create a dynamic JavaScript rule

1. Open Rules and choose New Rule.
2. Name it, select trigger, and set priority.
3. Write `function run(ctx)` using autocomplete.
4. Select a captured snapshot in Test Bench.
5. Run Test and inspect validated actions/conflicts/resources.
6. Save.
7. Enable becomes available only after a successful current test.
8. Explicitly Enable.
9. Run generation; rule events appear in Generation Trace.
10. On repeated failures, inspect diagnostics and re-enable only after editing and retesting.

### 20.7 Compare baseline and intervention

1. Run Baseline.
2. Arm selected stack and rules.
3. Run With Stack using the same experiment configuration.
4. Open Compare.
5. Review output, concept, layer, token, and action differences.
6. Export report with provenance.

## 21. Keyboard and Focus

| Shortcut | Command |
|---|---|
| `Ctrl+K` | Select Model Session |
| `Ctrl+O` | Open Project |
| `Ctrl+S` | Save |
| `Ctrl+Shift+S` | Save As |
| `Ctrl+N` | New Project |
| `Ctrl+F` | Context search/filter |
| `Ctrl+1` | Live |
| `Ctrl+2` | Rules |
| `Ctrl+3` | History |
| `Ctrl+Shift+P` | Advanced Tools |
| `Space` | Pause/Resume while focus is outside an editor |
| `F10` | Generate Next Token |
| `Ctrl+Enter` | Test current rule or run current configured action |
| `F5` | Refresh current model/list |
| `Ctrl+,` | Settings |
| `Esc` | Cancel transient operation or dismiss non-destructive dialog |

Dialog focus starts at the first meaningful field and returns to its invoker when closed. Removing a focused row moves focus to the next row, then previous if needed. Every icon-only command has tooltip, shortcut, and accessible name.

## 22. Accessibility and Localization

- Minimum contrast is 4.5:1 for normal text and 3:1 for large text/UI boundaries.
- Focus uses a 2 px high-contrast outline.
- Tables expose headers, coordinates, sort, checked, expanded, source, and operation states.
- Sweep completion, rule disablement, sandbox failure, and intervention arming are announced through accessibility events.
- Progress announcements are throttled.
- Reduced motion disables nonessential transitions.
- Layouts support 160% text and translated strings 40% longer than English.
- Strings are externalized through Qt translation catalogs.
- Tokens and identifiers remain directionally isolated in right-to-left locales.

## 23. Performance Requirements

- First paint within 500 ms after UI process initialization, excluding session loading.
- Local filter feedback within 100 ms.
- No widget-per-row implementation for activations, traces, rules, or experiment runs.
- Timeline and table scrolling target at least 50 frames per second on reference hardware.
- UI progress updates are capped at 10 per second.
- No inference, lens computation, or JavaScript execution occurs on the Qt GUI thread.
- Opening a menu performs no synchronous model query.

## 24. UI Testing

Use `pytest-qt` with fake session, lens, generation, and sandbox services.

Required paths:

1. First launch > type prompt > Generate > response and concepts stream.
2. Running > Pause > Next Token > Resume > Stop.
3. Running > select concept > Replace > queued > applied acknowledgement > Undo.
4. Live > Advanced Tools > Layer Explorer > return to Live without losing run.
5. Missing lens > compatibility error > load compatible lens > run.
6. Sweep > cancel > partial results retained and labeled.
7. Replace term > conflicting replacement > resolution shown > higher priority wins.
8. Offline trace > inspection enabled > intervention application disabled.
9. New rule > current test required > enable > action appears in trace.
10. Rule timeout > no action applied > failure displayed.
11. Three consecutive rule failures > automatic disablement.
12. Imported project > interventions disarmed > rules disabled.
13. Close multiple dirty documents > consolidated save dialog.
14. Keyboard-only completion of prompt, pause, next-token, replace, and rule-test paths.

Sandbox protocol tests must cover malformed JSON, oversized input/output, infinite loop, recursion, memory exhaustion, forbidden globals, dynamic import, `eval`, too many actions, non-finite numbers, invalid layers, invalid strength, worker crash, timeout, and parent cancellation. Every case fails closed.

Visual regression captures the native light palette and OS dark palette at 100%, 125%, and 160% scaling for the main Cheat Engine-shaped shell, Session Picker, read states, found list, Intervention List, Model View, Layer Explorer, Rules, Test Bench, Generation Trace, errors, and confirmations.

## 25. Acceptance Criteria

The UI design is implemented when:

1. A first-time user can enter a prompt, start generation, pause it, advance one token, and stop it without opening settings or Advanced Tools.
2. A user can click a live concept and inject, replace, or suppress it without leaving Live.
3. The main window has no dashboard navigation; expert tools open through menus, buttons, context menus, and secondary windows.
4. Injection, replacement, and suppression are visually and semantically distinct.
5. Every readout and intervention result shows complete provenance in Details, without forcing it into the default screen.
6. Baseline data is immutable and cannot be silently overwritten by an intervention run.
7. Rules use only the documented immutable context and declarative constructors.
8. Rule JavaScript runs outside the Qt process with mandatory resource limits and fails closed.
9. Installing J Studio requires no Node.js, npm, or separate JavaScript executable.
10. Imported rules are disabled and untrusted until reviewed and tested.
11. Offline traces cannot expose live intervention actions.
12. Large activation and trace datasets remain virtualized and responsive.
13. Light, dark, high-DPI, translated, keyboard-only, and reduced-motion configurations pass UI tests.
14. No UI module directly performs lens computation, inference, hidden-state access, or model intervention.

## 26. Research and Technical References

The terminology and product guardrails are grounded in:

- [Anthropic, “A global workspace in language models”](https://www.anthropic.com/research/global-workspace): J-space overview and distinction from textual chain-of-thought.
- [Transformer Circuits, “Verbalizable Representations Form a Global Workspace in Language Models”](https://transformer-circuits.pub/2026/workspace/index.html): Jacobian lens, J-space observations, and intervention experiments.
- [Neuronpedia J-lens demo](https://www.neuronpedia.org/qwen3.6-27b/jlens): a concrete readout interaction reference.
- [QuickJS documentation](https://bellard.org/quickjs/quickjs.pdf): embedded runtime memory limits and interrupt handling.
- [Python `quickjs` package](https://pypi.org/project/quickjs/): pip-installable binding to embedded QuickJS.
- [Node.js `vm` documentation](https://nodejs.org/api/vm.html): explicit warning that `node:vm` is not a security mechanism, supporting the decision not to use it.

Research links belong in Help > Research References and the About dialog. J Studio must not imply affiliation with Anthropic, Neuronpedia, or the authors unless separately authorized.
