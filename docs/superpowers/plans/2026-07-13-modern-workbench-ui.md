# Modern J Studio Workbench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Deliver a clearer, less cluttered, modern J Studio interface while preserving its Cheat Engine scan/results/intervention workflow and fully theming the interactive J-Lens page.

**Architecture:** Centralize semantic colors and Qt QSS in `jstudio/ui/theme.py`, then assign stable widget roles from the existing workspaces rather than scattering inline styles. Preserve service and signal boundaries. Theme the repository-owned J-Lens document at its source in `jlens/data/slice_vis.html`, without changing its data or JavaScript contracts.

**Tech Stack:** Python 3.11, PySide6/Qt 6, Qt Style Sheets, repository-owned HTML/CSS/JavaScript, pytest-qt, Ruff.

## Global Constraints

- Preserve Main, Chat, J-Lens, and Rules as permanent shared-session tabs.
- Preserve every existing command, signal, accessible name, project field, and J-Lens data value.
- Default to a polished graphite/violet theme while retaining callable light and system palette helpers.
- Do not add a navigation rail, card dashboard, web framework, or new runtime dependency.
- Do not push to the current `cheat-engine/cheat-engine` remote.

---

### Task 1: Shared Semantic Qt Theme

**Files:**
- Modify: `jstudio/ui/theme.py`
- Modify: `jstudio/ui/app.py`
- Test: `tests/ui/test_visual_smoke.py`

**Interfaces:**
- Produces: `apply_jstudio_theme(application, mode: str = "dark") -> None`
- Produces: semantic selectors for `#sessionBar`, `#workspaceTabs`, `.panel`, `.primary`, `.danger`, `.statusPill`, and data/editor surfaces.

- [ ] **Step 1: Write failing tests** asserting the application installs a non-empty stylesheet containing `#workspaceTabs`, `QWidget[role="panel"]`, and the violet accent, and that the window exposes the expected semantic object names.
- [ ] **Step 2: Run** `python3.11 -m pytest tests/ui/test_visual_smoke.py -q` and confirm the new assertions fail.
- [ ] **Step 3: Implement** `apply_jstudio_theme`, expanding the palette roles and a single QSS string. Call it during `create_application` before window construction so size hints are stable.
- [ ] **Step 4: Run** the focused test and `python3.11 -m ruff check jstudio/ui/theme.py jstudio/ui/app.py tests/ui/test_visual_smoke.py`.
- [ ] **Step 5: Commit** with `git commit -m "feat: add modern semantic J Studio theme"`.

### Task 2: Modern Shell and Progressive Main Workspace

**Files:**
- Modify: `jstudio/ui/shell/main_window.py`
- Modify: `jstudio/ui/shell/session_bar.py`
- Modify: `jstudio/ui/main_workspace.py`
- Modify: `jstudio/ui/interventions/stack.py`
- Test: `tests/ui/test_main_workflow.py`
- Test: `tests/ui/test_visual_smoke.py`

**Interfaces:**
- Produces: `MainReadWorkspace.advanced_scan` as a checkable disclosure button.
- Preserves: existing `prompt`, read buttons, scan option widgets, concept table, stack buttons, and workspace signals.

- [ ] **Step 1: Write failing tests** proving advanced scan controls are hidden initially, disclosure reveals them, First Read is the primary action, session state uses pills, and every previous button remains reachable.
- [ ] **Step 2: Run** the two focused UI test modules and confirm failures identify missing disclosure/roles.
- [ ] **Step 3: Refactor layout only:** wrap the found and controls regions in semantic panels, make the prompt dominant, combine read/concept selectors, place scan options in a hidden container, remove the redundant permanent bottom strip, and retain `advanced_button` as the compact disclosure/tool entry for compatibility.
- [ ] **Step 4: Restyle the session bar hierarchy and assign semantic properties to intervention actions without changing their connections.
- [ ] **Step 5: Run** focused tests and Ruff, then capture Main with `python3.11 -m jstudio --demo --screenshot /tmp/jstudio-main.png --quit-after 700`.
- [ ] **Step 6: Inspect** `/tmp/jstudio-main.png` for clipping, excess borders, low contrast, or duplicated actions; correct any issue and recapture.
- [ ] **Step 7: Commit** with `git commit -m "feat: declutter the main research workflow"`.

### Task 3: Cohesive Chat and Rules Workspaces

**Files:**
- Modify: `jstudio/ui/chat.py`
- Modify: `jstudio/ui/rules/workspace.py`
- Modify: `jstudio/ui/rules/editor.py`
- Test: `tests/ui/test_chat.py`
- Test: `tests/ui/test_visual_smoke.py`

**Interfaces:**
- Preserves: chat model roles and all generation/rule signals.
- Produces: semantic widget roles for transcript, composer, rule list, source editor, inspector, and output drawer.

- [ ] **Step 1: Write failing tests** for semantic object names/roles and unchanged action availability.
- [ ] **Step 2: Run** focused tests and confirm they fail before implementation.
- [ ] **Step 3: Apply** the shared panel/editor hierarchy, improve layout margins and toolbar grouping, make the source editor/transcript dominant, and reduce the Rules output drawer height without removing tabs.
- [ ] **Step 4: Run** focused tests and Ruff.
- [ ] **Step 5: Capture** Chat and Rules screenshots through the existing Qt test helpers or a short offscreen application invocation, inspect them, and correct visual defects.
- [ ] **Step 6: Commit** with `git commit -m "feat: modernize chat and rules workspaces"`.

### Task 4: Themed Native and HTML J-Lens Surface

**Files:**
- Modify: `jstudio/ui/lensview/workspace.py`
- Modify: `jstudio/ui/lensview/web_view.py`
- Modify: `/var/home/Heterodoxin/Desktop/J-Studio/jacobian-lens/jlens/data/slice_vis.html`
- Test: `tests/ui/test_jlens_workspace.py`
- Test: `/var/home/Heterodoxin/Desktop/J-Studio/jacobian-lens/tests/test_vis.py`

**Interfaces:**
- Preserves: `JLensBridge` slots/signals and `SlicePage.html` loading.
- Produces: CSS variables `--js-bg`, `--js-panel`, `--js-data`, `--js-text`, `--js-muted`, `--js-accent`, and `--js-grid` in the generated document.

- [ ] **Step 1: Write failing tests** asserting native header semantic roles and the required CSS variables in a rendered J-Lens page.
- [ ] **Step 2: Run** both focused suites and confirm the new assertions fail.
- [ ] **Step 3: Recompose** the native J-Lens header into title/status and action clusters; give fit progress and web frame semantic roles.
- [ ] **Step 4: Replace** hard-coded white/gray/pink document colors with theme variables while preserving DOM IDs, D3 selectors, bridge calls, plots, hover, pins, and grid geometry.
- [ ] **Step 5: Run** both focused test suites plus Ruff.
- [ ] **Step 6: Generate** a populated slice screenshot from deterministic test data, inspect grid readability, plot contrast, scrollbar appearance, and selected-cell clarity; refine and recapture.
- [ ] **Step 7: Commit backend HTML** with `git commit -m "feat: theme the interactive J-Lens surface"` in `jacobian-lens`, then commit native shell changes with `git commit -m "feat: modernize the J-Lens workspace"` in `app`.

### Task 5: Release Verification and Push Preparation

**Files:**
- Modify: `README.md`
- Create if absent: `CHANGELOG.md`
- Verify: both Git repositories

**Interfaces:**
- Produces: reproducible run/test/package commands and clean feature-branch commits.

- [ ] **Step 1: Update documentation** with the modern workbench behavior, automatic causal phrase strengths, duration semantics, screenshots command, and correct Python 3.11 launch command.
- [ ] **Step 2: Run app verification:** `python3.11 -m pytest -q`, `python3.11 -m ruff check jstudio tests`, and `python3.11 -m build` after installing `build` only if already available.
- [ ] **Step 3: Run backend verification:** `python3.11 -m pytest -q` and `python3.11 -m ruff check jlens tests`.
- [ ] **Step 4: Run a launch smoke** with `python3.11 -m jstudio --demo --quit-after 1000`, then launch the real cached Qwen model and confirm the process remains alive.
- [ ] **Step 5: Audit Git** with `git diff --check`, `git status --short`, `git log --oneline`, and `git remote -v`; preserve unrelated user changes.
- [ ] **Step 6: Commit release documentation** with `git commit -m "docs: prepare J Studio release workflow"`.
- [ ] **Step 7: Do not push** until a dedicated J Studio remote replaces the unrelated Cheat Engine origin. Report the exact safe push command and remaining remote blocker.
