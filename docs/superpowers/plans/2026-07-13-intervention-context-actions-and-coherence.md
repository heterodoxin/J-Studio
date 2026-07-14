# Intervention Context Actions and Coherent Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add complete right-click intervention management, make Trace Influence visible and seeded, and reject target-only injection completions.

**Architecture:** `InterventionStackView` owns menu presentation and emits semantic row commands; `MainReadWorkspace` owns project mutations. `JStudioMainWindow` routes concept trace requests into the existing tool window. Generated-token causal evidence gains a composition gate without adding a logit path.

**Tech Stack:** Python 3.11, PySide6, pytest/pytest-qt, existing J-space runtime.

## Global Constraints

- No logits or alternate steering fallback.
- Minimum-effective-strength search remains bounded by the configured maximum.
- Project mutations must preserve stable intervention IDs except for duplicates.
- Right-click actions must not add permanent inline controls.

---

### Task 1: Reject target-only injection probes

**Files:**
- Modify: `jstudio/services/hf_runtime.py`
- Test: `tests/test_hf_runtime.py`

**Interfaces:**
- Consumes: `_causal_token_effect(operation, baseline, candidate, source_variants, target_variants)`
- Produces: inject acceptance that requires target gain plus two non-target generated tokens.

- [ ] Write tests asserting `(banana,)` and repeated target tokens fail while `(banana, is, useful)` passes.
- [ ] Run `python3.11 -m pytest tests/test_hf_runtime.py -k causal_token_effect -q` and confirm the target-only case fails.
- [ ] Add a generated-token composition check to `_causal_token_effect` for inject only.
- [ ] Rerun the focused tests and commit the passing behavior.

### Task 2: Add intervention-row context actions

**Files:**
- Modify: `jstudio/ui/interventions/stack.py`
- Modify: `jstudio/ui/interventions/editor.py`
- Modify: `jstudio/ui/main_workspace.py`
- Test: `tests/ui/test_intervention_editor.py`
- Test: `tests/ui/test_main_workflow.py`

**Interfaces:**
- Produces: `InterventionStackView.action_requested: Signal(str, object)` with sorted row tuples.
- Produces: `InterventionEditor(..., draft: InterventionDraft | None = None)` prefill.
- Produces: `MainReadWorkspace._handle_stack_action(action: str, rows: tuple[int, ...])`.

- [ ] Write Qt tests for menu action labels, multi-selection enablement, editor prefill, duplicate, reorder, enable/disable, and remove.
- [ ] Run the focused Qt tests and confirm failures identify missing menu and prefill behavior.
- [ ] Implement the custom context menu and semantic action signal.
- [ ] Implement draft prefill and stable-ID project mutations in the workspace.
- [ ] Rerun the focused Qt tests and commit.

### Task 3: Route Trace Influence and release

**Files:**
- Modify: `jstudio/ui/main_workspace.py`
- Modify: `jstudio/ui/shell/main_window.py`
- Test: `tests/ui/test_main_workflow.py`
- Test: `tests/ui/test_secondary_tools.py`

**Interfaces:**
- Produces: `MainReadWorkspace.influence_requested: Signal(str)`.
- Produces: `JStudioMainWindow._open_influence_trace(term: str)`.

- [ ] Write a Qt test that invokes Trace Influence for a selected concept and expects a visible tool window with the matching seed.
- [ ] Run the focused test and confirm the current label-only action fails.
- [ ] Wire the context action to the signal and seed/refresh the trace window.
- [ ] Run focused tests, full app and lens suites, Ruff, offscreen launch, and a real Qwen banana-injection probe.
- [ ] Restart J Studio, commit the release, and push `main`.
