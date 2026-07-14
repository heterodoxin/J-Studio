# Baseline-Preserving J-Space Injection Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Keep J-space interventions causally effective without allowing them to replace or derail the model's baseline response.

**Architecture:** Extend the pure generated-token effect judge with sequence removal and longest-common-subsequence coverage. Apply operation-specific preservation references before a dose can pass minimum-effective-strength search; failed searches remain unapplied.

**Tech Stack:** Python 3.11, PyTorch/Transformers runtime, pytest, Ruff, Jacobian Lens residual hooks.

## Global Constraints

- Judge intervention effects from generated token IDs only; do not use logits as an intervention objective or fallback.
- Never rewrite the prompt or silently substitute another steering mechanism.
- Select the minimum tested J-space strength satisfying causal effect and baseline preservation.
- Preserve compatibility with multi-token source and target variants.

---

### Task 1: Generated-token trajectory preservation

**Files:**
- Modify: `tests/test_hf_runtime.py`
- Modify: `jstudio/services/hf_runtime.py`
- Modify: `CHANGELOG.md`

**Interfaces:**
- Consumes: `_causal_token_effect(operation, baseline, candidate, source_variants, target_variants)`.
- Produces: deterministic operation-specific acceptance with no signature change.

- [ ] **Step 1: Write failing regression tests**

Add cases proving an injected target that replaces the whole response, repeats,
or is merely concatenated onto the baseline fails, while one contextual insertion
with ordered baseline anchors before and after the target passes. Add equivalent
replacement and suppression preservation cases.

- [ ] **Step 2: Run the focused tests and verify RED**

Run: `python3.11 -m pytest tests/test_hf_runtime.py -k 'causal_token_effect or causal_injection' -q`

Expected: the new derailment and concatenation assertions fail against the current
target-presence-only judge.

- [ ] **Step 3: Implement the minimal preservation judge**

Add pure helpers that remove non-overlapping phrase variants and calculate ordered
overlap using a two-row longest-common-subsequence dynamic program. Require at
least 50% baseline coverage after operation-specific phrase removal, reject empty
preservation references, and reject exact edge concatenation for injection.

- [ ] **Step 4: Run focused and complete automated verification**

Run: `python3.11 -m pytest tests/test_hf_runtime.py -q`

Expected: all runtime tests pass.

Run: `python3.11 -m pytest tests -q && python3.11 -m ruff check jstudio tests`

Expected: the full J Studio suite and Ruff pass.

- [ ] **Step 5: Benchmark the real model and lens**

Run a deterministic `hi` baseline plus injection, replacement, and suppression
probes on `heterodoxin/qwen3-8b-apostate` with the fitted stable lens. Record the
selected scale, pass/fail state, and full generated response. Confirm no accepted
response is target-only, target-repeating, or a semantically unrelated definition.

- [ ] **Step 6: Verify the lens package and release state**

Run: `python3.11 -m pytest jacobian-lens/tests -q && python3.11 -m ruff check jacobian-lens/jlens jacobian-lens/tests`

Expected: all Jacobian Lens tests and Ruff pass.

Run: `git diff --check && git status --short`

Expected: no whitespace errors; only intended files are modified.

- [ ] **Step 7: Commit, push, and restart J Studio**

Commit the tests, runtime implementation, specification, plan, and changelog; push
`main`; then start J Studio with Python 3.11, the Qwen model, and its fitted lens.

### Task 2: Route default edits through the workspace

**Files:**
- Modify: `jacobian-lens/jlens/interventions.py`
- Modify: `jacobian-lens/tests/test_interventions.py`
- Modify: `jstudio/services/hf_runtime.py`
- Modify: `tests/test_hf_runtime.py`

**Interfaces:**
- Consumes: phrase intervention search and `InterventionEngine.apply()`.
- Produces: `application_positions` for group-wise causal probes and an `ordered`
  transform mode for token-sequence versus concept-centroid application.

- [ ] **Step 1: Add failing engine tests**

Prove one phrase candidate can be probed and recorded across a group of positions,
and prove an unordered multi-token schedule holds one centroid concept rather than
cycling target-token directions.

- [ ] **Step 2: Add failing runtime routing tests**

Prove the exact final occurrence of the current user-turn token sequence is selected,
missing spans raise an error, default injection passes that span to the engine, the
eligible fitted layers are limited to 38–75% depth, and application occurs once in
unordered concept mode. Prove replace/suppress retain localized response-boundary
application.

- [ ] **Step 3: Implement grouped positions and centroid application**

Add the optional `application_positions` phrase-search parameter, store it in the
trace, and pass it to causal probes. Add `ordered=False` support to phrase schedules
and `InterventionEngine.apply()` by using the normalized mean target direction.

- [ ] **Step 4: Implement exact user-turn workspace routing**

Locate the raw prompt token subsequence inside the fully formatted prompt, reject a
missing match, intersect fitted/user layers with the relative workspace band, and
use the user span for default Next Token injection. Keep replace/suppress and
explicit Steps/Generation modes on their ordered response-boundary schedule.

- [ ] **Step 5: Verify and benchmark**

Run both full test suites and Ruff, then manually compare baseline and intervention
responses on Qwen for arbitrary greetings, a verbal-report prompt, replacement,
and suppression. Forced-prefix candidates must fail closed.
