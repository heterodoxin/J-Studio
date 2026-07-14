# Generation Intervention Stress Benchmark Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure whether the real Qwen Stable-lens path produces coherent, causally changed responses across diverse injection, replacement, suppression, and multi-token cases.

**Architecture:** Load the model and lens once, run deterministic baseline and intervention completions through `HFModelRuntime`, retain every trace and response, and score observable invariants separately from engine acceptance. Save JSON and standalone HTML artifacts; change production code only after a repeatable failed case identifies a root cause.

**Tech Stack:** Python 3.11, PyTorch/ROCm, Transformers, J Studio runtime, Jacobian Lens.

## Global Constraints

- Use `heterodoxin/qwen3-8b-apostate` and its current Stable lens.
- Execute residual/J-space operators only: no logit, prompt, or output fallback.
- Strength is a maximum budget; record the minimum selected strength.
- Preserve exact prompts, outputs, carrier, delay, layers, warnings, and search points.
- Restart J Studio after the benchmark releases VRAM.

---

### Task 1: Run the generation matrix

**Files:**
- Generate: `jacobian-lens/reports/generation-intervention-stress/qwen3_8b_apostate/results.json`
- Generate: `jacobian-lens/reports/generation-intervention-stress/qwen3_8b_apostate/index.html`

**Interfaces:**
- Consumes: `HFModelRuntime.prepare_interventions()` and `HFModelRuntime.stream()`.
- Produces: per-case baseline, edited output, intervention trace, and invariant scores.

- [x] Run conversational, factual, instruction, reasoning, sensitive-word, and multi-token injection cases.
- [x] Run replacement and suppression cases where the source appears in the baseline.
- [x] Check target count, non-leading placement, relation realization, baseline ordered overlap, corruption, repetition, selected scale, and hook cleanup.
- [x] Save complete JSON and a standalone HTML summary with all outputs visible.

### Task 2: Diagnose reproducible failures

**Files:**
- Modify only after evidence: `jstudio/services/hf_runtime.py`
- Test only after evidence: `tests/test_hf_runtime.py`

**Interfaces:**
- Consumes: failed stress cases and their full traces.
- Produces: root-cause statement and, if needed, a red-green regression fix.

- [x] Group failures by observable mode rather than target word.
- [x] Re-run representative failures to distinguish deterministic defects from marginal thresholds.
- [x] If production behavior changes, first add a focused failing test and verify the expected RED result.
- [x] Implement one root-cause fix, then verify focused and full suites.

### Task 3: Verify and relaunch

**Files:**
- Update: this plan's checkboxes.

**Interfaces:**
- Consumes: benchmark artifacts and any regression change.
- Produces: pushed commit and a running J Studio process.

- [x] Run all app and J-lens tests plus both Ruff suites.
- [x] Review the final diff and benchmark failures.
- [ ] Commit and push any durable benchmark or behavior changes.
- [x] Leave J Studio stopped after verification to avoid sustained GPU allocation;
      launch is deferred until explicitly requested.
