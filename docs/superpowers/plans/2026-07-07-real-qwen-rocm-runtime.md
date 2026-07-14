# Real Qwen ROCm Runtime Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace J Studio's default fake services with a real Python 3.11 ROCm Qwen runtime while keeping demo mode explicit.

**Architecture:** Add one backend-only service module that lazily imports PyTorch,
Transformers, and `jlens`, implements the existing service protocols, and emits the
same immutable records as fake services. The launcher selects real services unless
`--demo` is provided.

**Tech Stack:** Python 3.11, ROCm 7.1 PyTorch, Transformers, Qwen3 8B BF16, PySide6

## Global Constraints

- System Python 3.11; no project virtual environment.
- Default model: `heterodoxin/qwen3-8b-apostate`, local cache first.
- Fake concepts are reachable only with `--demo`.
- Uncalibrated residual readouts are labeled vanilla and cannot apply interventions.
- No PyTorch, Transformers, or `jlens` import is allowed under `jstudio/ui`.

---

### Task 1: Real service contract implementation

**Files:**
- Create: `jstudio/services/hf_runtime.py`
- Test: `tests/test_hf_runtime.py`

**Interfaces:**
- Produces: `create_hf_services(model_id: str, *, local_files_only: bool = True) -> JStudioServices`
- Produces: `HFModelRuntime`, `HFGenerationService`, `HFLensService`, `HFSessionService`
- Consumes: the existing `GenerationRequest` and `GenerationEventSink` protocols.

- [ ] Write a failing test using an injected runtime double. Assert the session identifies
  the real model, generation emits the runtime's output, frames use runtime concept data,
  and intervention preview rejects an uncalibrated session.
- [ ] Run `python3.11 -m pytest tests/test_hf_runtime.py -q` and confirm the module is missing.
- [ ] Implement one-model session, bounded executor generation controls, frame storage,
  real residual readout conversion, and fail-closed intervention preview.
- [ ] Run the focused test and commit `feat: add real Hugging Face model services`.

### Task 2: Real-by-default launcher

**Files:**
- Modify: `jstudio/ui/app.py`
- Modify: `tests/ui/test_end_to_end.py`
- Test: `tests/test_app_arguments.py`

**Interfaces:**
- `--demo` selects `create_fake_services()`.
- `--model MODEL_ID` selects the local decoder checkpoint.
- Normal launch calls `create_hf_services()` before window construction.

- [ ] Write failing parser and service-selection tests proving normal launch is real and
  `--demo` is fake.
- [ ] Run the focused tests and confirm failure because the flags/selector do not exist.
- [ ] Add the flags and a pure `select_services(args)` function; preserve explicit service
  injection in `create_application()` for UI tests.
- [ ] Run focused tests and commit `feat: launch J Studio with real Qwen services`.

### Task 3: ROCm smoke test and launch

**Files:**
- Modify: `README.md`
- Modify: `jstudio/ui/shell/main_window.py`
- Modify: `tests/ui/test_visual_smoke.py`

**Interfaces:**
- The main window has a minimum size of 734 by 592 logical pixels.
- Direct launch command is `PYTHONPATH=app:jacobian-lens python3.11 -m jstudio`.

- [ ] Add the failing minimum-geometry regression test for an attempted 5112 by 366 resize.
- [ ] Enforce the compact minimum size and run UI tests.
- [ ] Load the cached Qwen checkpoint on ROCm and run `make me an ascii cat`; verify output
  and found terms do not equal the fake fixture.
- [ ] Run Ruff and both complete test suites, then commit documentation and verification.
