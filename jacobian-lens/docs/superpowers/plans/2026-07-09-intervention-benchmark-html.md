# Intervention Benchmark HTML Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a strict causal benchmark for J-space injection/replacement and emit a standalone HTML report showing how well interventions worked.

**Architecture:** Implement a backend-only `jlens.benchmark` module with serializable case/result dataclasses, exact next-token rank measurement, strict intervention execution through `InterventionEngine`, and an HTML renderer. Add a CLI wrapper in `scripts/benchmark_interventions.py` for local Hugging Face runs. Tests use the existing tiny decoder and never require a GPU.

**Tech Stack:** Python 3.11, PyTorch, existing `jlens.InterventionEngine`, existing `ActivationRecorder`, stdlib `argparse`, stdlib `html`, pytest, ruff.

## Global Constraints

- No logit fallback, no lexical fallback, no corrective residual fallback.
- The applied edit must come only from `InterventionEngine.apply(result)` when `result.success` is true.
- Benchmark readout uses a single forward pass over the supplied context, matching the original J-Lens notebook behavior.
- Generated-response inspection is represented by explicitly appending transcript text to the context before benchmarking; the benchmark itself must not call `generate`.
- HTML reports must be standalone and safe to open locally without external network dependencies.
- Live Hugging Face benchmarking is opt-in through the CLI; unit tests must stay CPU-only.

---

### Task 1: Benchmark Core Data and Measurement

**Files:**
- Create: `jlens/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `BenchmarkCase`, `BenchmarkResult`, `TokenRank`, `next_token_logits(model, prompt)`, `rank_token(logits, token_id)`, `run_case(model, lens, case)`.
- Consumes: `LensModel`, `JacobianLens`, `InterventionEngine`, `ActivationRecorder`.

- [ ] **Step 1: Write the failing test**

Add `tests/test_benchmark.py` with:

```python
import torch

from jlens.benchmark import BenchmarkCase, rank_token, run_case
from jlens.lens import JacobianLens
from tests.tiny import PROMPT, TinyDecoder


def test_rank_token_reports_zero_based_rank_and_logit():
    logits = torch.tensor([0.1, 3.0, -1.0, 2.0])

    rank = rank_token(logits, 3)

    assert rank.token_id == 3
    assert rank.rank == 1
    assert rank.logit == 2.0


def test_injection_case_records_strict_failed_noop():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    torch.manual_seed(10)
    lens = JacobianLens({2: torch.randn(8, 8) * 2}, n_prompts=1, d_model=8)
    case = BenchmarkCase(
        name="hard inject",
        operation="inject",
        prompt=PROMPT,
        target=20,
        layers=(2,),
        maximum_scale=1.0,
    )

    result = run_case(model, lens, case)

    assert not result.success
    assert result.applied is False
    assert result.target_after.rank == result.target_before.rank
    assert result.trace["message"] == "bounded search found no passing intervention"
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.11 -m pytest -q tests/test_benchmark.py`

Expected: import failure for `jlens.benchmark`.

- [ ] **Step 3: Write minimal implementation**

Create `jlens/benchmark.py` implementing:

```python
from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import torch

from jlens.hooks import ActivationRecorder
from jlens.interventions import InterventionEngine
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

Operation = Literal["inject", "replace", "suppress"]


@dataclass(frozen=True)
class TokenRank:
    token_id: int
    token: str
    rank: int
    logit: float


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    operation: Operation
    prompt: str
    target: str | int
    source: str | int | None = None
    layers: tuple[int, ...] = ()
    position: int = -1
    top_k: int = 1
    maximum_scale: float = 16.0
    context_mode: str = "prompt-only"


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    success: bool
    applied: bool
    target_before: TokenRank
    target_after: TokenRank
    source_before: TokenRank | None
    source_after: TokenRank | None
    top_before: tuple[TokenRank, ...]
    top_after: tuple[TokenRank, ...]
    selected_layer: int | None
    selected_scale: float
    normalized_cost: float
    trace: dict

    def to_dict(self) -> dict:
        return asdict(self)
```

Implement `next_token_logits`, `rank_token`, `_top`, `_resolve_single`, and `run_case` using `InterventionEngine`. `run_case` must use `nullcontext()` when the intervention fails, so failed cases are no-ops.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.11 -m pytest -q tests/test_benchmark.py`

Expected: `2 passed`.

- [ ] **Step 5: Commit**

Run:

```bash
git add jlens/benchmark.py tests/test_benchmark.py
git commit -m "feat: add intervention benchmark core"
```

### Task 2: Replacement Benchmark and HTML Renderer

**Files:**
- Modify: `jlens/benchmark.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: `render_html_report(results, title="J-Lens Intervention Benchmark") -> str`.
- Consumes: `BenchmarkResult.to_dict()`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark.py`:

```python
from jlens.benchmark import render_html_report


def test_replacement_case_records_source_and_target_ranks():
    model = TinyDecoder(n_layers=4, d_model=8, seed=1)
    torch.manual_seed(2)
    lens = JacobianLens({2: torch.randn(8, 8)}, n_prompts=1, d_model=8)
    baseline = run_case(
        model,
        lens,
        BenchmarkCase(
            name="replace",
            operation="replace",
            prompt=PROMPT,
            source=1,
            target=2,
            layers=(2,),
            maximum_scale=0.25,
        ),
    )

    assert baseline.source_before is not None
    assert baseline.source_after is not None
    assert baseline.target_before.token_id == 2
    assert baseline.source_before.token_id == 1


def test_html_report_contains_summary_and_case_rows():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    torch.manual_seed(10)
    lens = JacobianLens({2: torch.randn(8, 8) * 2}, n_prompts=1, d_model=8)
    result = run_case(
        model,
        lens,
        BenchmarkCase(
            name="html inject",
            operation="inject",
            prompt=PROMPT,
            target=4,
            layers=(2,),
            maximum_scale=32.0,
        ),
    )

    page = render_html_report((result,), title="Smoke Report")

    assert "<!doctype html>" in page.lower()
    assert "Smoke Report" in page
    assert "html inject" in page
    assert "Target rank" in page
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.11 -m pytest -q tests/test_benchmark.py`

Expected: import failure for `render_html_report`.

- [ ] **Step 3: Write minimal implementation**

Extend `jlens/benchmark.py` with `render_html_report`. It must include:

- pass-rate summary
- per-case row with name, operation, context mode, success, applied, target rank before/after, source rank before/after, selected layer, selected scale, normalized cost, message
- inline SVG bar for target rank improvement
- escaped prompt/details in `<details>`

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.11 -m pytest -q tests/test_benchmark.py`

Expected: all benchmark tests pass.

- [ ] **Step 5: Commit**

Run:

```bash
git add jlens/benchmark.py tests/test_benchmark.py
git commit -m "feat: render intervention benchmark html"
```

### Task 3: CLI Runner and Live Report Output

**Files:**
- Create: `scripts/benchmark_interventions.py`
- Modify: `jlens/__init__.py`
- Test: `tests/test_benchmark.py`

**Interfaces:**
- Produces: CLI flags `--model`, `--lens`, `--out`, `--local-files-only`, `--case-set`.
- Consumes: `run_case`, `render_html_report`.

- [ ] **Step 1: Write the failing test**

Append to `tests/test_benchmark.py`:

```python
def test_benchmark_exports_public_api():
    import jlens

    assert hasattr(jlens, "BenchmarkCase")
    assert hasattr(jlens, "run_benchmark")
    assert hasattr(jlens, "render_benchmark_html")
```

- [ ] **Step 2: Run test to verify it fails**

Run: `python3.11 -m pytest -q tests/test_benchmark.py::test_benchmark_exports_public_api`

Expected: assertion failure.

- [ ] **Step 3: Write minimal implementation**

Add public aliases in `jlens/__init__.py`:

```python
from jlens.benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    TokenRank,
    render_html_report as render_benchmark_html,
    run_benchmark,
    run_case,
)
```

Create `scripts/benchmark_interventions.py` that loads a Hugging Face decoder, wraps it with `jlens.from_hf`, loads a lens, runs the default cases, writes `index.html` and `results.json` into `--out`, and prints the output path plus pass count.

- [ ] **Step 4: Run test to verify it passes**

Run: `python3.11 -m pytest -q tests/test_benchmark.py::test_benchmark_exports_public_api`

Expected: pass.

- [ ] **Step 5: Run full verification**

Run:

```bash
python3.11 -m pytest -q
python3.11 -m ruff check .
```

Expected: all tests pass and ruff reports no errors.

- [ ] **Step 6: Commit**

Run:

```bash
git add jlens/__init__.py scripts/benchmark_interventions.py tests/test_benchmark.py
git commit -m "feat: add intervention benchmark cli"
```

### Task 4: Live Qwen Benchmark

**Files:**
- Generated only: `reports/intervention-benchmark/<timestamp>/index.html`
- Generated only: `reports/intervention-benchmark/<timestamp>/results.json`

**Interfaces:**
- Consumes: `scripts/benchmark_interventions.py`.
- Produces: local HTML report path for user inspection.

- [ ] **Step 1: Stop J Studio if it is holding the GPU**

Run: `pgrep -af "[p]ython3.11 -m jstudio.ui.app" || true`

If a process is listed, run `kill <pid>` for that J Studio process only.

- [ ] **Step 2: Run the benchmark**

Run:

```bash
python3.11 scripts/benchmark_interventions.py \
  --model Qwen/Qwen2.5-7B-Instruct \
  --lens /var/home/Heterodoxin/.cache/jstudio/lenses/Qwen--Qwen2.5-7B-Instruct/lens.pt \
  --local-files-only \
  --out reports/intervention-benchmark/qwen2_5_7b
```

Expected: command exits 0 and prints the HTML report path.

- [ ] **Step 3: Restart J Studio**

Run from `/var/home/Heterodoxin/Desktop/J-Studio/app`:

```bash
PYTHONPATH=/var/home/Heterodoxin/Desktop/J-Studio/app:/var/home/Heterodoxin/Desktop/J-Studio/jacobian-lens PYTORCH_HIP_ALLOC_CONF=expandable_segments:True python3.11 -m jstudio.ui.app
```

Expected: process remains running.
