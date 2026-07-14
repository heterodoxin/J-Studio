# Fast ROCm Generation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Q4_K_M llama.cpp ROCm chat backend for the same Qwen3-8B checkpoint, expose exact BF16 fallback, and measure at least 40 tokens/s on the current host.

**Architecture:** A process-isolated llama-server streams OpenAI-compatible SSE into the existing generation service. Generation mode and timing provenance are immutable run data; the BF16 Transformers runtime remains loaded for J-space analysis and exact generation.

**Tech Stack:** Python 3.11, llama.cpp HIP/ROCm, GGUF Q4_K_M, PySide6, local HTTP/SSE, pytest

## Global Constraints

- Fast mode is the default and must identify itself as Q4_K_M, not BF16.
- Exact BF16 remains available and behaviorally compatible.
- The fast artifact is converted from `heterodoxin/qwen3-8b-apostate` and stored outside Git.
- ROCm graph execution remains disabled because it crashed during validation.
- A failed fast backend falls back to Exact BF16 without losing the prompt.
- Each completed run reports backend, quantization, time to first token, and decode tokens/s.
- Fast-mode acceptance is at least 40 tokens/s over 256 generated tokens after warmup; 100 tokens/s is preferred.

---

### Task 1: Generation Mode and Provenance Records

**Files:**
- Modify: `jstudio/domain.py`
- Modify: `jstudio/services/protocols.py`
- Modify: `jstudio/project.py`
- Test: `tests/test_domain.py`
- Test: `tests/test_project.py`

**Interfaces:**
- Produces: `GenerationBackend(StrEnum): FAST = "fast-q4", EXACT = "exact-bf16"`
- Adds: `GenerationRequest.backend: GenerationBackend = GenerationBackend.FAST`
- Adds: `RunRecord.generation_backend`, `quantization`, `ttft_seconds`, `decode_tokens_per_second`

- [ ] **Step 1: Write failing round-trip tests**

```python
def test_run_records_generation_provenance():
    run = replace(RunRecord.create(prompt="hello", mode=RunMode.BASELINE),
                  generation_backend=GenerationBackend.FAST,
                  quantization="Q4_K_M", ttft_seconds=0.2,
                  decode_tokens_per_second=72.5)
    assert run.decode_tokens_per_second == 72.5

def test_project_round_trips_generation_provenance(tmp_path):
    project = ProjectDocument.new("Timing")
    project.runs.append(replace(
        RunRecord.create(prompt="hello", mode=RunMode.BASELINE),
        generation_backend=GenerationBackend.FAST,
        quantization="Q4_K_M",
        ttft_seconds=0.2,
        decode_tokens_per_second=72.5,
    ))
    path = tmp_path / "timing.jstudio.json"
    project.save(path)
    loaded = ProjectDocument.load(path)
    assert loaded.runs[0].generation_backend is GenerationBackend.FAST
    assert loaded.runs[0].quantization == "Q4_K_M"
    assert loaded.runs[0].ttft_seconds == 0.2
    assert loaded.runs[0].decode_tokens_per_second == 72.5
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_domain.py tests/test_project.py -q`

- [ ] **Step 3: Add validated optional timing fields and serialization**

Reject negative/non-finite timing values. Keep absent fields readable for existing project files by supplying defaults in the project decoder.

- [ ] **Step 4: Run tests and verify GREEN**

Run: `python -m pytest tests/test_domain.py tests/test_project.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jstudio/domain.py jstudio/services/protocols.py jstudio/project.py tests/test_domain.py tests/test_project.py
git commit -m "feat: record generation backend and throughput"
```

### Task 2: Reproducible llama.cpp and GGUF Preparation

**Files:**
- Create: `scripts/prepare_fast_runtime.py`
- Create: `tests/test_prepare_fast_runtime.py`
- Modify: `.gitignore`

**Interfaces:**
- Produces: `prepare(source_model, cache_root, revision) -> PreparedRuntime`
- Artifact defaults: `~/.cache/jstudio/llama.cpp/`, `~/.cache/jstudio/models/qwen3-8b-apostate-Q4_K_M.gguf`

- [ ] **Step 1: Write failing command-planning tests**

```python
def test_prepare_builds_hip_and_quantizes_outside_repo(tmp_path, runner):
    prepared = prepare("heterodoxin/qwen3-8b-apostate", tmp_path, "pinned-revision", runner=runner)
    assert any("GGML_HIP=ON" in " ".join(call) for call in runner.calls)
    assert any("Q4_K_M" in call for call in runner.calls)
    assert prepared.server.name == "llama-server"
```

Test idempotence by pre-creating both executable and GGUF and asserting no build/convert commands run.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_prepare_fast_runtime.py -q`

- [ ] **Step 3: Implement explicit preparation stages**

The script must clone/update a user-supplied revision, configure with `cmake -B build -DGGML_HIP=ON -DLLAMA_CURL=OFF`, build `llama-server`, `llama-quantize`, and `llama-gguf-split`, run `convert_hf_to_gguf.py`, then quantize to Q4_K_M. Every subprocess uses argument arrays, `check=True`, and a visible stage label. Never remove an existing artifact automatically.

- [ ] **Step 4: Run tests and dry-run output**

Run: `python -m pytest tests/test_prepare_fast_runtime.py -q`

Run: `python scripts/prepare_fast_runtime.py --dry-run`

Expected: prints cache-only paths and exact build/convert commands without changing files.

- [ ] **Step 5: Commit**

```bash
git add scripts/prepare_fast_runtime.py tests/test_prepare_fast_runtime.py .gitignore
git commit -m "feat: prepare ROCm llama.cpp generation runtime"
```

### Task 3: Process-Isolated Streaming Client

**Files:**
- Create: `jstudio/services/llama_runtime.py`
- Test: `tests/test_llama_runtime.py`

**Interfaces:**
- Produces: `LlamaServerRuntime.start()`, `stream(messages, max_new_tokens)`, `close()`
- Produces: `FastRuntimeUnavailable(RuntimeError)`
- Consumes a llama-server executable and GGUF path; binds only to `127.0.0.1`.

- [ ] **Step 1: Write failing lifecycle and SSE tests**

Use a temporary Python HTTP server fixture that returns:

```text
data: {"choices":[{"delta":{"content":"hello"}}]}
data: {"choices":[{"delta":{"content":" world"}}]}
data: [DONE]
```

Assert `list(runtime.stream(...)) == ["hello", " world"]`, startup timeout raises `FastRuntimeUnavailable`, cancellation closes the response, and `close()` terminates the child process.

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_llama_runtime.py -q`

- [ ] **Step 3: Implement safe process startup and health probing**

Launch with `--host 127.0.0.1 --port <reserved-port> --model <gguf> --ctx-size 4096 --n-gpu-layers 99 --flash-attn on`. Poll `/health` with a bounded timeout, capture stderr to a rotating cache log, and include the last diagnostic lines in startup errors.

- [ ] **Step 4: Implement incremental SSE parsing**

Use `urllib.request` with a JSON body for `/v1/chat/completions`, `stream=True`, `temperature=0`, and the requested token cap. Parse only `data:` lines, stop on `[DONE]`, and yield non-empty delta content.

- [ ] **Step 5: Run tests and verify GREEN**

Run: `python -m pytest tests/test_llama_runtime.py -q`

- [ ] **Step 6: Commit**

```bash
git add jstudio/services/llama_runtime.py tests/test_llama_runtime.py
git commit -m "feat: stream from process-isolated llama.cpp"
```

### Task 4: Hybrid Generation Routing, Timing, and Fallback

**Files:**
- Create: `jstudio/services/hybrid_generation.py`
- Modify: `jstudio/services/hf_runtime.py`
- Modify: `jstudio/ui/app.py`
- Test: `tests/test_hybrid_generation.py`
- Modify: `tests/test_app_arguments.py`

**Interfaces:**
- Produces: `HybridGenerationRuntime.stream(prompt, backend, max_new_tokens) -> Iterator[str]`
- Produces: `TimedStreamResult` values applied to the final `RunRecord`
- Adds CLI options: `--fast-model PATH`, `--llama-server PATH`, `--exact-generation`

- [ ] **Step 1: Write failing routing tests**

```python
def test_fast_failure_falls_back_without_losing_prompt():
    runtime = HybridGenerationRuntime(FailingFast(), ExactDouble())
    chunks = list(runtime.stream("same prompt", GenerationBackend.FAST, 32))
    assert chunks == ["exact"]
    assert runtime.last_backend is GenerationBackend.EXACT
    assert ExactDouble.last_prompt == "same prompt"
```

```python
def test_timing_excludes_first_token_latency():
    clock = iter((0.0, 0.20, 0.22, 0.24, 0.26)).__next__
    timing = StreamTiming(clock=clock)
    timing.start()
    for _ in range(4):
        timing.token()
    result = timing.finish(GenerationBackend.FAST, "Q4_K_M")
    assert result.ttft_seconds == pytest.approx(0.20)
    assert result.decode_tokens_per_second == pytest.approx(50.0)
```

- [ ] **Step 2: Run tests and verify RED**

Run: `python -m pytest tests/test_hybrid_generation.py tests/test_app_arguments.py -q`

- [ ] **Step 3: Implement routing and measured completion**

Keep J-space reads on `HFModelRuntime`. Route only chat token streaming through the hybrid runtime. When fast startup/streaming fails before the first token, retry Exact BF16; after any fast token has been emitted, surface the failure rather than splicing different model outputs.

- [ ] **Step 4: Wire CLI discovery**

Resolve explicit CLI paths first, then `JSTUDIO_LLAMA_SERVER`/`JSTUDIO_FAST_MODEL`, then the cache defaults produced by Task 2. Missing artifacts are a recoverable Exact-mode condition.

- [ ] **Step 5: Run focused tests**

Run: `python -m pytest tests/test_hybrid_generation.py tests/test_hf_runtime.py tests/test_app_arguments.py -q`

Expected: PASS.

- [ ] **Step 6: Commit**

```bash
git add jstudio/services/hybrid_generation.py jstudio/services/hf_runtime.py jstudio/ui/app.py tests/test_hybrid_generation.py tests/test_app_arguments.py
git commit -m "feat: route chat through fast ROCm generation"
```

### Task 5: Chat Mode and Live Throughput UI

**Files:**
- Modify: `jstudio/ui/chat.py`
- Modify: `jstudio/services/fake.py`
- Modify: `tests/ui/test_chat.py`

**Interfaces:**
- Consumes: `GenerationBackend`, final `RunRecord` timing fields
- Produces: Fast/Exact selector and visible backend/TTFT/tokens-per-second status.

- [ ] **Step 1: Write failing UI tests**

```python
def test_chat_defaults_to_fast_and_sends_selected_backend(window, qtbot):
    assert window.chat_workspace.backend.currentData() is GenerationBackend.FAST
    window.chat_workspace.backend.setCurrentIndex(1)
    window.chat_workspace.composer.setPlainText("hello")
    qtbot.mouseClick(window.chat_workspace.send_button, Qt.MouseButton.LeftButton)
    assert window.services.generation.last_request.backend is GenerationBackend.EXACT

def test_chat_shows_completed_throughput(window):
    window.chat_workspace._on_finished(run_with_metrics)
    assert "72.5 tok/s" in window.chat_workspace.performance.text()
    assert "Q4_K_M" in window.chat_workspace.performance.text()
```

- [ ] **Step 2: Run tests and verify RED**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_chat.py -q`

- [ ] **Step 3: Add compact generation controls**

Place a `QComboBox` with `Fast (Q4)` and `Exact (BF16)` beside Send. Add a performance label that shows `Starting…`, then backend and live elapsed state, then final `TTFT 0.20 s · 72.5 tok/s · Q4_K_M`.

Store the most recent request on `FakeGenerationService.last_request` so the UI test asserts the real service boundary rather than widget internals.

- [ ] **Step 4: Run UI tests**

Run: `QT_QPA_PLATFORM=offscreen python -m pytest tests/ui/test_chat.py tests/ui/test_accessibility.py -q`

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add jstudio/ui/chat.py jstudio/services/fake.py tests/ui/test_chat.py
git commit -m "feat: expose fast generation and throughput in chat"
```

### Task 6: Real ROCm Benchmark and Release Gate

**Files:**
- Create: `scripts/benchmark_generation.py`
- Create: `tests/test_benchmark_generation.py`
- Modify: `README.md`

**Interfaces:**
- Consumes both generation backends.
- Produces machine-readable JSON containing backend, prompt tokens, output tokens, TTFT, decode seconds, and decode tokens/s.

- [ ] **Step 1: Write a failing benchmark-math test**

```python
def test_benchmark_excludes_first_token_from_decode_rate():
    result = summarize([0.20, 0.22, 0.24, 0.26])
    assert result.ttft_seconds == 0.20
    assert result.decode_tokens_per_second == 50.0
```

- [ ] **Step 2: Run test and verify RED**

Run: `python -m pytest tests/test_benchmark_generation.py -q`

- [ ] **Step 3: Implement warmup plus 256-token benchmark**

Run one 32-token warmup, synchronize completion, then request 256 tokens with a prompt that avoids early EOS. Exit nonzero under `--require-tok-s 40` when Fast mode misses the threshold.

- [ ] **Step 4: Run all automated checks**

Run: `QT_QPA_PLATFORM=offscreen QTWEBENGINE_CHROMIUM_FLAGS=--disable-gpu python -m pytest -q`

Run: `python -m ruff check jstudio tests scripts`

Expected: all tests and lint PASS.

- [ ] **Step 5: Prepare the local runtime and run the acceptance benchmark**

```bash
python scripts/prepare_fast_runtime.py
python scripts/benchmark_generation.py --backend fast --output-tokens 256 --require-tok-s 40
```

Expected: exit 0 and JSON reports at least `40.0` decode tokens/s. Record the actual result in README; do not claim 100 tokens/s unless measured.

- [ ] **Step 6: Launch and smoke-test J Studio**

Run: `python -m jstudio`

Verify Fast is the default, tokens stream, Stop works, throughput is shown, Exact BF16 remains selectable, and Inspect with J-Lens uses the BF16 analysis runtime.

- [ ] **Step 7: Commit**

```bash
git add scripts/benchmark_generation.py tests/test_benchmark_generation.py README.md
git commit -m "test: gate fast generation on measured ROCm throughput"
```
