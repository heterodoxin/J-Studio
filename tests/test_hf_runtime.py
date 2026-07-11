import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

from jstudio.domain import (
    ConceptActivation,
    InterventionDraft,
    InterventionOperation,
    LensFitState,
    RuleRecord,
    RuleTrigger,
)
from jstudio.rules.protocol import RuleEvaluationResult, ValidatedRuleAction
from jstudio.services.hf_runtime import (
    HFLensService,
    HFModelRuntime,
    ThinkingFilter,
    _readout_values,
    services_for_runtime,
)
from jstudio.services.protocols import GenerationRequest


class RuntimeDouble:
    model_id = "Qwen/test-real"
    revision = "cached-revision"
    layer_count = 36
    device = "ROCm 0"
    precision = "BF16"

    def read_activations(self, prompt, *, token_index, layers=(), max_concepts=200):
        assert prompt == "make me an ascii cat"
        return (
            ConceptActivation(
                term="cat",
                score=0.91,
                confidence=0.88,
                layer=18,
                token_index=token_index,
                rank=0,
            ),
        )

    def stream(self, prompt, *, max_new_tokens, history=()):
        self.last_history = history
        yield " /\\_/\\"
        yield "\n( o.o )"

    def close(self):
        pass


class CoordinatedRuntimeDouble(RuntimeDouble):
    def __init__(self):
        from jstudio.services.lens_fitting import GPUCoordinator

        self.coordinator = GPUCoordinator()


class Sink:
    def __init__(self):
        self.tokens = []
        self.frames = []
        self.interventions = []
        self.finished = threading.Event()
        self.error = None
    def on_started(self, run):
        self.run = run

    def on_token(self, run_id, token, output_text):
        self.tokens.append(token)

    def on_frame(self, frame):
        self.frames.append(frame)

    def on_intervention(self, intervention_id, state, detail):
        self.interventions.append((intervention_id, state, detail))

    def on_finished(self, run):
        self.run = run
        self.finished.set()

    def on_error(self, run_id, message, detail=""):
        self.error = (message, detail)
        self.finished.set()


class RuleSandboxDouble:
    available = True
    unavailable_reason = None

    def __init__(self, result):
        self.result = result
        self.requests = []

    def evaluate(self, request, *, cancel_event=None):
        self.requests.append(request)
        return self.result


class ControllerDouble:
    def __init__(self, status):
        self._status = status
        self.started = []
        self.cancelled = False

    def status(self):
        return self._status

    def start(self, *, force=False):
        self.started.append(force)

    def cancel(self):
        self.cancelled = True

    def subscribe(self, callback):
        return lambda: None


def test_readout_score_preserves_raw_z_score():
    score, confidence = _readout_values(8.0)
    assert score == 8.0
    assert 0.99 < confidence <= 1.0


def test_runtime_services_emit_real_runtime_output_and_readouts():
    services = services_for_runtime(RuntimeDouble())
    session = services.sessions.list_sessions()[0]
    sink = Sink()

    services.generation.start(
        GenerationRequest(
            session_id=session.session_id,
            prompt="make me an ascii cat",
        ),
        sink,
    )
    assert sink.finished.wait(timeout=2)

    assert session.model_id == "Qwen/test-real"
    assert session.device == "ROCm 0"
    assert not session.capabilities.intervene
    assert sink.error is None
    assert "".join(sink.tokens) == " /\\_/\\\n( o.o )"
    assert sink.frames[0].activations[0].term == "cat"
    assert sink.run.ttft_seconds is not None
    assert sink.run.decode_tokens_per_second is not None
    services.generation.close()


def test_generation_forwards_conversation_history_to_runtime():
    runtime = RuntimeDouble()
    services = services_for_runtime(runtime)
    session = services.sessions.list_sessions()[0]
    sink = Sink()
    history = (("user", "hi"), ("assistant", "hello"))

    services.generation.start(
        GenerationRequest(
            session_id=session.session_id,
            prompt="make me an ascii cat",
            history=history,
        ),
        sink,
    )
    assert sink.finished.wait(timeout=2)

    assert runtime.last_history == history
    services.generation.close()


def test_uncalibrated_runtime_rejects_intervention_preview():
    services = services_for_runtime(RuntimeDouble())
    session = services.sessions.list_sessions()[0]

    compatible, detail = services.interventions.preview(session.session_id, object())

    assert not compatible
    assert "calibrated" in detail.lower()
    services.generation.close()


def test_generation_remains_available_without_a_lens():
    runtime = RuntimeDouble()
    runtime.lens = None
    runtime.lens_id = None
    runtime.calibrated = False
    services = services_for_runtime(runtime)
    session = services.sessions.list_sessions()[0]
    sink = Sink()

    services.generation.start(
        GenerationRequest(session_id=session.session_id, prompt="make me an ascii cat"),
        sink,
    )
    assert sink.finished.wait(timeout=2)

    assert not session.capabilities.inspect
    assert sink.error is None
    assert sink.frames == []
    assert services.lens.fit_status().state is LensFitState.MISSING
    services.generation.close()


def test_lens_service_allows_refit_when_stable_lens_is_already_active():
    runtime = SimpleNamespace(
        lens=SimpleNamespace(metadata={"quality_stage": "Stable"}),
        lens_id="stable-id",
        calibrated=True,
    )
    controller = ControllerDouble(
        SimpleNamespace(state=LensFitState.MISSING, stage="", completed=0, total=0)
    )
    service = HFLensService(runtime, controller)

    assert service.fit_status().state is LensFitState.STABLE

    service.start_fit()

    assert controller.started == [True]


def test_lens_service_resumes_when_preview_lens_is_active():
    runtime = SimpleNamespace(
        lens=SimpleNamespace(metadata={"quality_stage": "Preview"}),
        lens_id="preview-id",
        calibrated=False,
    )
    controller = ControllerDouble(
        SimpleNamespace(state=LensFitState.FAILED, stage="Stable", completed=32, total=100)
    )
    service = HFLensService(runtime, controller)

    service.start_fit()

    assert controller.started == [False]


def test_runtime_rejects_stale_estimator_metadata():
    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.model_id = "model/test"
    runtime.revision = "abc"
    runtime.lens_model = SimpleNamespace(d_model=8, n_layers=4)
    lens = SimpleNamespace(
        d_model=8,
        metadata={
            "model": "model/test",
            "revision": "abc",
            "estimator": "independent-probe-blocks-v1",
            "target_layer": "2",
        },
    )

    try:
        runtime._validate_lens(lens)
    except ValueError as exc:
        assert "estimator" in str(exc)
    else:
        raise AssertionError("stale estimator was accepted")


def stable_lens_double(*, quality_stage="Stable", calibrated=True, quality_gate=True):
    metadata = {
        "model": "model/test",
        "revision": "abc",
        "estimator": "prompt-averaged-orthogonal-sketch-v2",
        "target_layer": "2",
        "quality_stage": quality_stage,
    }
    if quality_gate:
        metadata.update(
            {
                "quality_gate_version": "jspace-v1",
                "fit_quality_pass_at_10": "0.75",
                "fit_quality_rank_overlap": "0.90",
            }
        )
    lens = SimpleNamespace(
        d_model=8,
        source_layers=(0, 1),
        metadata=metadata,
    )
    lens.metric = lambda layer: SimpleNamespace(calibrated=calibrated)
    return lens


def stable_runtime_double():
    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.model_id = "model/test"
    runtime.revision = "abc"
    runtime.lens_model = SimpleNamespace(d_model=8, n_layers=4)
    return runtime


def test_runtime_accepts_preview_lens_for_viewing():
    runtime = stable_runtime_double()
    lens = stable_lens_double(quality_stage="Preview")

    runtime._validate_lens(lens, require_stable=False)


def test_runtime_rejects_stable_lens_without_current_jspace_quality_gate():
    runtime = stable_runtime_double()
    lens = stable_lens_double(quality_gate=False)

    try:
        runtime._validate_lens(lens, require_stable=True)
    except ValueError as exc:
        assert "quality gate" in str(exc).lower()
    else:
        raise AssertionError("stale stable lens was accepted without quality gate")


def test_runtime_rejects_uncalibrated_stable_lens_for_interventions():
    runtime = stable_runtime_double()
    lens = stable_lens_double(calibrated=False)

    try:
        runtime._validate_lens(lens, require_stable=True)
    except ValueError as exc:
        assert "calibrated" in str(exc).lower()
    else:
        raise AssertionError("uncalibrated stable lens was accepted for viewing")


def test_runtime_candidate_lenses_exclude_preview_artifacts(tmp_path):
    selected = tmp_path / "lens.pt"

    candidates = HFModelRuntime._candidate_lens_paths(selected)

    assert selected in candidates
    assert tmp_path / "stable.lens.pt" in candidates
    assert tmp_path / ".fit" / "stable.lens.pt" in candidates
    assert tmp_path / "preview.lens.pt" not in candidates
    assert tmp_path / ".fit" / "preview.lens.pt" not in candidates


def test_formatted_prompt_disables_chat_template_thinking():
    class Tokenizer:
        enable_thinking = None

        def apply_chat_template(self, messages, **kwargs):
            self.messages = messages
            self.enable_thinking = kwargs.get("enable_thinking")
            return "templated"

    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.tokenizer = Tokenizer()

    assert runtime._formatted_prompt("pick a word") == "templated"
    assert runtime.tokenizer.enable_thinking is False
    assert "Do not emit <think>" in runtime.tokenizer.messages[0]["content"]


def test_thinking_filter_hides_split_think_blocks():
    filtr = ThinkingFilter()
    visible = []
    for chunk in ("<thi", "nk>secret</th", "ink>Answer"):
        part = filtr.feed(chunk)
        if part:
            visible.append(part)
    tail = filtr.flush()
    if tail:
        visible.append(tail)

    assert visible == ["Answer"]


def test_steer_injection_reports_not_applied_when_no_coherent_steer(monkeypatch):
    """Inject routes through J-space generation steering; when no strength
    produces the concept coherently, the result must be honestly not-applied."""
    import jlens.hooks as hooks
    import jlens.interventions as ji
    import torch

    monkeypatch.setattr(hooks, "ActivationEditor", lambda layers, edits: nullcontext())
    monkeypatch.setattr(
        hooks, "ActivationRecorder",
        lambda layers, at: _RecorderDouble({layer: torch.ones(1, 3, 4) for layer in at}),
    )

    class Tok:
        eos_token_id = 0

        def __call__(self, text, return_tensors=None):
            return SimpleNamespace(input_ids=torch.zeros(1, 3, dtype=torch.long))

        def decode(self, ids, skip_special_tokens=False):
            return "the the the"  # coherent but never the concept

    class Model:
        device = "cpu"

        def generate(self, input_ids, **kwargs):
            return torch.zeros(1, input_ids.shape[1] + 4, dtype=torch.long)

    class LensModel:
        layers = [object()]

        def forward(self, ids):
            return None

    class Coordinator:
        def exclusive(self, name):
            return nullcontext()

    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.lens = SimpleNamespace(source_layers=(0, 1, 2, 3))
    runtime.lens_model = LensModel()
    runtime.model = Model()
    runtime.tokenizer = Tok()
    runtime._torch = torch
    runtime.calibrated = True
    runtime.coordinator = Coordinator()
    runtime._formatted_prompt = lambda prompt: f"formatted:{prompt}"
    monkeypatch.setattr(
        ji, "ConceptResolver",
        lambda tokenizer: SimpleNamespace(
            resolve=lambda term: SimpleNamespace(token_ids=(42,))
        ),
    )
    draft = InterventionDraft(InterventionOperation.INJECT, None, "cat", 16.0, 0, 3)

    editors, results = runtime.prepare_interventions("prompt", (draft,))

    assert len(editors) == 1
    assert not results[0].success
    assert "no-coherent-steer" in results[0].trace.warnings


class _RecorderDouble:
    def __init__(self, activations):
        self.activations = activations

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_prepare_interventions_verifies_replace_on_generation_path(monkeypatch):
    """replace edits must reproduce on the generation path, not just the search."""
    import jlens
    from jlens.interventions import InterventionResult, InterventionTrace

    trace = InterventionTrace(
        operation="replace",
        target_ids=(42,),
        source_ids=(7,),
        experimental=False,
        selected_layer=2,
        selected_positions=(5,),
        selected_scale=1.0,
        normalized_cost=1.0,
        baseline_scores={"42": 0.0},
        after_scores={"42": 1.0},
        baseline_top_ids=(1,),
        after_top_ids=(42,),
        search_points=(),
        warnings=(),
    )
    engine_result = InterventionResult(True, trace, None, "minimum passing")

    class EngineDouble:
        def __init__(self, model, lens):
            pass

        def replace(self, prompt, source, target, **options):
            return engine_result

        def apply(self, result):
            return nullcontext()

    class Coordinator:
        def exclusive(self, name):
            return nullcontext()

    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.lens = SimpleNamespace(source_layers=(0, 1, 2, 3))
    runtime.lens_model = object()
    runtime.calibrated = True
    runtime.coordinator = Coordinator()
    runtime._formatted_prompt = lambda prompt: f"formatted:{prompt}"
    # search claims success, but the edit does not reproduce on generation
    runtime._editor_targets_next_token = lambda prompt, editor, target_ids: False
    monkeypatch.setattr(jlens, "InterventionEngine", EngineDouble)
    draft = InterventionDraft(InterventionOperation.REPLACE, "dog", "cat", 16.0, 0, 3)

    editors, results = runtime.prepare_interventions("prompt", (draft,))

    assert not results[0].success
    assert "generation-path-verification-failed" in results[0].trace.warnings


def test_generation_applies_intervention_drafts():
    runtime = RuntimeDouble()
    runtime.prepared = []

    def prepare(prompt, drafts):
        runtime.prepared.append((prompt, drafts))
        result = SimpleNamespace(success=True, message="minimum passing scale")
        return (nullcontext(),), (result,)

    runtime.prepare_interventions = prepare
    services = services_for_runtime(runtime)
    session = services.sessions.list_sessions()[0]
    sink = Sink()
    draft = InterventionDraft(
        InterventionOperation.INJECT,
        None,
        "cat",
        4.0,
        0,
        20,
    )

    services.generation.start(
        GenerationRequest(
            session_id=session.session_id,
            prompt="make me an ascii cat",
            intervention_ids=("inject-cat",),
            intervention_drafts=(draft,),
        ),
        sink,
    )
    assert sink.finished.wait(timeout=2)

    assert runtime.prepared == [("make me an ascii cat", (draft,))]
    assert sink.interventions == [
        ("inject-cat", "applied", "minimum passing scale")
    ]
    services.generation.close()


def test_generation_converts_rule_actions_to_intervention_drafts():
    runtime = RuntimeDouble()
    runtime.prepared = []

    def prepare(prompt, drafts):
        runtime.prepared.append((prompt, drafts))
        result = SimpleNamespace(success=True, message="minimum passing scale")
        return tuple(nullcontext() for _ in drafts), tuple(result for _ in drafts)

    runtime.prepare_interventions = prepare
    rule = RuleRecord(
        "rule-cat",
        "Cat Rule",
        "function run(ctx) { return []; }",
        RuleTrigger.BEFORE_TOKEN,
        enabled=True,
        trusted=True,
        config={"target": "cat"},
    )
    sandbox = RuleSandboxDouble(
        RuleEvaluationResult(
            True,
            (
                ValidatedRuleAction(
                    "inject",
                    {
                        "type": "inject",
                        "term": "cat",
                        "strength": 3.0,
                        "layers": {"from": 2, "to": 4},
                        "duration": "next-token",
                    },
                ),
            ),
        )
    )
    services = services_for_runtime(runtime, rules=sandbox)
    session = services.sessions.list_sessions()[0]
    sink = Sink()

    services.generation.start(
        GenerationRequest(
            session_id=session.session_id,
            prompt="make me an ascii cat",
            rule_ids=("rule-cat",),
            rule_records=(rule,),
        ),
        sink,
    )
    assert sink.finished.wait(timeout=2)

    assert sandbox.requests[0].context["config"] == {"target": "cat"}
    generated = runtime.prepared[0][1][0]
    assert generated.operation is InterventionOperation.INJECT
    assert generated.target_term == "cat"
    assert generated.strength == 3.0
    assert generated.layer_start == 2
    assert generated.layer_end == 4
    assert generated.trigger == "rule:rule-cat"
    assert sink.interventions[0][0].startswith("rule-cat:")
    services.generation.close()


def test_generation_streams_before_deferred_lens_analysis():
    runtime = RuntimeDouble()
    events = []
    real_read = runtime.read_activations
    real_stream = runtime.stream

    def read(*args, **kwargs):
        events.append("read")
        return real_read(*args, **kwargs)

    def stream(*args, **kwargs):
        for token in real_stream(*args, **kwargs):
            events.append("token")
            yield token

    runtime.read_activations = read
    runtime.stream = stream
    services = services_for_runtime(runtime)
    session = services.sessions.list_sessions()[0]
    sink = Sink()

    services.generation.start(
        GenerationRequest(session_id=session.session_id, prompt="make me an ascii cat"),
        sink,
    )
    assert sink.finished.wait(timeout=2)

    assert events[0] == "token"
    assert events[-1] == "read"
    services.generation.close()


def test_generation_finishes_before_slow_lens_analysis():
    runtime = RuntimeDouble()
    read_entered = threading.Event()
    release_read = threading.Event()

    def read(*args, **kwargs):
        read_entered.set()
        release_read.wait(timeout=2)
        return ()

    runtime.read_activations = read
    services = services_for_runtime(runtime)
    session = services.sessions.list_sessions()[0]
    sink = Sink()

    services.generation.start(
        GenerationRequest(session_id=session.session_id, prompt="make me an ascii cat"),
        sink,
    )
    assert read_entered.wait(timeout=2)
    assert sink.finished.wait(timeout=0.1)
    release_read.set()

    assert sink.error is None
    assert sink.run.output_text == " /\\_/\\\n( o.o )"
    services.generation.close()


def test_lens_readout_failure_does_not_fail_generation():
    runtime = RuntimeDouble()

    def read(*args, **kwargs):
        raise RuntimeError("lens failed")

    runtime.read_activations = read
    services = services_for_runtime(runtime)
    session = services.sessions.list_sessions()[0]
    sink = Sink()

    services.generation.start(
        GenerationRequest(session_id=session.session_id, prompt="make me an ascii cat"),
        sink,
    )
    assert sink.finished.wait(timeout=2)

    assert sink.error is None
    assert sink.run.output_text == " /\\_/\\\n( o.o )"
    assert sink.frames == []
    services.generation.close()


def test_generation_priority_blocks_background_fit_between_prompts():
    runtime = CoordinatedRuntimeDouble()
    fit_entered = threading.Event()
    release_generation = threading.Event()
    events = []

    def stream(prompt, *, max_new_tokens, history=()):
        with runtime.coordinator.generation():
            events.append("generation")
            release_generation.wait(timeout=2)
            yield "token"

    def fit():
        with runtime.coordinator.exclusive("lens-fit-prompt"):
            events.append("fit")
            fit_entered.set()

    runtime.stream = stream
    services = services_for_runtime(runtime)
    session = services.sessions.list_sessions()[0]
    sink = Sink()
    services.generation.start(
        GenerationRequest(session_id=session.session_id, prompt="make me an ascii cat"),
        sink,
    )
    while events != ["generation"]:
        time.sleep(0.001)
    thread = threading.Thread(target=fit)
    thread.start()
    assert not fit_entered.wait(timeout=0.05)
    release_generation.set()
    assert sink.finished.wait(timeout=2)
    thread.join(timeout=2)

    assert events == ["generation", "fit"]
    services.generation.close()
