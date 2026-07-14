import threading
import time
from contextlib import nullcontext
from types import SimpleNamespace

import torch

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
    RuntimeInterventionService,
    ThinkingFilter,
    _causal_token_effect,
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
        assert prompt == "make me an ascii cat\n\n /\\_/\\\n( o.o )"
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


def test_generation_uses_the_same_full_context_for_interventions_and_jlens():
    class ContextRuntime(RuntimeDouble):
        lens = object()

        def __init__(self):
            self.prepared = []
            self.read_contexts = []
            self.read_finished = threading.Event()

        def inspection_context(self, prompt, output, *, history=()):
            self.inspection_args = (prompt, output, history)
            return "<chat>prior turn|current prompt|assistant:generated answer"

        def prepare_interventions(self, prompt, drafts, *, history=()):
            self.prepared.append((prompt, drafts, history))
            result = SimpleNamespace(success=True, message="minimum passing scale")
            return (nullcontext(),), (result,)

        def read_activations(
            self, prompt, *, token_index, layers=(), max_concepts=200
        ):
            self.read_contexts.append(prompt)
            self.read_finished.set()
            return ()

    runtime = ContextRuntime()
    services = services_for_runtime(runtime)
    session = services.sessions.list_sessions()[0]
    sink = Sink()
    history = (("user", "prior turn"), ("assistant", "prior answer"))
    draft = InterventionDraft(
        InterventionOperation.INJECT, None, "cat", 4.0, 0, 20
    )

    services.generation.start(
        GenerationRequest(
            session_id=session.session_id,
            prompt="make me an ascii cat",
            history=history,
            intervention_ids=("inject-cat",),
            intervention_drafts=(draft,),
        ),
        sink,
    )
    assert sink.finished.wait(timeout=2)
    assert runtime.read_finished.wait(timeout=2)

    assert runtime.prepared == [("make me an ascii cat", (draft,), history)]
    assert runtime.inspection_args == (
        "make me an ascii cat",
        " /\\_/\\\n( o.o )",
        history,
    )
    assert runtime.read_contexts == [
        "<chat>prior turn|current prompt|assistant:generated answer"
    ]
    assert sink.run.inspection_text == runtime.read_contexts[0]
    services.generation.close()


def test_uncalibrated_runtime_rejects_intervention_preview():
    services = services_for_runtime(RuntimeDouble())
    session = services.sessions.list_sessions()[0]

    compatible, detail = services.interventions.preview(session.session_id, object())

    assert not compatible
    assert "calibrated" in detail.lower()
    services.generation.close()


def test_bake_converts_enabled_stack_to_jspace_projection_export(
    monkeypatch, tmp_path
):
    import jlens

    runtime = SimpleNamespace(
        calibrated=True,
        lens=SimpleNamespace(source_layers=(2, 4, 6)),
        lens_model=object(),
    )
    service = RuntimeInterventionService(runtime)
    draft = InterventionDraft(
        InterventionOperation.SUPPRESS,
        "refusal",
        None,
        4.0,
        2,
        4,
    )
    seen = {}

    def bake(model, lens, rules):
        seen["rules"] = rules
        return SimpleNamespace()

    def save(path, result):
        seen["path"] = path
        return tmp_path / "edit.safetensors", tmp_path / "edit.json"

    monkeypatch.setattr(jlens, "bake_projection", bake)
    monkeypatch.setattr(jlens, "save_projection_bake", save)

    weights, manifest = service.bake("session", (draft,), tmp_path / "edit")

    rule = seen["rules"][0]
    assert rule.operation == "suppress"
    assert rule.source == "refusal"
    assert rule.strength == 0.25
    assert rule.layers == (2, 4)
    assert weights.name == "edit.safetensors"
    assert manifest.name == "edit.json"


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


def test_runtime_requires_dense_lens_to_pass_the_reference_viewing_gate():
    runtime = stable_runtime_double()
    lens = stable_lens_double()
    lens.metadata.update(
        {
            "estimator": "projected-dense-jacobian-v2",
            "target_layer": "3",
            "transport_shrinkage": "0.75",
        }
    )

    try:
        runtime._validate_lens(lens, require_stable=True)
    except ValueError as exc:
        assert "viewing" in str(exc).lower()
    else:
        raise AssertionError("dense lens without viewing evidence was accepted")

    lens.metadata.update(
        {
            "quality_gate_version": "jspace-viewing-v2",
            "viewing_passed": "3",
            "viewing_total": "3",
        }
    )
    runtime._validate_lens(lens, require_stable=True)


def test_dense_lens_identity_reports_viewing_evidence_and_shrinkage():
    runtime = stable_runtime_double()
    lens = stable_lens_double()
    lens.n_prompts = 16
    lens.metadata.update(
        {
            "estimator": "projected-dense-jacobian-v2",
            "viewing_passed": "3",
            "viewing_total": "3",
            "transport_shrinkage": "0.75",
        }
    )
    runtime.lens = lens

    runtime._update_lens_identity()

    assert runtime.lens_id == "dense-jacobian-n16 · viewing 3/3 · J×0.75"


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


def test_inspection_context_appends_response_to_exact_chat_template():
    class Tokenizer:
        def apply_chat_template(self, messages, **kwargs):
            return "<assistant>"

    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.tokenizer = Tokenizer()

    context = runtime.inspection_context(
        "current", "generated", history=(("user", "prior"),)
    )

    assert context == "<assistant>generated"


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


def test_causal_token_effect_requires_directional_generated_change():
    baseline = (1, 2, 3, 4)
    target = ((9, 10),)
    source = ((2, 3),)

    assert _causal_token_effect("inject", baseline, (1, 9, 10, 4), (), target)[0]
    assert not _causal_token_effect("inject", baseline, (1, 8, 3, 4), (), target)[0]
    assert _causal_token_effect(
        "replace", baseline, (1, 9, 10, 4), source, target
    )[0]
    assert not _causal_token_effect(
        "replace", baseline, (9, 10, 1, 2, 3, 4), source, target
    )[0]
    assert _causal_token_effect("suppress", baseline, (1, 7, 8, 4), source, ())[0]


def test_causal_injection_rejects_target_only_completion():
    baseline = (1, 2, 3, 4)
    target = ((9,),)

    assert not _causal_token_effect("inject", baseline, (9,), (), target)[0]
    assert not _causal_token_effect("inject", baseline, (9, 9, 9), (), target)[0]
    assert not _causal_token_effect(
        "inject", baseline, (1, 2, 3, 4, 9, 9, 9), (), target
    )[0]
    assert _causal_token_effect("inject", baseline, (9, 7, 8), (), target)[0]


def test_causal_suppression_without_literal_baseline_requires_divergence():
    baseline = (1, 2, 3)

    assert not _causal_token_effect("suppress", baseline, baseline, ((9,),), ())[0]
    assert _causal_token_effect("suppress", baseline, (1, 2, 4), ((9,),), ())[0]


def test_phrase_effect_probe_measures_generated_tokens_and_cleans_hooks():
    runtime = HFModelRuntime.__new__(HFModelRuntime)
    layer = torch.nn.Identity()
    runtime.lens_model = SimpleNamespace(layers=[layer])
    samples = iter(((1, 2, 3), (1, 2, 3), (9, 2, 3)))
    runtime._causal_probe_ids = lambda _prompt: next(samples)
    runtime._token_variants = lambda term: ((9,),) if term else ()
    draft = InterventionDraft(InterventionOperation.INJECT, None, "cat", 16, 0, 0)
    operator = SimpleNamespace(
        make_transform=lambda: (lambda residual: residual)
    )

    probe = runtime._make_phrase_effect_probe("formatted", draft)
    first = probe(((0, operator),), (-1,))
    second = probe(((0, operator),), (-1,))

    assert not first[0]
    assert second[0]
    assert not layer._forward_hooks


def test_phrase_injection_failure_stays_unapplied_without_fallback(monkeypatch):
    import jlens

    class EngineDouble:
        def __init__(self, model, lens):
            pass

        def phrase_inject(self, prompt, target, **options):
            trace = jlens.InterventionTrace(
                operation="inject",
                target_ids=(42, 43),
                source_ids=(),
                experimental=True,
                selected_layer=None,
                selected_positions=(-1,),
                selected_scale=8.0,
                normalized_cost=0.0,
                baseline_scores={},
                after_scores={},
                baseline_top_ids=(),
                after_top_ids=(),
                search_points=(),
                warnings=("no-effective-jspace-strength",),
            )
            return jlens.InterventionResult(False, trace, None, "bounded failure")

        def apply(self, result, **_options):
            raise AssertionError("failed interventions must not be applied")

    class Coordinator:
        def exclusive(self, name):
            return nullcontext()

    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.lens = SimpleNamespace(source_layers=(0, 1, 2, 3))
    runtime.lens_model = object()
    runtime.layer_count = 4
    runtime.calibrated = True
    runtime.coordinator = Coordinator()
    runtime._formatted_prompt = lambda prompt, history=(): f"formatted:{prompt}"
    runtime._make_phrase_effect_probe = lambda *_args: (lambda *_probe: (True, 1.0))
    monkeypatch.setattr(jlens, "InterventionEngine", EngineDouble)
    draft = InterventionDraft(
        InterventionOperation.INJECT, None, "ASCII cat", 16.0, 0, 3
    )

    editors, results = runtime.prepare_interventions("prompt", (draft,))

    assert len(editors) == 1
    assert not results[0].success
    assert "no-effective-jspace-strength" in results[0].trace.warnings


class _RecorderDouble:
    def __init__(self, activations):
        self.activations = activations

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def test_prepare_interventions_does_not_require_next_token_match(monkeypatch):
    """J-space phrase edits are causal transforms, not forced next tokens."""
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

        def phrase_replace(self, prompt, source, target, **options):
            return engine_result

        def apply(self, result, **_options):
            return nullcontext()

    class Coordinator:
        def exclusive(self, name):
            return nullcontext()

    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.lens = SimpleNamespace(source_layers=(0, 1, 2, 3))
    runtime.lens_model = object()
    runtime.layer_count = 4
    runtime.calibrated = True
    runtime.coordinator = Coordinator()
    runtime._formatted_prompt = lambda prompt, history=(): f"formatted:{prompt}"
    runtime._make_phrase_effect_probe = lambda *_args: (lambda *_probe: (True, 1.0))
    # A J-space replacement need not make its target the immediate next token.
    runtime._editor_targets_next_token = lambda prompt, editor, target_ids: False
    monkeypatch.setattr(jlens, "InterventionEngine", EngineDouble)
    draft = InterventionDraft(InterventionOperation.REPLACE, "dog", "cat", 16.0, 0, 3)

    editors, results = runtime.prepare_interventions("prompt", (draft,))

    assert results[0].success
    assert "generation-path-verification-failed" not in results[0].trace.warnings


def test_prepare_interventions_dispatches_all_operations_to_phrase_engine(monkeypatch):
    import jlens

    calls = []
    applied = []

    class EngineDouble:
        def __init__(self, model, lens):
            pass

        def phrase_inject(self, prompt, target, **options):
            calls.append(("inject", prompt, None, target, options))
            return SimpleNamespace(
                success=True,
                message="inject ready",
                trace=SimpleNamespace(target_ids=(42, 43)),
            )

        def phrase_suppress(self, prompt, source, **options):
            calls.append(("suppress", prompt, source, None, options))
            return SimpleNamespace(
                success=True,
                message="suppress ready",
                trace=SimpleNamespace(target_ids=()),
            )

        def phrase_replace(self, prompt, source, target, **options):
            calls.append(("replace", prompt, source, target, options))
            return SimpleNamespace(
                success=True,
                message="replace ready",
                trace=SimpleNamespace(target_ids=(44, 45, 46)),
            )

        def apply(self, result, **_options):
            applied.append(result.message)
            return nullcontext()

    class Coordinator:
        def exclusive(self, name):
            return nullcontext()

    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.lens = SimpleNamespace(source_layers=(0, 1, 2, 3))
    runtime.lens_model = object()
    runtime.layer_count = 4
    runtime.calibrated = True
    runtime.coordinator = Coordinator()
    runtime._formatted_prompt = lambda prompt, history=(): f"formatted:{prompt}"
    runtime._make_phrase_effect_probe = lambda *_args: (lambda *_probe: (True, 1.0))
    runtime._editor_targets_next_token = lambda *args: (_ for _ in ()).throw(
        AssertionError("next-token verification is forbidden")
    )
    monkeypatch.setattr(jlens, "InterventionEngine", EngineDouble)
    drafts = (
        InterventionDraft(InterventionOperation.INJECT, None, "ASCII cat", 8, 0, 3),
        InterventionDraft(
            InterventionOperation.SUPPRESS, "large language model", None, 8, 0, 3
        ),
        InterventionDraft(
            InterventionOperation.REPLACE,
            "large language model",
            "helpful research assistant",
            8,
            0,
            3,
        ),
    )

    editors, results = runtime.prepare_interventions("prompt", drafts)

    assert [call[0] for call in calls] == ["inject", "suppress", "replace"]
    assert all(result.success for result in results)
    assert len(editors) == 3
    assert applied == ["inject ready", "suppress ready", "replace ready"]


def test_prepare_interventions_isolates_failed_phrase_rule(monkeypatch):
    import jlens

    class EngineDouble:
        def __init__(self, model, lens):
            pass

        def phrase_replace(self, prompt, source, target, **options):
            raise ValueError("phrase has rank zero")

        def phrase_suppress(self, prompt, source, **options):
            return SimpleNamespace(
                success=True,
                message="minimum strength 4",
                trace=SimpleNamespace(target_ids=()),
            )

        def apply(self, result, **_options):
            return nullcontext()

    class Coordinator:
        def exclusive(self, name):
            return nullcontext()

    runtime = HFModelRuntime.__new__(HFModelRuntime)
    runtime.lens = SimpleNamespace(source_layers=(0, 1, 2, 3))
    runtime.lens_model = object()
    runtime.layer_count = 4
    runtime.calibrated = True
    runtime.coordinator = Coordinator()
    runtime._formatted_prompt = lambda prompt, history=(): f"formatted:{prompt}"
    runtime._make_phrase_effect_probe = lambda *_args: (lambda *_probe: (True, 1.0))
    monkeypatch.setattr(jlens, "InterventionEngine", EngineDouble)
    drafts = (
        InterventionDraft(InterventionOperation.REPLACE, "bad phrase", "cat", 8, 0, 3),
        InterventionDraft(InterventionOperation.SUPPRESS, "noise phrase", None, 8, 0, 3),
    )

    editors, results = runtime.prepare_interventions("prompt", drafts)

    assert len(editors) == len(results) == 2
    assert not results[0].success
    assert "rank zero" in results[0].message
    assert results[1].success


def test_generation_applies_intervention_drafts():
    runtime = RuntimeDouble()
    runtime.prepared = []

    def prepare(prompt, drafts, *, history=()):
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

    def prepare(prompt, drafts, *, history=()):
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
