# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import json
from dataclasses import replace

import pytest
import torch

from jlens.fitting import fit
from jlens.hooks import ActivationRecorder
from jlens.interventions import (
    ConceptResolver,
    InterventionEngine,
    InterventionTrace,
    PhraseResidualOperator,
    PhraseResidualSchedule,
    SearchPoint,
    downstream_score_covectors,
    local_score_covectors,
    transported_unembedding_covectors,
)
from jlens.lens import JacobianLens

from .tiny import TinyDecoder

PROMPT = "the quick brown fox jumps over the lazy dog " * 2


@pytest.fixture
def tiny_engine():
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[2],
        dim_batch=4,
        max_seq_len=64,
    )
    return InterventionEngine(model, lens)


def test_resolver_prefers_single_token_and_strips_bos():
    model = TinyDecoder()
    resolver = ConceptResolver(model.tokenizer)

    concept = resolver.resolve("a")

    assert concept.token_ids == (8,)
    assert not concept.experimental
    assert all(0 not in variant for variant in concept.variants)


def test_resolver_marks_multitoken_text_experimental():
    model = TinyDecoder()

    concept = ConceptResolver(model.tokenizer).resolve("ab")

    assert concept.token_ids == (8, 9)
    assert concept.experimental
    assert concept.text == "ab"


def test_resolver_accepts_explicit_token_ids():
    model = TinyDecoder()
    resolver = ConceptResolver(model.tokenizer)

    single = resolver.resolve(7)
    multiple = resolver.resolve([7, 8])

    assert single.token_ids == (7,) and not single.experimental
    assert multiple.token_ids == (7, 8) and multiple.experimental


def test_local_covector_matches_finite_difference():
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = JacobianLens({1: torch.randn(8, 8)}, n_prompts=1, d_model=8)
    residual = torch.randn(8)
    token_id = 3

    gradient = local_score_covectors(
        model, lens, residual, layer=1, token_ids=[token_id]
    )[0]
    direction = torch.randn_like(residual)
    epsilon = 1e-3

    def score(value):
        return model.unembed(lens.transport(value, 1))[token_id]

    finite_difference = (
        score(residual + epsilon * direction) - score(residual - epsilon * direction)
    ) / (2 * epsilon)
    torch.testing.assert_close(
        gradient @ direction, finite_difference, atol=2e-3, rtol=2e-2
    )


def test_phrase_covectors_use_raw_unembedding_through_effective_j_transport():
    model = TinyDecoder(n_layers=4, d_model=8)
    jacobian = torch.randn(8, 8)
    lens = JacobianLens(
        {2: jacobian},
        n_prompts=1,
        d_model=8,
        metadata={"transport_shrinkage": "0.75"},
    )
    token_ids = (8, 9)

    actual = transported_unembedding_covectors(model, lens, 2, token_ids)

    effective = torch.eye(8) + 0.75 * (jacobian - torch.eye(8))
    expected = model.lm_head.weight.detach().float()[list(token_ids)] @ effective
    torch.testing.assert_close(actual, expected)


def test_downstream_covector_matches_finite_difference():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompt = PROMPT
    layer = 2
    token_id = 3
    gradient = downstream_score_covectors(
        model, prompt, layer=layer, position=-1, token_ids=[token_id]
    )[0]
    input_ids = model.encode(prompt)
    position = input_ids.shape[1] - 1
    direction = torch.randn_like(gradient)
    epsilon = 1e-3

    def score(delta):
        from jlens.hooks import ActivationEditor, ResidualEdit

        with ActivationEditor(
            model.layers, [ResidualEdit(layer, (position,), delta)]
        ):
            with ActivationRecorder(model.layers, at=[model.n_layers - 1]) as recorder:
                model.forward(input_ids)
        residual = recorder.activations[model.n_layers - 1][0, -1].detach().float()
        return model.unembed(residual)[token_id]

    finite_difference = (
        score(epsilon * direction) - score(-epsilon * direction)
    ) / (2 * epsilon)
    torch.testing.assert_close(
        gradient @ direction, finite_difference, atol=2e-3, rtol=2e-2
    )


def test_successful_injection_must_change_measured_next_token_logits():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    torch.manual_seed(10)
    lens = JacobianLens({2: torch.randn(8, 8) * 2}, n_prompts=1, d_model=8)
    engine = InterventionEngine(model, lens)
    target = 20
    input_ids = model.encode(PROMPT)

    result = engine.inject(
        PROMPT,
        target,
        layers=[2],
        positions=(-1,),
        top_k=5,
        maximum_scale=32.0,
        relative_tolerance=0.05,
    )

    with engine.apply(result):
        with ActivationRecorder(model.layers, at=[model.n_layers - 1]) as recorder:
            model.forward(input_ids)
    logits = model.unembed(
        recorder.activations[model.n_layers - 1][0, -1].detach().float()
    )

    assert result.success, result.message
    assert target in logits.topk(5).indices.tolist()


def test_default_injection_targets_measured_next_token_winner():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    torch.manual_seed(10)
    lens = JacobianLens({2: torch.randn(8, 8) * 2}, n_prompts=1, d_model=8)
    engine = InterventionEngine(model, lens)
    target = 4
    input_ids = model.encode(PROMPT)

    result = engine.inject(
        PROMPT,
        target,
        layers=[2],
        positions=(-1,),
        maximum_scale=32.0,
        relative_tolerance=0.05,
    )

    with engine.apply(result):
        with ActivationRecorder(model.layers, at=[model.n_layers - 1]) as recorder:
            model.forward(input_ids)
    logits = model.unembed(
        recorder.activations[model.n_layers - 1][0, -1].detach().float()
    )

    assert result.success, result.message
    assert int(logits.argmax()) == target


def test_failed_primary_search_stays_failed_and_unapplied():
    model = TinyDecoder(n_layers=4, d_model=8, seed=0)
    torch.manual_seed(10)
    lens = JacobianLens({2: torch.randn(8, 8) * 2}, n_prompts=1, d_model=8)
    engine = InterventionEngine(model, lens)
    target = 20
    input_ids = model.encode(PROMPT)

    result = engine.inject(
        PROMPT,
        target,
        layers=[2],
        positions=(-1,),
        maximum_scale=128.0,
        relative_tolerance=0.05,
    )

    del input_ids

    assert not result.success
    assert "corrective" not in result.trace.warnings
    with pytest.raises(ValueError, match="successful intervention"):
        engine.apply(result)


def test_trace_json_round_trip():
    trace = InterventionTrace(
        operation="inject",
        target_ids=(7,),
        source_ids=(),
        experimental=False,
        selected_layer=2,
        selected_positions=(-1,),
        selected_scale=0.5,
        normalized_cost=0.25,
        baseline_scores={"7": 0.1},
        after_scores={"7": 0.5},
        baseline_top_ids=(1, 2, 3),
        after_top_ids=(7, 1, 2),
        search_points=(SearchPoint(0.5, True, 0.4, 0.2),),
        warnings=(),
    )

    encoded = json.dumps(trace.to_dict())
    decoded = InterventionTrace.from_dict(json.loads(encoded))

    assert decoded == trace

    with pytest.raises(ValueError, match="finite"):
        replace(trace, normalized_cost=float("inf"))


def test_inject_reaches_target_with_calibrated_minimum(tiny_engine):
    baseline = tiny_engine.read(PROMPT, layer=2, position=-1, top_n=32)
    target = baseline.top_ids[12]

    result = tiny_engine.inject(
        PROMPT,
        target,
        layers=[2],
        positions=(-1,),
        top_k=5,
        maximum_scale=32.0,
        relative_tolerance=0.02,
    )

    assert result.success, result.message
    assert target in result.trace.after_top_ids[:5]
    assert result.trace.selected_scale > 0
    assert any(not point.passed for point in result.trace.search_points)


def test_apply_scopes_selected_edit_to_next_forward(tiny_engine):
    baseline = tiny_engine.read(PROMPT, layer=2, position=-1, top_n=32)
    result = tiny_engine.inject(
        PROMPT,
        baseline.top_ids[12],
        layers=[2],
        positions=(-1,),
        top_k=5,
        maximum_scale=32,
    )
    input_ids = tiny_engine.model.encode(PROMPT)

    with ActivationRecorder(tiny_engine.model.layers, at=[2]) as recorder:
        tiny_engine.model.forward(input_ids)
        unedited = recorder.activations[2].detach().clone()
    with tiny_engine.apply(result):
        with ActivationRecorder(tiny_engine.model.layers, at=[2]) as recorder:
            tiny_engine.model.forward(input_ids)
            edited = recorder.activations[2].detach().clone()
        # A cached decode step has length one; one-shot editing must not try to
        # reuse the absolute prefill position.
        tiny_engine.model.forward(input_ids[:, -1:])
    with ActivationRecorder(tiny_engine.model.layers, at=[2]) as recorder:
        tiny_engine.model.forward(input_ids)
        restored = recorder.activations[2].detach().clone()

    assert not torch.equal(edited, unedited)
    torch.testing.assert_close(restored, unedited)


def test_suppress_removes_top_target(tiny_engine):
    baseline = tiny_engine.read(PROMPT, layer=2, position=-1, top_n=32)
    target = baseline.top_ids[0]

    result = tiny_engine.suppress(
        PROMPT,
        target,
        layers=[2],
        positions=(-1,),
        top_k=5,
        maximum_scale=32.0,
    )

    assert result.success, result.message
    assert target not in result.trace.after_top_ids[:5]


def test_replace_target_outranks_source(tiny_engine):
    baseline = tiny_engine.read(PROMPT, layer=2, position=-1, top_n=32)
    source = baseline.top_ids[0]
    target = baseline.top_ids[10]

    result = tiny_engine.replace(
        PROMPT,
        source,
        target,
        layers=[2],
        positions=(-1,),
        preserve_top_k=0,
        maximum_scale=32.0,
    )

    assert result.success, result.message
    assert result.trace.after_scores[str(target)] > result.trace.after_scores[str(source)]


def test_multitoken_request_is_experimental_and_measured(tiny_engine):
    result = tiny_engine.inject(
        PROMPT,
        "ab",
        layers=[2],
        positions=(-1,),
        top_k=16,
        maximum_scale=4.0,
    )

    assert result.trace.experimental
    assert result.trace.sequence_logprob_before is not None
    assert result.trace.sequence_logprob_after is not None


def test_bounded_failure_does_not_hide_excessive_strength(tiny_engine):
    baseline = tiny_engine.read(PROMPT, layer=2, position=-1, top_n=32)
    target = baseline.top_ids[-1]

    result = tiny_engine.inject(
        PROMPT,
        target,
        layers=[2],
        positions=(-1,),
        top_k=1,
        maximum_scale=1e-5,
    )

    assert not result.success
    assert result.trace.selected_scale <= 1e-5
    assert "bounded" in result.message


def test_multi_layer_search_selects_lowest_cost_success():
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[1, 2],
        dim_batch=4,
        max_seq_len=64,
    )
    engine = InterventionEngine(model, lens)
    target = engine.read(PROMPT, layer=2, position=-1, top_n=32).top_ids[12]
    options = [
        engine.inject(PROMPT, target, layers=[layer], top_k=8, maximum_scale=32)
        for layer in (1, 2)
    ]
    passing = [result for result in options if result.success]
    assert len(passing) == 2

    combined = engine.inject(PROMPT, target, layers=[1, 2], top_k=8, maximum_scale=32)

    assert combined.success
    assert combined.trace.normalized_cost == min(
        result.trace.normalized_cost for result in passing
    )


def test_phrase_operator_suppression_saturates_without_inversion():
    operator = PhraseResidualOperator(
        operation="suppress",
        source_basis=torch.tensor([[1.0], [0.0]]),
        target_directions=None,
        alignment=None,
        scale=4.0,
    )

    edited = operator.apply(torch.tensor([2.0, 3.0]))

    torch.testing.assert_close(edited, torch.tensor([0.0, 3.0]))


def test_phrase_schedule_injects_multitoken_directions_in_order():
    operator = PhraseResidualOperator(
        operation="inject",
        source_basis=None,
        target_directions=torch.eye(2),
        alignment=None,
        scale=1.0,
    )
    schedule = PhraseResidualSchedule(operator)
    residual = torch.tensor([1.0, 1.0])

    first = schedule(residual)
    second = schedule(residual)

    assert first[0] > first[1]
    assert second[1] > second[0]
    assert schedule.step == 2


def test_multitoken_phrase_suppress_uses_all_source_tokens(tiny_engine):
    result = tiny_engine.phrase_suppress(
        PROMPT, "ab", layers=[2], maximum_scale=16.0
    )

    assert result.success, result.message
    assert result.trace.source_ids == (8, 9)
    assert result.trace.experimental
    assert 0 < result.trace.selected_scale <= 16
    assert result.operator is not None
    assert result.trace.after_scores["source_energy"] < result.trace.baseline_scores[
        "source_energy"
    ]

    input_ids = tiny_engine.model.encode(PROMPT)
    with tiny_engine.apply(result):
        tiny_engine.model.forward(input_ids)
        tiny_engine.model.forward(input_ids[:, -1:])


def test_unequal_multitoken_phrase_replace_builds_measured_operator(tiny_engine):
    result = tiny_engine.phrase_replace(
        PROMPT,
        "ab",
        "cdef",
        layers=[2],
        maximum_scale=16.0,
    )

    assert result.success, result.message
    assert result.trace.source_ids == (8, 9)
    assert result.trace.target_ids == (10, 11, 12, 13)
    assert result.operator is not None
    assert result.operator.alignment.shape[0] == 4
    assert result.trace.after_scores["target_energy"] > result.trace.baseline_scores[
        "target_energy"
    ]


def test_multitoken_phrase_inject_uses_target_centroid(tiny_engine):
    result = tiny_engine.phrase_inject(
        PROMPT, "abcd", layers=[2], maximum_scale=16.0
    )

    assert result.success, result.message
    assert result.trace.target_ids == (8, 9, 10, 11)
    assert result.trace.experimental
    assert result.trace.baseline_scores["residual_rms"] > 0
    assert result.trace.after_scores["target_energy"] > result.trace.baseline_scores[
        "target_energy"
    ]


def test_phrase_search_uses_first_causally_effective_strength(
    tiny_engine, monkeypatch
):
    observed_scales = []
    monkeypatch.setattr(tiny_engine, "_phrase_passes", lambda *_args: True)

    def probe(operator_pairs, positions):
        assert positions == (-1,)
        scale = operator_pairs[0][1].scale
        observed_scales.append(scale)
        return scale >= 1.0, scale

    result = tiny_engine.phrase_inject(
        PROMPT,
        "abcd",
        layers=[2],
        maximum_scale=16.0,
        effect_probe=probe,
    )

    assert result.success, result.message
    assert result.trace.selected_scale == 1.0
    assert result.operator is not None and result.operator.scale == 1.0
    assert observed_scales == [1 / 16, 1 / 8, 1 / 4, 1 / 2, 3 / 4, 1]
    assert result.trace.search_points[-1].downstream_shift == 1.0
    assert "generation-causal-probe" in result.trace.warnings


def test_phrase_interventions_do_not_use_downstream_logit_solver(
    tiny_engine, monkeypatch
):
    def forbidden(*args, **kwargs):
        raise AssertionError("downstream logit solver must not be used")

    monkeypatch.setattr("jlens.interventions.downstream_score_covectors", forbidden)

    result = tiny_engine.phrase_replace(
        PROMPT, "ab", "cd", layers=[2], maximum_scale=16.0
    )

    assert result.success, result.message


def test_phrase_transform_applies_across_selected_layer_range():
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = fit(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[1, 2],
        dim_batch=4,
        max_seq_len=64,
    )
    engine = InterventionEngine(model, lens)

    result = engine.phrase_suppress(
        PROMPT, "ab", layers=[1, 2], maximum_scale=16.0
    )

    assert result.success, result.message
    assert {layer for layer, _operator in result.operators} == {1, 2}
    with engine.apply(result):
        model.forward(model.encode(PROMPT))
    assert all(not layer._forward_hooks for layer in model.layers)


def test_phrase_apply_once_does_not_reinject_on_later_decode_steps(tiny_engine):
    result = tiny_engine.phrase_inject(
        PROMPT, "abcd", layers=[2], maximum_scale=16.0
    )
    input_ids = tiny_engine.model.encode(PROMPT)

    with ActivationRecorder(tiny_engine.model.layers, at=[2]) as recorder:
        tiny_engine.model.forward(input_ids[:, -1:])
        baseline = recorder.activations[2].detach().clone()
    with tiny_engine.apply(result, once=True):
        with ActivationRecorder(tiny_engine.model.layers, at=[2]) as recorder:
            tiny_engine.model.forward(input_ids[:, -1:])
            first = recorder.activations[2].detach().clone()
            tiny_engine.model.forward(input_ids[:, -1:])
            second = recorder.activations[2].detach().clone()

    assert not torch.equal(first, baseline)
    torch.testing.assert_close(second, baseline)
