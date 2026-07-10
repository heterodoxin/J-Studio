from dataclasses import FrozenInstanceError, replace

import pytest

from jstudio.domain import (
    BackendKind,
    ConceptActivation,
    GenerationBackend,
    InterventionDraft,
    InterventionOperation,
    ModelSessionSummary,
    RunMode,
    RunRecord,
    SessionState,
)


def test_run_records_generation_provenance():
    run = replace(
        RunRecord.create(prompt="hello", mode=RunMode.BASELINE),
        generation_backend=GenerationBackend.FAST,
        quantization="Q4_K_M",
        ttft_seconds=0.2,
        decode_tokens_per_second=72.5,
    )
    assert run.decode_tokens_per_second == 72.5


def test_run_records_are_immutable_and_modes_are_distinct():
    baseline = RunRecord.create(prompt="hello", mode=RunMode.BASELINE)
    controlled = baseline.derive(mode=RunMode.WITH_STACK, intervention_ids=("i1",))

    assert baseline.run_id != controlled.run_id
    assert baseline.intervention_ids == ()
    assert controlled.baseline_run_id == baseline.run_id
    with pytest.raises(FrozenInstanceError):
        baseline.prompt = "changed"


def test_offline_session_capabilities_disable_mutation():
    session = ModelSessionSummary.offline_trace("trace-1", layers=64)

    assert session.backend_kind is BackendKind.OFFLINE_TRACE
    assert session.capabilities.inspect
    assert not session.capabilities.generate
    assert not session.capabilities.intervene


def test_concept_activation_rejects_nonfinite_score():
    with pytest.raises(ValueError, match="finite"):
        ConceptActivation(
            term="unsafe",
            score=float("inf"),
            confidence=0.8,
            layer=3,
            token_index=2,
        )


def test_intervention_draft_validates_operation_fields():
    with pytest.raises(ValueError, match="target"):
        InterventionDraft(
            operation=InterventionOperation.INJECT,
            source_term=None,
            target_term="",
            strength=0.4,
            layer_start=0,
            layer_end=8,
        )
    with pytest.raises(ValueError, match="source"):
        InterventionDraft(
            operation=InterventionOperation.REPLACE,
            source_term=None,
            target_term="trusted",
            strength=0.8,
            layer_start=2,
            layer_end=5,
        )


def test_model_session_summary_requires_valid_layer_count():
    with pytest.raises(ValueError, match="layer_count"):
        ModelSessionSummary(
            session_id="bad",
            model_id="model",
            revision="main",
            lens_id=None,
            layer_count=0,
            backend_kind=BackendKind.LOCAL,
            state=SessionState.READY,
        )
