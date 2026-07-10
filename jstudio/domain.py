"""Immutable UI-facing records shared by J Studio views and services."""

from __future__ import annotations

import math
from dataclasses import dataclass, field, replace
from datetime import UTC, datetime
from enum import StrEnum
from typing import Any
from uuid import uuid4


def _new_id(prefix: str) -> str:
    return f"{prefix}-{uuid4().hex}"


def _require_finite(name: str, value: float) -> None:
    if not math.isfinite(value):
        raise ValueError(f"{name} must be finite")


class BackendKind(StrEnum):
    LOCAL = "local"
    REMOTE_WORKER = "remote-worker"
    OFFLINE_TRACE = "offline-trace"


class SessionState(StrEnum):
    DISCONNECTED = "disconnected"
    LOADING = "loading"
    READY = "ready"
    GENERATING = "generating"
    PAUSED = "paused"
    FAILED = "failed"


#: Estimators that produce a dense J_l transport; usable at any fit stage
#: because the sketch-era pass@10 gate does not apply to them.
DENSE_LENS_ESTIMATORS = frozenset(
    {"dense-mean-jacobian-v1", "dense-jacobian-v2", "projected-dense-jacobian-v2"}
)


class LensFitState(StrEnum):
    MISSING = "missing"
    WAITING = "waiting"
    PREVIEW = "preview"
    REFINING = "refining"
    STABLE = "stable"
    FAILED = "failed"


class RunMode(StrEnum):
    BASELINE = "baseline"
    WITH_STACK = "with-stack"


class GenerationBackend(StrEnum):
    FAST = "fast-q4"
    EXACT = "exact-bf16"


class RunState(StrEnum):
    READY = "ready"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETE = "complete"
    CANCELLED = "cancelled"
    FAILED = "failed"


class InterventionOperation(StrEnum):
    INJECT = "inject"
    REPLACE = "replace"
    SUPPRESS = "suppress"


class InterventionState(StrEnum):
    DRAFT = "draft"
    VALID = "valid"
    CONFLICT = "conflict"
    ARMED = "armed"
    QUEUED = "queued"
    APPLIED = "applied"
    FAILED = "failed"
    UNSUPPORTED = "unsupported"


class ActivationSource(StrEnum):
    OBSERVED = "observed"
    INJECTED = "injected"
    REPLACED = "replaced"
    RULE = "rule"


class RuleTrigger(StrEnum):
    JSPACE_FRAME = "jspace.frame"
    BEFORE_TOKEN = "generation.beforeToken"
    AFTER_TOKEN = "generation.afterToken"
    BEFORE_APPLY = "intervention.beforeApply"
    SWEEP_COMPLETE = "sweep.afterComplete"


@dataclass(frozen=True, slots=True)
class SessionCapabilities:
    inspect: bool = True
    generate: bool = True
    intervene: bool = True
    rules: bool = True
    strength_min: float = 0.0
    strength_max: float = 1.0

    def __post_init__(self) -> None:
        _require_finite("strength_min", self.strength_min)
        _require_finite("strength_max", self.strength_max)
        if self.strength_min > self.strength_max:
            raise ValueError("strength_min cannot exceed strength_max")


@dataclass(frozen=True, slots=True)
class ModelSessionSummary:
    session_id: str
    model_id: str
    revision: str
    lens_id: str | None
    layer_count: int
    backend_kind: BackendKind
    state: SessionState
    capabilities: SessionCapabilities = SessionCapabilities()
    display_name: str | None = None
    device: str = "unknown"
    precision: str = "unknown"

    def __post_init__(self) -> None:
        if self.layer_count <= 0:
            raise ValueError("layer_count must be positive")
        if not self.session_id or not self.model_id or not self.revision:
            raise ValueError("session_id, model_id, and revision are required")

    @classmethod
    def offline_trace(cls, name: str, *, layers: int) -> ModelSessionSummary:
        return cls(
            session_id=f"offline:{name}",
            model_id=name,
            revision="captured",
            lens_id="embedded-trace",
            layer_count=layers,
            backend_kind=BackendKind.OFFLINE_TRACE,
            state=SessionState.READY,
            capabilities=SessionCapabilities(
                inspect=True,
                generate=False,
                intervene=False,
                rules=False,
            ),
            display_name=name,
            device="offline",
            precision="captured",
        )


@dataclass(frozen=True, slots=True)
class LensFitStatus:
    state: LensFitState
    stage: str
    completed: int
    total: int
    elapsed_seconds: float = 0.0
    quality: str = "unchecked"
    detail: str = ""

    def __post_init__(self) -> None:
        if self.completed < 0 or self.total < 0 or self.completed > self.total:
            raise ValueError("lens fit progress must satisfy 0 <= completed <= total")
        _require_finite("elapsed_seconds", self.elapsed_seconds)
        if self.elapsed_seconds < 0:
            raise ValueError("elapsed_seconds must be non-negative")


@dataclass(frozen=True, slots=True)
class ConceptActivation:
    term: str
    score: float
    confidence: float | None
    layer: int
    token_index: int
    source: ActivationSource = ActivationSource.OBSERVED
    rank: int | None = None
    previous_score: float | None = None

    def __post_init__(self) -> None:
        if not self.term:
            raise ValueError("term is required")
        _require_finite("score", self.score)
        if self.confidence is not None:
            _require_finite("confidence", self.confidence)
            if not 0 <= self.confidence <= 1:
                raise ValueError("confidence must lie in [0, 1]")
        if self.previous_score is not None:
            _require_finite("previous_score", self.previous_score)
        if self.layer < 0 or self.token_index < 0:
            raise ValueError("layer and token_index must be non-negative")


@dataclass(frozen=True, slots=True)
class InterventionDraft:
    operation: InterventionOperation
    source_term: str | None
    target_term: str | None
    strength: float
    layer_start: int
    layer_end: int
    duration: str = "next-token"
    step_count: int | None = None
    match_mode: str = "exact"
    trigger: str = "manual"

    def __post_init__(self) -> None:
        _require_finite("strength", self.strength)
        if self.strength < 0:
            raise ValueError("strength must be non-negative")
        if self.layer_start < 0 or self.layer_end < self.layer_start:
            raise ValueError("invalid layer range")
        if self.operation is InterventionOperation.INJECT and not self.target_term:
            raise ValueError("inject requires a target term")
        if self.operation is InterventionOperation.REPLACE:
            if not self.source_term:
                raise ValueError("replace requires a source term")
            if not self.target_term:
                raise ValueError("replace requires a target term")
        if self.operation is InterventionOperation.SUPPRESS and not self.source_term:
            raise ValueError("suppress requires a source term")
        if self.duration == "steps" and (self.step_count is None or self.step_count <= 0):
            raise ValueError("steps duration requires a positive step_count")


@dataclass(frozen=True, slots=True)
class InterventionEntry:
    intervention_id: str
    draft: InterventionDraft
    label: str
    enabled: bool = False
    state: InterventionState = InterventionState.DRAFT
    status_detail: str = "Draft"
    applied_run_id: str | None = None

    @classmethod
    def from_draft(cls, draft: InterventionDraft, *, label: str = "") -> InterventionEntry:
        return cls(
            intervention_id=_new_id("intervention"),
            draft=draft,
            label=label or draft.operation.value.title(),
        )


@dataclass(frozen=True, slots=True)
class RunRecord:
    run_id: str
    prompt: str
    mode: RunMode
    state: RunState
    created_at: str
    baseline_run_id: str | None = None
    intervention_ids: tuple[str, ...] = ()
    rule_ids: tuple[str, ...] = ()
    output_text: str = ""
    partial: bool = False
    generation_backend: GenerationBackend = GenerationBackend.EXACT
    quantization: str = "BF16"
    ttft_seconds: float | None = None
    decode_tokens_per_second: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("ttft_seconds", self.ttft_seconds),
            ("decode_tokens_per_second", self.decode_tokens_per_second),
        ):
            if value is not None:
                _require_finite(name, value)
                if value < 0:
                    raise ValueError(f"{name} must be non-negative")

    @classmethod
    def create(cls, *, prompt: str, mode: RunMode) -> RunRecord:
        return cls(
            run_id=_new_id("run"),
            prompt=prompt,
            mode=mode,
            state=RunState.READY,
            created_at=datetime.now(UTC).isoformat(),
        )

    def derive(
        self,
        *,
        mode: RunMode,
        intervention_ids: tuple[str, ...] = (),
        rule_ids: tuple[str, ...] = (),
    ) -> RunRecord:
        baseline_id = self.run_id if self.mode is RunMode.BASELINE else self.baseline_run_id
        return RunRecord(
            run_id=_new_id("run"),
            prompt=self.prompt,
            mode=mode,
            state=RunState.READY,
            created_at=datetime.now(UTC).isoformat(),
            baseline_run_id=baseline_id,
            intervention_ids=tuple(intervention_ids),
            rule_ids=tuple(rule_ids),
        )

    def with_state(self, state: RunState, *, output_text: str | None = None) -> RunRecord:
        return replace(
            self,
            state=state,
            output_text=self.output_text if output_text is None else output_text,
            partial=state is RunState.CANCELLED,
        )


@dataclass(frozen=True, slots=True)
class JLensFrame:
    run_id: str
    sequence: int
    token_index: int
    token_text: str
    layer_count: int
    activations: tuple[ConceptActivation, ...]
    timestamp: str
    interventions_active: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class RuleRecord:
    rule_id: str
    name: str
    source: str
    trigger: RuleTrigger
    priority: int = 100
    enabled: bool = False
    trusted: bool = False
    source_hash: str | None = None
    tested_hash: str | None = None
    consecutive_failures: int = 0
    last_result: str = "Never tested"
    config: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActivityRecord:
    activity_id: str
    title: str
    state: str
    progress: float | None = None
    detail: str = ""
    cancellable: bool = False


@dataclass(frozen=True, slots=True)
class ExperimentRecord:
    experiment_id: str
    name: str
    prompt_ids: tuple[str, ...]
    run_ids: tuple[str, ...] = ()
