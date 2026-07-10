"""Backend-neutral service interfaces. This module intentionally has no Qt imports."""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import Future
from dataclasses import dataclass
from typing import Protocol

from jstudio.domain import (
    ConceptActivation,
    GenerationBackend,
    InterventionDraft,
    JLensFrame,
    LensFitStatus,
    ModelSessionSummary,
    RuleRecord,
    RunMode,
    RunRecord,
)


@dataclass(frozen=True, slots=True)
class ReadConfiguration:
    layers: tuple[int, ...] = ()
    max_concepts: int = 200
    max_new_tokens: int = 2048

    def __post_init__(self) -> None:
        if self.max_concepts <= 0 or self.max_new_tokens <= 0:
            raise ValueError("read limits must be positive")
        if any(layer < 0 for layer in self.layers):
            raise ValueError("layers must be non-negative")


@dataclass(frozen=True, slots=True)
class GenerationRequest:
    session_id: str
    prompt: str
    mode: RunMode = RunMode.BASELINE
    intervention_ids: tuple[str, ...] = ()
    intervention_drafts: tuple[InterventionDraft, ...] = ()
    rule_ids: tuple[str, ...] = ()
    rule_records: tuple[RuleRecord, ...] = ()
    read: ReadConfiguration = ReadConfiguration()
    backend: GenerationBackend = GenerationBackend.FAST

    def __post_init__(self) -> None:
        if self.intervention_drafts and (
            len(self.intervention_ids) != len(self.intervention_drafts)
        ):
            raise ValueError("intervention IDs and drafts must have equal length")


@dataclass(frozen=True, slots=True)
class SliceRequest:
    run_id: str
    text: str
    title: str
    layer_stride: int = 1
    last_n_tokens: int | None = None
    top_n: int = 10
    mask_display: bool = True

    def __post_init__(self) -> None:
        if not self.run_id or not self.text or not self.title:
            raise ValueError("slice run_id, text, and title are required")
        if self.layer_stride <= 0 or self.top_n <= 0:
            raise ValueError("slice layer_stride and top_n must be positive")
        if self.last_n_tokens is not None and self.last_n_tokens <= 0:
            raise ValueError("slice last_n_tokens must be positive when set")


@dataclass(frozen=True, slots=True)
class SlicePage:
    run_id: str
    generation: int
    html: str


class GenerationEventSink(Protocol):
    def on_started(self, run: RunRecord) -> None: ...

    def on_token(self, run_id: str, token: str, output_text: str) -> None: ...

    def on_frame(self, frame: JLensFrame) -> None: ...

    def on_intervention(self, intervention_id: str, state: str, detail: str) -> None: ...

    def on_finished(self, run: RunRecord) -> None: ...

    def on_error(self, run_id: str, message: str, detail: str = "") -> None: ...


class SessionService(Protocol):
    def list_sessions(self) -> tuple[ModelSessionSummary, ...]: ...

    def open_session(self, session_id: str) -> ModelSessionSummary: ...

    def refresh(self) -> tuple[ModelSessionSummary, ...]: ...


class GenerationService(Protocol):
    def start(self, request: GenerationRequest, sink: GenerationEventSink) -> str: ...

    def pause(self, run_id: str) -> None: ...

    def resume(self, run_id: str) -> None: ...

    def next_token(self, run_id: str) -> None: ...

    def stop(self, run_id: str) -> None: ...

    def close(self) -> None: ...


class LensService(Protocol):
    def current_activations(self, run_id: str) -> tuple[ConceptActivation, ...]: ...

    def frames(self, run_id: str) -> tuple[JLensFrame, ...]: ...

    def request_slice(self, request: SliceRequest) -> Future[SlicePage]: ...

    def fit_status(self) -> LensFitStatus: ...

    def start_fit(self) -> None: ...

    def cancel_fit(self) -> None: ...

    def subscribe_fit(
        self, callback: Callable[[LensFitStatus], None]
    ) -> Callable[[], None]: ...


class InterventionService(Protocol):
    def preview(self, session_id: str, draft: InterventionDraft) -> tuple[bool, str]: ...


class RuleSandboxProtocol(Protocol):
    available: bool
    unavailable_reason: str | None

    def evaluate(self, request, *, cancel_event=None): ...


@dataclass(frozen=True, slots=True)
class JStudioServices:
    sessions: SessionService
    generation: GenerationService
    lens: LensService
    interventions: InterventionService
    rules: RuleSandboxProtocol
