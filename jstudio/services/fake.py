"""Deterministic non-Qt services for the demo UI and tests."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, replace
from datetime import UTC, datetime

from jstudio.domain import (
    ActivationSource,
    BackendKind,
    ConceptActivation,
    JLensFrame,
    LensFitState,
    LensFitStatus,
    ModelSessionSummary,
    RunRecord,
    RunState,
    SessionState,
)
from jstudio.services.protocols import (
    GenerationRequest,
    JStudioServices,
    SlicePage,
    SliceRequest,
)


class FakeSessionService:
    def __init__(self) -> None:
        self._sessions = (
            ModelSessionSummary(
                session_id="local:qwen-27b",
                model_id="Qwen/Qwen3.6-27B",
                revision="main@7f2c9d1",
                lens_id="qwen3.6-27b/jlens",
                layer_count=64,
                backend_kind=BackendKind.LOCAL,
                state=SessionState.READY,
                display_name="Qwen3.6-27B",
                device="CUDA 0",
                precision="BF16",
            ),
            ModelSessionSummary.offline_trace("ASCII face trace", layers=64),
        )

    def list_sessions(self) -> tuple[ModelSessionSummary, ...]:
        return self._sessions

    def open_session(self, session_id: str) -> ModelSessionSummary:
        for session in self._sessions:
            if session.session_id == session_id:
                return session
        raise KeyError(session_id)

    def refresh(self) -> tuple[ModelSessionSummary, ...]:
        return self._sessions


class FakeLensService:
    def __init__(self) -> None:
        self._frames: dict[str, list[JLensFrame]] = {}
        self._lock = threading.Lock()
        self._slice_generation = 0
        self._fit_status = LensFitStatus(
            LensFitState.STABLE, "Stable", 32, 32, quality="passed"
        )
        self._fit_subscribers: list[Callable[[LensFitStatus], None]] = []
        self.fit_requests = 0

    def record(self, frame: JLensFrame) -> None:
        with self._lock:
            self._frames.setdefault(frame.run_id, []).append(frame)

    def current_activations(self, run_id: str) -> tuple[ConceptActivation, ...]:
        frames = self.frames(run_id)
        return frames[-1].activations if frames else ()

    def frames(self, run_id: str) -> tuple[JLensFrame, ...]:
        with self._lock:
            return tuple(self._frames.get(run_id, ()))

    def request_slice(self, request: SliceRequest) -> Future[SlicePage]:
        with self._lock:
            self._slice_generation += 1
            generation = self._slice_generation
        future: Future[SlicePage] = Future()
        future.set_result(
            SlicePage(
                request.run_id,
                generation,
                f"<html><body><h1>J-lens — {request.title}</h1>"
                f"<pre>{request.text}</pre></body></html>",
            )
        )
        return future

    def fit_status(self) -> LensFitStatus:
        with self._lock:
            return self._fit_status

    def subscribe_fit(
        self, callback: Callable[[LensFitStatus], None]
    ) -> Callable[[], None]:
        with self._lock:
            self._fit_subscribers.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._fit_subscribers:
                    self._fit_subscribers.remove(callback)

        return unsubscribe

    def _publish_fit(self, status: LensFitStatus) -> None:
        with self._lock:
            self._fit_status = status
            subscribers = tuple(self._fit_subscribers)
        for subscriber in subscribers:
            subscriber(status)

    def start_fit(self) -> None:
        self.fit_requests += 1
        for status in (
            LensFitStatus(LensFitState.PREVIEW, "Preview", 8, 8),
            LensFitStatus(LensFitState.REFINING, "Stable", 8, 32),
            LensFitStatus(
                LensFitState.STABLE, "Stable", 32, 32, quality="passed"
            ),
        ):
            self._publish_fit(status)

    def publish_fit(
        self,
        state: LensFitState,
        completed: int,
        total: int,
        quality: str = "unchecked",
        detail: str = "",
    ) -> None:
        self._publish_fit(
            LensFitStatus(
                state,
                state.value.title(),
                completed,
                total,
                quality=quality,
                detail=detail,
            )
        )

    def cancel_fit(self) -> None:
        current = self.fit_status()
        self._publish_fit(
            LensFitStatus(
                LensFitState.WAITING,
                current.stage,
                current.completed,
                current.total,
                current.elapsed_seconds,
                current.quality,
                "Fitting paused",
            )
        )


@dataclass
class _Control:
    condition: threading.Condition
    paused: bool = False
    stopped: bool = False
    step_budget: int = 0


class FakeGenerationService:
    RESPONSE = (
        "The search results contain a prompt injection attempt. I will ignore the "
        "embedded instruction and summarize only the relevant information."
    )
    TERMS = (
        "injection",
        "malicious",
        "ignore",
        "untrusted",
        "instruction",
        "comply",
        "warning",
        "trusted",
    )

    def __init__(self, lens: FakeLensService, *, token_delay: float = 0.03) -> None:
        self._lens = lens
        self._token_delay = token_delay
        self.last_request: GenerationRequest | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=2, thread_name_prefix="jstudio-fake"
        )
        self._controls: dict[str, _Control] = {}
        self._lock = threading.Lock()

    def start(self, request: GenerationRequest, sink) -> str:
        self.last_request = request
        run = RunRecord.create(prompt=request.prompt, mode=request.mode)
        run = replace(
            run,
            intervention_ids=request.intervention_ids,
            rule_ids=request.rule_ids,
            state=RunState.RUNNING,
        )
        control = _Control(threading.Condition())
        with self._lock:
            self._controls[run.run_id] = control
        self._executor.submit(self._stream, run, sink, control)
        return run.run_id

    def _stream(self, run: RunRecord, sink, control: _Control) -> None:
        try:
            sink.on_started(run)
            output: list[str] = []
            for sequence, token in enumerate(self.RESPONSE.split()):
                with control.condition:
                    while (
                        control.paused and control.step_budget == 0 and not control.stopped
                    ):
                        control.condition.wait(timeout=0.2)
                    if control.stopped:
                        break
                    if control.paused and control.step_budget:
                        control.step_budget -= 1
                if self._token_delay:
                    time.sleep(self._token_delay)
                output.append(token)
                output_text = " ".join(output)
                sink.on_token(run.run_id, token, output_text)
                frame = self._frame(run, sequence, token)
                self._lens.record(frame)
                sink.on_frame(frame)
            state = RunState.CANCELLED if control.stopped else RunState.COMPLETE
            sink.on_finished(run.with_state(state, output_text=" ".join(output)))
        except Exception as exc:
            sink.on_error(run.run_id, "Fake generation failed", repr(exc))
        finally:
            with self._lock:
                self._controls.pop(run.run_id, None)

    def _frame(self, run: RunRecord, sequence: int, token: str) -> JLensFrame:
        activations = []
        for rank, term in enumerate(self.TERMS):
            sign = -1 if term == "comply" else 1
            score = sign * max(0.12, 0.92 - rank * 0.075 + (sequence % 4) * 0.01)
            activations.append(
                ConceptActivation(
                    term=term,
                    score=score,
                    previous_score=score - 0.06,
                    confidence=max(0.35, 0.96 - rank * 0.07),
                    layer=38 + ((sequence + rank * 3) % 18),
                    token_index=sequence,
                    rank=rank,
                    source=ActivationSource.OBSERVED,
                )
            )
        return JLensFrame(
            run_id=run.run_id,
            sequence=sequence,
            token_index=sequence,
            token_text=token,
            layer_count=64,
            activations=tuple(activations),
            timestamp=datetime.now(UTC).isoformat(),
            interventions_active=run.intervention_ids,
        )

    def _control(self, run_id: str) -> _Control:
        with self._lock:
            control = self._controls.get(run_id)
        if control is None:
            raise KeyError(run_id)
        return control

    def pause(self, run_id: str) -> None:
        control = self._control(run_id)
        with control.condition:
            control.paused = True

    def resume(self, run_id: str) -> None:
        control = self._control(run_id)
        with control.condition:
            control.paused = False
            control.condition.notify_all()

    def next_token(self, run_id: str) -> None:
        control = self._control(run_id)
        with control.condition:
            control.paused = True
            control.step_budget += 1
            control.condition.notify_all()

    def stop(self, run_id: str) -> None:
        control = self._control(run_id)
        with control.condition:
            control.stopped = True
            control.condition.notify_all()

    def close(self) -> None:
        with self._lock:
            controls = list(self._controls.values())
        for control in controls:
            with control.condition:
                control.stopped = True
                control.condition.notify_all()
        self._executor.shutdown(wait=True, cancel_futures=True)


class FakeInterventionService:
    def preview(self, session_id, draft):
        return True, (
            f"{draft.operation.value.title()} is compatible with {session_id}; "
            f"layers {draft.layer_start}–{draft.layer_end}."
        )


class _UnavailableRules:
    available = False
    unavailable_reason = "Rules sandbox has not been initialized"

    def evaluate(self, request, *, cancel_event=None):
        raise RuntimeError(self.unavailable_reason)


def create_fake_services(*, token_delay: float = 0.03) -> JStudioServices:
    from jstudio.rules.sandbox import QuickJSSandbox

    lens = FakeLensService()
    return JStudioServices(
        sessions=FakeSessionService(),
        generation=FakeGenerationService(lens, token_delay=token_delay),
        lens=lens,
        interventions=FakeInterventionService(),
        rules=QuickJSSandbox(),
    )
