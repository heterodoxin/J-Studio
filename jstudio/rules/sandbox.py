"""Parent-side one-shot QuickJS process isolation."""

from __future__ import annotations

import importlib.util
import json
import multiprocessing
import struct
import time
from dataclasses import asdict

from jstudio.rules.protocol import (
    RuleEvaluationRequest,
    RuleEvaluationResult,
    RuleMetrics,
)
from jstudio.rules.validation import validate_actions, validate_rule_source
from jstudio.rules.worker import worker_main


def _encode(value: dict) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _decode(message: bytes, maximum: int) -> dict:
    if len(message) < 4:
        raise ValueError("malformed response")
    size = struct.unpack(">I", message[:4])[0]
    if size != len(message) - 4 or size > maximum:
        raise ValueError("malformed or oversized response")
    value = json.loads(message[4:].decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("response must be an object")
    return value


class QuickJSSandbox:
    def __init__(self, *, startup_timeout_ms: int = 5000) -> None:
        self.available = importlib.util.find_spec("quickjs") is not None
        self.unavailable_reason = (
            None if self.available else "Install the pip package 'quickjs' to test rules"
        )
        self.startup_timeout_ms = startup_timeout_ms

    @staticmethod
    def _failure(
        error: str, *, wall_ms: float = 0.0, input_bytes: int = 0
    ) -> RuleEvaluationResult:
        return RuleEvaluationResult(
            success=False,
            error=error,
            metrics=RuleMetrics(wall_ms=wall_ms, input_bytes=input_bytes),
        )

    def evaluate(
        self, request: RuleEvaluationRequest, *, cancel_event=None
    ) -> RuleEvaluationResult:
        started = time.perf_counter()
        if cancel_event is not None and cancel_event.is_set():
            return self._failure("Rule evaluation cancelled")
        if not self.available:
            return self._failure(self.unavailable_reason or "QuickJS unavailable")
        source_validation = validate_rule_source(
            request.source, max_bytes=request.limits.max_source_bytes
        )
        if not source_validation.valid:
            return self._failure(
                "Forbidden or invalid rule source: " + "; ".join(source_validation.problems)
            )
        try:
            payload = {
                "source": request.source,
                "trigger": request.trigger,
                "context": request.context,
                "limits": asdict(request.limits),
            }
            message = _encode(payload)
        except (TypeError, ValueError) as exc:
            return self._failure(f"Rule input is not strict JSON: {exc}")
        input_bytes = len(message) - 4
        if input_bytes > request.limits.max_input_bytes:
            return self._failure(
                f"Rule input exceeds {request.limits.max_input_bytes} bytes",
                input_bytes=input_bytes,
            )

        context = multiprocessing.get_context("spawn")
        request_receiver, request_sender = context.Pipe(duplex=False)
        response_receiver, response_sender = context.Pipe(duplex=False)
        process = context.Process(
            target=worker_main,
            args=(request_receiver, response_sender),
            name="jstudio-rule-worker",
            daemon=True,
        )
        process.start()
        request_receiver.close()
        response_sender.close()
        try:
            request_sender.send_bytes(message)
            request_sender.close()
            startup_timeout_s = max(0.1, self.startup_timeout_ms / 1000.0)
            startup_deadline = time.perf_counter() + startup_timeout_s
            ready = None
            while time.perf_counter() < startup_deadline:
                if cancel_event is not None and cancel_event.is_set():
                    process.terminate()
                    process.join(timeout=0.2)
                    return self._failure(
                        "Rule evaluation cancelled",
                        wall_ms=(time.perf_counter() - started) * 1000,
                        input_bytes=input_bytes,
                    )
                if response_receiver.poll(0.002):
                    ready = _decode(response_receiver.recv_bytes(), 1024)
                    break
            if ready != {"ready": True}:
                process.terminate()
                process.join(timeout=0.2)
                wall_ms = (time.perf_counter() - started) * 1000
                return self._failure(
                    "Rule worker failed to start within "
                    f"{self.startup_timeout_ms} ms (elapsed {wall_ms:.0f} ms)",
                    wall_ms=wall_ms,
                    input_bytes=input_bytes,
                )
            deadline = time.perf_counter() + request.limits.wall_time_ms / 1000.0
            response = None
            while time.perf_counter() < deadline:
                if cancel_event is not None and cancel_event.is_set():
                    process.terminate()
                    process.join(timeout=0.2)
                    return self._failure(
                        "Rule evaluation cancelled",
                        wall_ms=(time.perf_counter() - started) * 1000,
                        input_bytes=input_bytes,
                    )
                if response_receiver.poll(0.002):
                    response = _decode(
                        response_receiver.recv_bytes(),
                        request.limits.max_output_bytes + 64 * 1024,
                    )
                    break
            if response is None:
                process.terminate()
                process.join(timeout=0.2)
                return self._failure(
                    f"Rule exceeded wall time limit of {request.limits.wall_time_ms} ms",
                    wall_ms=(time.perf_counter() - started) * 1000,
                    input_bytes=input_bytes,
                )
            process.join(timeout=0.2)
            wall_ms = (time.perf_counter() - started) * 1000
            if not response.get("success"):
                return self._failure(
                    response.get("error", "Rule worker failed"),
                    wall_ms=wall_ms,
                    input_bytes=input_bytes,
                )
            raw_json = response.get("raw_json", "[]")
            try:
                raw_actions = json.loads(raw_json)
            except json.JSONDecodeError as exc:
                return self._failure(
                    f"Rule returned malformed JSON: {exc}",
                    wall_ms=wall_ms,
                    input_bytes=input_bytes,
                )
            validation = validate_actions(
                raw_actions,
                layer_count=request.layer_count,
                max_actions=request.limits.max_actions,
            )
            if validation.rejected:
                reasons = "; ".join(item.reason for item in validation.rejected)
                return RuleEvaluationResult(
                    success=False,
                    rejected=validation.rejected,
                    raw_json=raw_json,
                    error=reasons,
                    metrics=RuleMetrics(
                        wall_ms=wall_ms,
                        execution_ms=float(response.get("execution_ms", 0.0)),
                        peak_worker_bytes=int(response.get("peak_worker_bytes", 0)),
                        input_bytes=input_bytes,
                        output_bytes=int(response.get("output_bytes", 0)),
                    ),
                )
            log_bytes = sum(
                len(str(action.payload.get("message", "")).encode("utf-8"))
                for action in validation.validated
                if action.kind == "log"
            )
            log_count = sum(action.kind == "log" for action in validation.validated)
            if (
                log_count > request.limits.max_logs
                or log_bytes > request.limits.max_log_bytes
            ):
                return self._failure(
                    "Rule log limit exceeded",
                    wall_ms=wall_ms,
                    input_bytes=input_bytes,
                )
            return RuleEvaluationResult(
                success=True,
                actions=validation.validated,
                raw_json=raw_json,
                metrics=RuleMetrics(
                    wall_ms=wall_ms,
                    execution_ms=float(response.get("execution_ms", 0.0)),
                    peak_worker_bytes=int(response.get("peak_worker_bytes", 0)),
                    input_bytes=input_bytes,
                    output_bytes=int(response.get("output_bytes", 0)),
                    log_bytes=log_bytes,
                ),
            )
        except (EOFError, OSError, ValueError) as exc:
            if process.is_alive():
                process.terminate()
            process.join(timeout=0.2)
            return self._failure(
                f"Rule worker protocol failure: {exc}",
                wall_ms=(time.perf_counter() - started) * 1000,
                input_bytes=input_bytes,
            )
        finally:
            response_receiver.close()
