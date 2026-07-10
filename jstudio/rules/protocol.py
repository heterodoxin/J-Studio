"""Bounded JSON protocol records for rule evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass(frozen=True, slots=True)
class SandboxLimits:
    wall_time_ms: int = 50
    execution_time_ms: int = 25
    heap_bytes: int = 16 * 1024 * 1024
    address_space_bytes: int = 256 * 1024 * 1024
    stack_bytes: int = 512 * 1024
    max_source_bytes: int = 128 * 1024
    max_input_bytes: int = 512 * 1024
    max_output_bytes: int = 256 * 1024
    max_actions: int = 32
    max_logs: int = 100
    max_log_bytes: int = 8 * 1024
    max_results: int = 100

    def __post_init__(self) -> None:
        for name in self.__dataclass_fields__:
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class RuleEvaluationRequest:
    source: str
    trigger: str
    context: dict[str, Any]
    layer_count: int
    limits: SandboxLimits = field(default_factory=SandboxLimits)


@dataclass(frozen=True, slots=True)
class ValidatedRuleAction:
    kind: str
    payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class RejectedRuleAction:
    index: int
    reason: str


@dataclass(frozen=True, slots=True)
class RuleMetrics:
    wall_ms: float = 0.0
    execution_ms: float = 0.0
    peak_worker_bytes: int = 0
    input_bytes: int = 0
    output_bytes: int = 0
    log_bytes: int = 0


@dataclass(frozen=True, slots=True)
class RuleEvaluationResult:
    success: bool
    actions: tuple[ValidatedRuleAction, ...] = ()
    rejected: tuple[RejectedRuleAction, ...] = ()
    raw_json: str = "[]"
    error: str = ""
    metrics: RuleMetrics = field(default_factory=RuleMetrics)
