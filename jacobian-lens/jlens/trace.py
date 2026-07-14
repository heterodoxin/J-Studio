# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Serializable causal trace artifacts."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


@dataclass(frozen=True)
class DosePoint:
    strength: float
    predicted_effect: float
    observed_effect: float
    norm: float


@dataclass(frozen=True)
class ControlResult:
    name: str
    observed_effect: float
    norm: float
    seed: int | None = None


@dataclass(frozen=True)
class CausalTrace:
    model_id: str
    prompt: str
    tokenization: tuple[int, ...]
    operator_id: str
    layer: int
    position: int
    seed: int
    strength: float
    target: dict[str, Any]
    protected: list[dict[str, Any]]
    dose_response: tuple[DosePoint, ...] = field(default_factory=tuple)
    controls: tuple[ControlResult, ...] = field(default_factory=tuple)
    evidence: str = "intervention-validated"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> CausalTrace:
        data = dict(value)
        data["tokenization"] = tuple(data["tokenization"])
        data["dose_response"] = tuple(
            point if isinstance(point, DosePoint) else DosePoint(**point)
            for point in data.get("dose_response", ())
        )
        data["controls"] = tuple(
            control if isinstance(control, ControlResult) else ControlResult(**control)
            for control in data.get("controls", ())
        )
        return cls(**data)
