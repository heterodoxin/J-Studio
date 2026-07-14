# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Covariance-aware geometry for minimum-disturbance interventions."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class CovarianceMetric:
    """A positive-semidefinite covariance represented as diagonal plus low rank.

    ``factors`` has shape ``[rank, d_model]`` and represents ``F.T @ F``.
    Intervention cost is measured in the inverse of this covariance, while the
    constrained solver only needs efficient covariance-vector products.
    """

    diagonal: torch.Tensor
    factors: torch.Tensor | None = None
    calibrated: bool = True

    def __post_init__(self) -> None:
        if self.diagonal.ndim != 1 or self.diagonal.numel() == 0:
            raise ValueError("diagonal must be a non-empty 1D tensor")
        if not torch.isfinite(self.diagonal).all() or (self.diagonal < 0).any():
            raise ValueError("diagonal must be finite and non-negative")
        if self.factors is not None:
            if self.factors.ndim != 2 or self.factors.shape[1] != len(self.diagonal):
                raise ValueError("factors must have shape [rank, d_model]")
            if not torch.isfinite(self.factors).all():
                raise ValueError("factors must be finite")
        if self.diagonal.max() == 0 and (
            self.factors is None or self.factors.shape[0] < len(self.diagonal)
        ):
            raise ValueError("covariance representation must have full support")

    @classmethod
    def identity(
        cls,
        d_model: int,
        *,
        device: torch.device | str | None = None,
        dtype: torch.dtype = torch.float32,
        calibrated: bool = False,
    ) -> CovarianceMetric:
        if d_model <= 0:
            raise ValueError("d_model must be positive")
        return cls(
            diagonal=torch.ones(d_model, device=device, dtype=dtype),
            calibrated=calibrated,
        )

    @classmethod
    def from_dense(
        cls, covariance: torch.Tensor, *, calibrated: bool = True
    ) -> CovarianceMetric:
        if (
            covariance.ndim != 2
            or covariance.shape[0] != covariance.shape[1]
            or covariance.numel() == 0
        ):
            raise ValueError("covariance must be a non-empty square matrix")
        if not torch.isfinite(covariance).all():
            raise ValueError("covariance must be finite")
        if not torch.allclose(covariance, covariance.T, atol=1e-7, rtol=1e-6):
            raise ValueError("covariance must be symmetric")
        factor = torch.linalg.cholesky(covariance).T
        return cls(
            diagonal=torch.zeros(
                covariance.shape[0],
                device=covariance.device,
                dtype=covariance.dtype,
            ),
            factors=factor,
            calibrated=calibrated,
        )

    @property
    def d_model(self) -> int:
        return len(self.diagonal)

    def apply(self, value: torch.Tensor) -> torch.Tensor:
        """Right-multiply one or more row vectors by the covariance."""
        if value.shape[-1] != self.d_model:
            raise ValueError(
                f"value width {value.shape[-1]} does not match d_model={self.d_model}"
            )
        diagonal = self.diagonal.to(device=value.device, dtype=value.dtype)
        result = value * diagonal
        if self.factors is not None and self.factors.numel():
            factors = self.factors.to(device=value.device, dtype=value.dtype)
            result = result + (value @ factors.T) @ factors
        return result


@dataclass(frozen=True)
class ProjectionSolution:
    delta: torch.Tensor
    dual: torch.Tensor
    cost: float
    feasible: bool
    iterations: int
    max_violation: float


def minimum_cost_perturbation(
    constraints: torch.Tensor,
    deficits: torch.Tensor,
    metric: CovarianceMetric,
    *,
    tolerance: float = 1e-6,
    max_iterations: int = 2000,
) -> ProjectionSolution:
    """Solve ``min delta.T Sigma^-1 delta`` subject to ``C delta >= b``.

    The non-negative dual is solved by cyclic coordinate ascent (Hildreth's
    method). The dual system is constraint-sized, so runtime depends on the
    number of concepts being controlled rather than the model width.
    """
    if constraints.ndim != 2:
        raise ValueError("constraints must have shape [n_constraints, d_model]")
    if deficits.ndim != 1 or len(deficits) != len(constraints):
        raise ValueError("deficits must have shape [n_constraints]")
    if constraints.shape[1] != metric.d_model:
        raise ValueError("constraint width does not match the covariance metric")
    if len(deficits) == 0:
        raise ValueError("at least one constraint is required")
    if tolerance <= 0 or max_iterations <= 0:
        raise ValueError("tolerance and max_iterations must be positive")
    if not torch.isfinite(constraints).all() or not torch.isfinite(deficits).all():
        raise ValueError("constraints and deficits must be finite")

    output_dtype = constraints.dtype
    work_dtype = torch.float64
    work_constraints = constraints.to(dtype=work_dtype)
    work_deficits = deficits.to(device=constraints.device, dtype=work_dtype)
    sigma_constraints = metric.apply(work_constraints)
    gram = work_constraints @ sigma_constraints.T
    diagonal = gram.diag()
    if (diagonal <= 1e-14).any():
        raise ValueError("constraint has zero covariance norm")

    dual = torch.zeros_like(work_deficits)
    iterations = 0
    for iteration in range(1, max_iterations + 1):
        iterations = iteration
        previous = dual.clone()
        for i in range(len(dual)):
            residual = work_deficits[i] - gram[i] @ dual
            dual[i] = torch.clamp(dual[i] + residual / diagonal[i], min=0.0)
        if (dual - previous).abs().max() <= tolerance:
            break

    delta = dual @ sigma_constraints
    violations = torch.clamp(work_deficits - work_constraints @ delta, min=0.0)
    max_violation = float(violations.max())
    cost = float(dual @ (gram @ dual))
    return ProjectionSolution(
        delta=delta.to(dtype=output_dtype),
        dual=dual.to(dtype=deficits.dtype),
        cost=cost,
        feasible=max_violation <= tolerance * 10,
        iterations=iterations,
        max_violation=max_violation,
    )


@dataclass(frozen=True)
class ScaleEvaluation:
    scale: float
    passed: bool


@dataclass(frozen=True)
class ScaleSearchResult:
    scale: float
    passed: bool
    evaluations: tuple[ScaleEvaluation, ...]


def minimum_passing_scale(
    predicate: Callable[[float], bool],
    *,
    initial: float = 1.0,
    maximum: float = 16.0,
    relative_tolerance: float = 0.01,
) -> ScaleSearchResult:
    """Bracket and bisect the smallest non-negative passing scale."""
    if initial <= 0 or maximum <= 0 or initial > maximum:
        raise ValueError("require 0 < initial <= maximum")
    if not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must lie in (0, 1)")

    evaluations: list[ScaleEvaluation] = []

    def evaluate(scale: float) -> bool:
        passed = bool(predicate(scale))
        evaluations.append(ScaleEvaluation(scale=scale, passed=passed))
        return passed

    if evaluate(0.0):
        return ScaleSearchResult(0.0, True, tuple(evaluations))

    lower = 0.0
    upper = initial
    while True:
        if evaluate(upper):
            break
        lower = upper
        if upper >= maximum:
            return ScaleSearchResult(maximum, False, tuple(evaluations))
        upper = min(maximum, upper * 2.0)

    while (upper - lower) / max(upper, 1e-12) > relative_tolerance:
        midpoint = (lower + upper) / 2.0
        if evaluate(midpoint):
            upper = midpoint
        else:
            lower = midpoint
    return ScaleSearchResult(upper, True, tuple(evaluations))
