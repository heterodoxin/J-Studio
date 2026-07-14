# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Matrix-free causal intervention proposals for adaptive J-operators."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

import torch

from jlens.jspace import TokenSetContrast
from jlens.operator import AdaptiveCausalOperator
from jlens.trace import CausalTrace, ControlResult, DosePoint


@dataclass(frozen=True)
class InterventionProposal:
    layer: int
    delta: torch.Tensor
    predicted_effect: float
    norm: float
    converged: bool
    iterations: int
    residual_error: float
    condition_estimate: float


def _cg(
    matvec: Callable[[torch.Tensor], torch.Tensor],
    rhs: torch.Tensor,
    *,
    max_iter: int,
    tolerance: float,
) -> tuple[torch.Tensor, bool, int, float]:
    x = torch.zeros_like(rhs)
    r = rhs - matvec(x)
    p = r.clone()
    rs_old = float(r @ r)
    if math.sqrt(rs_old) <= tolerance:
        return x, True, 0, math.sqrt(rs_old)
    for iteration in range(1, max_iter + 1):
        ap = matvec(p)
        denom = float(p @ ap)
        if denom <= 0.0 or not math.isfinite(denom):
            return x, False, iteration, math.sqrt(max(rs_old, 0.0))
        alpha = rs_old / denom
        x = x + alpha * p
        r = r - alpha * ap
        rs_new = float(r @ r)
        error = math.sqrt(max(rs_new, 0.0))
        if error <= tolerance:
            return x, True, iteration, error
        beta = rs_new / max(rs_old, 1e-30)
        p = r + beta * p
        rs_old = rs_new
    return x, False, max_iter, math.sqrt(max(rs_old, 0.0))


class CausalInterventionSolver:
    """Solve regularized minimum-change edits against an adaptive operator."""

    def __init__(self, operator: AdaptiveCausalOperator, readout: torch.Tensor) -> None:
        if readout.ndim != 2 or readout.shape[1] != operator.d_model:
            raise ValueError(
                "readout must have shape [vocab_size, operator.d_model]"
            )
        self.operator = operator
        self.readout = readout.float()

    @property
    def vocab_size(self) -> int:
        return self.readout.shape[0]

    def _logit_matvec(self, layer: int, delta: torch.Tensor) -> torch.Tensor:
        transported = self.operator.matvec(layer=layer, residual=delta)
        return self.readout.to(transported.device) @ transported

    def _logit_rmatvec(self, layer: int, covector: torch.Tensor) -> torch.Tensor:
        readout = self.readout.to(covector.device)
        return self.operator.rmatvec(layer=layer, covector=readout.T @ covector)

    def solve(
        self,
        *,
        layer: int,
        target: TokenSetContrast,
        protected: Sequence[TokenSetContrast] = (),
        lambda_reg: float = 1e-2,
        protected_weight: float = 0.0,
        max_iter: int = 128,
        tolerance: float = 1e-5,
    ) -> InterventionProposal:
        if lambda_reg <= 0.0:
            raise ValueError("lambda_reg must be positive")
        if protected_weight < 0.0:
            raise ValueError("protected_weight must be non-negative")
        y = target.target_vector(vocab_size=self.vocab_size)
        protected_vectors = tuple(
            item.target_vector(vocab_size=self.vocab_size) for item in protected
        )

        rhs = self._logit_rmatvec(layer, y)

        def normal_matvec(delta: torch.Tensor) -> torch.Tensor:
            logits = self._logit_matvec(layer, delta)
            weighted = logits.clone()
            for vector in protected_vectors:
                weighted = weighted + protected_weight * (vector @ logits) * vector
            return self._logit_rmatvec(layer, weighted) + lambda_reg * delta

        delta, converged, iterations, residual_error = _cg(
            normal_matvec,
            rhs,
            max_iter=max_iter,
            tolerance=tolerance,
        )
        effect = self._logit_matvec(layer, delta)
        predicted_effect = target.score(effect)
        norm = float(delta.norm())
        condition_estimate = max(
            1.0,
            (1.0 + protected_weight + lambda_reg) / lambda_reg,
        )
        return InterventionProposal(
            layer=layer,
            delta=delta.detach().cpu(),
            predicted_effect=predicted_effect,
            norm=norm,
            converged=converged,
            iterations=iterations,
            residual_error=residual_error,
            condition_estimate=condition_estimate,
        )


@dataclass(frozen=True)
class ControlSpec:
    name: str
    seed: int | None
    transform: Callable[[torch.Tensor], torch.Tensor]


def noop_control() -> ControlSpec:
    return ControlSpec("noop", None, lambda delta: torch.zeros_like(delta))


def norm_matched_random_control(*, seed: int) -> ControlSpec:
    def transform(delta: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=delta.device)
        generator.manual_seed(seed)
        random = torch.randn(delta.shape, generator=generator, device=delta.device)
        random_norm = random.norm().clamp_min(1e-12)
        return random * (delta.norm() / random_norm)

    return ControlSpec("norm-matched-random", seed, transform)


def shuffled_target_control(*, seed: int) -> ControlSpec:
    def transform(delta: torch.Tensor) -> torch.Tensor:
        generator = torch.Generator(device=delta.device)
        generator.manual_seed(seed)
        indices = torch.randperm(delta.numel(), generator=generator, device=delta.device)
        return delta.reshape(-1)[indices].reshape_as(delta)

    return ControlSpec("shuffled-target", seed, transform)


def measure_dose_response(
    trace: CausalTrace,
    proposal: InterventionProposal,
    *,
    strengths: Sequence[float],
    measure: Callable[[torch.Tensor], float],
    controls: Sequence[ControlSpec] = (),
) -> CausalTrace:
    dose_points = []
    for strength in strengths:
        if strength < 0.0:
            raise ValueError("dose strengths must be non-negative")
        delta = proposal.delta * float(strength)
        dose_points.append(
            DosePoint(
                strength=float(strength),
                predicted_effect=proposal.predicted_effect * float(strength),
                observed_effect=float(measure(delta)),
                norm=float(delta.norm()),
            )
        )
    control_results = []
    full_delta = proposal.delta * trace.strength
    for control in controls:
        delta = control.transform(full_delta)
        control_results.append(
            ControlResult(
                name=control.name,
                observed_effect=float(measure(delta)),
                norm=float(delta.norm()),
                seed=control.seed,
            )
        )
    return replace(
        trace,
        dose_response=tuple(dose_points),
        controls=tuple(control_results),
    )
