# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Adaptive causal J-operators.

This module adds an operator surface around existing :class:`JacobianLens`
checkpoints without replacing the saved lens format or ``JacobianLens.apply``.
The first implementation is deliberately global-only: local corrections are an
explicit zero contribution with diagnostics, so downstream causal code can be
written against the adaptive interface before prompt-local fitting is enabled.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch

from jlens.lens import JacobianLens


@dataclass(frozen=True)
class OperatorConfig:
    model_id: str
    operator_id: str
    alpha: float = 0.0
    profile: str = "laptop"

    def __post_init__(self) -> None:
        if not self.model_id:
            raise ValueError("model_id must not be empty")
        if not self.operator_id:
            raise ValueError("operator_id must not be empty")
        if not 0.0 <= self.alpha <= 1.0:
            raise ValueError("alpha must be in [0, 1]")


@dataclass(frozen=True)
class OperatorDiagnostics:
    layer: int
    local_status: str
    alpha: float
    evidence: str
    uncertainty: float | None = None


class AdaptiveCausalOperator:
    """Matrix-free composition ``A[l,x] = J_global + alpha * DeltaJ_local``.

    ``DeltaJ_local`` is not estimated in the initial milestone. The fallback is
    observable through :meth:`diagnostics`, and ``matvec``/``rmatvec`` remain
    usable for global-only causal prediction and intervention solving.
    """

    def __init__(
        self,
        lens: JacobianLens,
        config: OperatorConfig,
    ) -> None:
        self.lens = lens
        self.config = config

    @property
    def d_model(self) -> int:
        return self.lens.d_model

    def _check_layer(self, layer: int) -> None:
        if layer not in self.lens.source_layers:
            raise ValueError(
                f"layer {layer} not in source_layers; fitted layers are "
                f"{self.lens.source_layers}"
            )

    def matvec(self, *, layer: int, residual: torch.Tensor) -> torch.Tensor:
        """Apply the current operator to a source-layer residual vector."""
        self._check_layer(layer)
        return self.lens.transport(residual, layer)

    def rmatvec(self, *, layer: int, covector: torch.Tensor) -> torch.Tensor:
        """Apply the transpose operator to an output-space covector."""
        self._check_layer(layer)
        value = covector.float()
        if layer in self.lens.jacobians:
            jacobian = self.lens.jacobians[layer].to(value.device)
            return jacobian.T @ value
        sketch = self.lens.sketches[layer]
        probes = sketch.probes.to(value.device, non_blocking=True)
        corrections = sketch.corrections.to(value.device, non_blocking=True)
        return value + corrections.T @ (probes @ value) / sketch.rank

    def diagnostics(self, *, layer: int, context: str | None = None) -> OperatorDiagnostics:
        """Return explicit fallback status for a layer/context pair."""
        del context
        self._check_layer(layer)
        return OperatorDiagnostics(
            layer=layer,
            local_status="global-only",
            alpha=0.0,
            evidence="predicted",
            uncertainty=None,
        )
