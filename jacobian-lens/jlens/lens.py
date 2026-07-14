# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Applying a fitted Jacobian lens.

A :class:`JacobianLens` holds the per-layer ``J_l`` matrices produced by
:func:`jlens.fitting.fit`. :meth:`JacobianLens.apply` runs a forward pass and
reads out the requested layers; :meth:`JacobianLens.transport` is the bare
``J_l @ h`` for callers that already have residuals.
"""

from __future__ import annotations

import os
from collections.abc import Sequence
from dataclasses import dataclass

import torch

from jlens.geometry import CovarianceMetric
from jlens.hooks import ActivationRecorder
from jlens.protocol import LensModel


@dataclass(frozen=True)
class SketchedJacobian:
    """Identity plus a low-rank randomized estimate of a Jacobian.

    Rows of ``probes`` are output-space directions ``z_i`` and rows of
    ``corrections`` are ``J.T @ z_i - z_i``. Therefore the transport is
    ``h + (h @ corrections.T) @ probes / rank``.
    """

    probes: torch.Tensor
    corrections: torch.Tensor

    def __post_init__(self) -> None:
        probes = self.probes.detach().float().cpu()
        corrections = self.corrections.detach().float().cpu()
        if probes.ndim != 2 or corrections.shape != probes.shape:
            raise ValueError(
                "sketch probes and corrections must have equal [rank, d] shape"
            )
        if not len(probes):
            raise ValueError("sketch rank must be positive")
        object.__setattr__(self, "probes", probes)
        object.__setattr__(self, "corrections", corrections)

    @property
    def rank(self) -> int:
        return self.probes.shape[0]

    @property
    def d_model(self) -> int:
        return self.probes.shape[1]

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        value = residual.float()
        probes = self.probes.to(value.device, non_blocking=True)
        corrections = self.corrections.to(value.device, non_blocking=True)
        return value + (value @ corrections.T) @ probes / self.rank


class JacobianLens:
    """A fitted Jacobian lens: per-layer ``J_l`` matrices and the readout method.

    Attributes:
        jacobians: ``{layer_index: Tensor[d_model, d_model]}``. Each ``J_l``
            maps the residual at layer ``l`` into the final-layer basis.
        source_layers: Sorted list of fitted layer indices.
        n_prompts: Number of prompts the lens was averaged over.
        d_model: Residual-stream width.
    """

    def __init__(
        self,
        jacobians: dict[int, torch.Tensor] | None = None,
        *,
        n_prompts: int,
        d_model: int,
        sketches: dict[int, SketchedJacobian] | None = None,
        geometry: dict[int, CovarianceMetric] | None = None,
        metadata: dict[str, str] | None = None,
    ) -> None:
        self.jacobians = {layer: J.float() for layer, J in (jacobians or {}).items()}
        self.sketches = dict(sketches or {})
        overlap = set(self.jacobians) & set(self.sketches)
        if overlap:
            raise ValueError(
                f"layers cannot have dense and sketched transports: {overlap}"
            )
        self.source_layers = sorted(set(self.jacobians) | set(self.sketches))
        if not self.source_layers:
            raise ValueError("a lens needs at least one fitted source layer")
        self.n_prompts = n_prompts
        self.d_model = d_model
        self.geometry = dict(geometry or {})
        self.metadata = dict(metadata or {})
        unknown_geometry = set(self.geometry) - set(self.source_layers)
        if unknown_geometry:
            raise ValueError(f"geometry has unknown layers {sorted(unknown_geometry)}")
        for layer, metric in self.geometry.items():
            if metric.d_model != d_model:
                raise ValueError(
                    f"geometry at layer {layer} has d_model={metric.d_model}, "
                    f"expected {d_model}"
                )
        for layer, sketch in self.sketches.items():
            if sketch.d_model != d_model:
                raise ValueError(
                    f"sketch at layer {layer} has d_model={sketch.d_model}, "
                    f"expected {d_model}"
                )

    def __repr__(self) -> str:
        return (
            f"JacobianLens(d_model={self.d_model}, n_prompts={self.n_prompts}, "
            f"source_layers=[{self.source_layers[0]}..{self.source_layers[-1]}] "
            f"({len(self.source_layers)} layers))"
        )

    def save(self, path: str, *, dtype: torch.dtype = torch.float16) -> None:
        """Save to ``path``. Jacobians are stored as ``dtype`` (default fp16:
        halves file size; entries are O(1) so the range is not a constraint
        and fp16's extra mantissa bits beat bf16 here)."""
        torch.save(
            {
                "format_version": 3,
                "J": {layer: J.to(dtype) for layer, J in self.jacobians.items()},
                "sketches": {
                    layer: {
                        "probes": sketch.probes.to(dtype),
                        "corrections": sketch.corrections.to(dtype),
                    }
                    for layer, sketch in self.sketches.items()
                },
                "n_prompts": self.n_prompts,
                "source_layers": self.source_layers,
                "d_model": self.d_model,
                "geometry": {
                    layer: {
                        "diagonal": metric.diagonal.float(),
                        "factors": (
                            None if metric.factors is None else metric.factors.float()
                        ),
                        "calibrated": metric.calibrated,
                    }
                    for layer, metric in self.geometry.items()
                },
                "metadata": self.metadata,
            },
            path,
        )

    @classmethod
    def load(cls, path: str) -> JacobianLens:
        """Load a lens previously written by :meth:`save`."""
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        if "J" not in checkpoint and "sketches" not in checkpoint:
            raise ValueError(
                f"{path} is not a JacobianLens file "
                f"(found keys {sorted(checkpoint)!r}; a fit() checkpoint?)"
            )
        geometry = {
            int(layer): CovarianceMetric(
                diagonal=state["diagonal"],
                factors=state.get("factors"),
                calibrated=state.get("calibrated", True),
            )
            for layer, state in checkpoint.get("geometry", {}).items()
        }
        sketches = {
            int(layer): SketchedJacobian(
                probes=state["probes"], corrections=state["corrections"]
            )
            for layer, state in checkpoint.get("sketches", {}).items()
        }
        return cls(
            jacobians=checkpoint.get("J", {}),
            n_prompts=checkpoint["n_prompts"],
            d_model=checkpoint["d_model"],
            sketches=sketches,
            geometry=geometry,
            metadata=checkpoint.get("metadata", {}),
        )

    @classmethod
    def from_pretrained(
        cls,
        name_or_path: str,
        *,
        filename: str = "lens.pt",
        revision: str | None = None,
    ) -> JacobianLens:
        """Load a lens from a local file, a local directory, or a HuggingFace
        Hub ``repo_id``. ``filename`` is the path inside the directory or repo
        (so one Hub repo can host lenses for many models); ignored when
        ``name_or_path`` is itself a file. ``revision`` selects a Hub branch,
        tag, or commit. Deserialisation goes through :meth:`load`
        (``weights_only=True``)."""
        if os.path.isfile(name_or_path):
            return cls.load(name_or_path)
        if not os.path.isdir(name_or_path):
            from huggingface_hub import snapshot_download

            name_or_path = snapshot_download(
                name_or_path, allow_patterns=[filename], revision=revision
            )
        return cls.load(os.path.join(name_or_path, filename))

    @classmethod
    def merge(cls, lenses: Sequence[JacobianLens]) -> JacobianLens:
        """Combine lenses fitted on disjoint prompt subsets into one
        (``n_prompts``-weighted mean of the inputs).

        Args:
            lenses: Lenses to merge. Must agree on ``source_layers`` and
                ``d_model``.

        Raises:
            ValueError: If ``lenses`` is empty or the inputs disagree on shape.
        """
        if not lenses:
            raise ValueError("merge() needs at least one lens")
        first = lenses[0]
        for other in lenses[1:]:
            if (
                other.source_layers != first.source_layers
                or other.d_model != first.d_model
                or set(other.jacobians) != set(first.jacobians)
                or set(other.sketches) != set(first.sketches)
            ):
                raise ValueError(
                    "lenses disagree on transport type / source layers / d_model"
                )
        n_total = sum(lens.n_prompts for lens in lenses)
        merged: dict[int, torch.Tensor] = {}
        for layer in first.jacobians:
            weighted_sum = sum(
                lens.jacobians[layer] * lens.n_prompts for lens in lenses
            )
            merged[layer] = weighted_sum / n_total
        merged_sketches = {}
        for layer, reference in first.sketches.items():
            if any(
                not torch.allclose(
                    lens.sketches[layer].probes,
                    reference.probes,
                    rtol=0,
                    atol=3e-3,
                )
                for lens in lenses[1:]
            ):
                raise ValueError("sketched lenses use different probes")
            correction = (
                sum(
                    lens.sketches[layer].corrections * lens.n_prompts for lens in lenses
                )
                / n_total
            )
            merged_sketches[layer] = SketchedJacobian(
                probes=reference.probes, corrections=correction
            )
        return cls(
            jacobians=merged,
            sketches=merged_sketches,
            n_prompts=n_total,
            d_model=first.d_model,
        )

    def metric(self, layer: int) -> CovarianceMetric:
        """Return calibrated residual geometry or an explicit identity fallback."""
        if layer not in self.source_layers:
            raise ValueError(
                f"layer {layer} not in source_layers; fitted layers are {self.source_layers}"
            )
        return self.geometry.get(
            layer, CovarianceMetric.identity(self.d_model, calibrated=False)
        )

    @property
    def transport_shrinkage(self) -> float:
        """Held-out-selected weight on the fitted Jacobian correction.

        ``1`` is the original fitted operator. Values in ``(0, 1)`` contract
        only ``J - I`` and therefore remain an explicit Jacobian transport,
        rather than switching to a separate readout path.
        """
        raw = self.metadata.get("transport_shrinkage", "1")
        try:
            value = float(raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("transport_shrinkage must be a number") from exc
        if not 0.0 < value <= 1.0:
            raise ValueError("transport_shrinkage must lie in (0, 1]")
        return value

    def with_transport_shrinkage(self, value: float) -> JacobianLens:
        """Return an equivalent lens with a contracted fitted correction."""
        value = float(value)
        if not 0.0 < value <= 1.0:
            raise ValueError("transport shrinkage must lie in (0, 1]")
        metadata = dict(self.metadata)
        metadata["transport_shrinkage"] = f"{value:.8g}"
        return JacobianLens(
            jacobians=self.jacobians,
            sketches=self.sketches,
            n_prompts=self.n_prompts,
            d_model=self.d_model,
            geometry=self.geometry,
            metadata=metadata,
        )

    def transport(self, residual: torch.Tensor, layer: int) -> torch.Tensor:
        """Map a residual at ``layer`` into the final-layer basis: ``J_l @ h``.

        Args:
            residual: Tensor of shape ``[..., d_model]``.
            layer: Source layer index (must be in :attr:`source_layers`).
        """
        value = residual.float()
        if layer in self.sketches:
            transported = self.sketches[layer].apply(value)
        else:
            J_bar = self.jacobians[layer].to(value.device)
            transported = value @ J_bar.T
        alpha = self.transport_shrinkage
        return value + alpha * (transported - value)

    @torch.no_grad()
    def apply(
        self,
        model: LensModel,
        prompt: str,
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] | None = None,
        max_seq_len: int = 512,
        use_jacobian: bool = True,
    ) -> tuple[dict[int, torch.Tensor], torch.Tensor, torch.Tensor]:
        """Run ``model`` on ``prompt`` and return lens logits at ``positions``.

        Args:
            model: The model to read out from.
            prompt: Input text.
            layers: Layers to read out at. Defaults to all of
                :attr:`source_layers`. Must be a subset of
                :attr:`source_layers` when ``use_jacobian`` is ``True``.
            positions: Token positions to read out (Python indexing into the
                sequence; negative indices count from the end). ``None`` returns
                every position.
            max_seq_len: Truncate the prompt to this many tokens.
            use_jacobian: If ``False``, skip the ``J_l`` transport (vanilla
                logit-lens baseline).

        Returns:
            A triple ``(lens_logits, model_logits, input_ids)``. ``lens_logits``
            maps each requested layer to a ``[n_positions, vocab_size]`` tensor;
            ``model_logits`` is the model's actual final-layer logits at the
            same positions (same shape). ``n_positions`` is ``len(positions)``,
            or the full sequence length when ``positions`` is ``None``.

        Raises:
            ValueError: If any requested layer is out of range for the model,
                or (with ``use_jacobian``) not in :attr:`source_layers`.
        """
        if layers is None:
            layers = self.source_layers
        out_of_range = sorted(l for l in set(layers) if not 0 <= l < model.n_layers)
        if out_of_range:
            raise ValueError(
                f"layers {out_of_range} out of range for a {model.n_layers}-layer model"
            )
        unknown = set(layers) - set(self.source_layers)
        if use_jacobian and unknown:
            raise ValueError(
                f"layers {sorted(unknown)} not in source_layers; "
                f"fitted layers are {self.source_layers}"
            )
        final_layer = model.n_layers - 1
        record_at = sorted(set(layers) | {final_layer})

        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=record_at) as recorder:
            model.forward(input_ids)
            activations = {i: recorder.activations[i].detach() for i in record_at}

        def select(layer: int) -> torch.Tensor:
            """Residuals at the requested positions: ``[n_positions, d_model]``."""
            full = activations[layer][0]  # [seq_len, d_model]
            return (full if positions is None else full[list(positions)]).float()

        transported = []
        for layer in layers:
            residual = select(layer)
            if use_jacobian:
                residual = self.transport(residual, layer)
            transported.append(residual)

        # One matrix-matrix unembedding is substantially faster than one
        # matrix-vector-heavy call per layer on both ROCm and CUDA.
        counts = [len(value) for value in transported]
        combined = model.unembed(torch.cat(transported, dim=0)).float().cpu()
        chunks = combined.split(counts)
        lens_logits = dict(zip(layers, chunks, strict=True))

        model_logits = model.unembed(select(final_layer)).float().cpu()
        return lens_logits, model_logits, input_ids
