# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Concept resolution and calibrated interventions in J-space."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import asdict, dataclass
from typing import Any

import torch

from jlens.concepts import compact_basis, normalize_directions, sequence_alignment
from jlens.geometry import minimum_cost_perturbation, minimum_passing_scale
from jlens.hooks import (
    ActivationEditor,
    ActivationRecorder,
    ResidualEdit,
    ResidualTransform,
    ResidualTransformEditor,
)
from jlens.lens import JacobianLens
from jlens.protocol import LensModel


@dataclass(frozen=True)
class ConceptSpec:
    text: str | None
    token_ids: tuple[int, ...]
    variants: tuple[tuple[int, ...], ...]
    display: tuple[str, ...]
    experimental: bool


class ConceptResolver:
    """Resolve text or explicit IDs into visible tokenizer-specific concepts."""

    def __init__(self, tokenizer: Any) -> None:
        self.tokenizer = tokenizer

    def _tokenize(self, text: str) -> tuple[int, ...]:
        try:
            encoded = self.tokenizer(
                text,
                return_tensors="pt",
                add_special_tokens=False,
            )
        except TypeError:
            encoded = self.tokenizer(text, return_tensors="pt")
        ids = encoded.input_ids[0].tolist()
        special_ids = {
            value
            for name in ("bos_token_id", "eos_token_id", "pad_token_id")
            if (value := getattr(self.tokenizer, name, None)) is not None
        }
        return tuple(int(token_id) for token_id in ids if token_id not in special_ids)

    def resolve(self, value: str | int | Sequence[int]) -> ConceptSpec:
        if isinstance(value, str):
            if not value:
                raise ValueError("concept text must not be empty")
            variants = []
            for text_variant in (value, f" {value}"):
                token_ids = self._tokenize(text_variant)
                if token_ids and token_ids not in variants:
                    variants.append(token_ids)
            if not variants:
                raise ValueError(f"concept {value!r} produced no non-special tokens")
            selected = next(
                (variant for variant in variants if len(variant) == 1),
                min(variants, key=len),
            )
            text = value
        elif isinstance(value, int):
            selected = (value,)
            variants = [selected]
            text = None
        else:
            selected = tuple(int(token_id) for token_id in value)
            variants = [selected]
            text = None
        if not selected or any(token_id < 0 for token_id in selected):
            raise ValueError(
                "token IDs must be a non-empty sequence of non-negative ints"
            )
        display = tuple(self.tokenizer.decode([token_id]) for token_id in selected)
        return ConceptSpec(
            text=text,
            token_ids=selected,
            variants=tuple(variants),
            display=display,
            experimental=len(selected) != 1,
        )


def local_score_covectors(
    model: LensModel,
    lens: JacobianLens,
    residual: torch.Tensor,
    layer: int,
    token_ids: Sequence[int],
) -> torch.Tensor:
    """Exact local derivatives of implemented J-lens logits with respect to h."""
    if residual.ndim != 1 or residual.shape[0] != lens.d_model:
        raise ValueError(f"residual must have shape [{lens.d_model}]")
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    differentiable = residual.detach().float().requires_grad_(True)
    logits = model.unembed(lens.transport(differentiable, layer))
    if logits.ndim != 1:
        raise ValueError("unembedding a single residual must return [vocab_size]")
    if min(token_ids) < 0 or max(token_ids) >= len(logits):
        raise ValueError("token ID out of vocabulary range")
    gradients = []
    for index, token_id in enumerate(token_ids):
        gradient = torch.autograd.grad(
            logits[token_id],
            differentiable,
            retain_graph=index < len(token_ids) - 1,
        )[0]
        gradients.append(gradient.detach())
    return torch.stack(gradients)


def transported_unembedding_covectors(
    model: LensModel,
    lens: JacobianLens,
    layer: int,
    token_ids: Sequence[int],
) -> torch.Tensor:
    """Pull raw unembedding rows back through the effective J transport.

    This is the phrase-editing basis. It deliberately excludes the final norm:
    residual hooks operate before that norm, and including its prompt-local
    derivative makes the direction nearly orthogonal to the residual stream.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    head = getattr(model, "_lm_head", None) or getattr(model, "lm_head", None)
    weight = getattr(head, "weight", None)
    if weight is None or weight.ndim != 2:
        raise ValueError("phrase interventions require an unembedding weight")
    if min(token_ids) < 0 or max(token_ids) >= weight.shape[0]:
        raise ValueError("token ID out of vocabulary range")
    rows = weight.detach().float()[list(token_ids)]
    probe = torch.zeros(
        lens.d_model, device=rows.device, dtype=torch.float32, requires_grad=True
    )
    transported = lens.transport(probe, layer)
    scores = rows @ transported
    gradients = []
    for index, score in enumerate(scores):
        gradients.append(
            torch.autograd.grad(
                score,
                probe,
                retain_graph=index < len(scores) - 1,
            )[0].detach()
        )
    return torch.stack(gradients)


def downstream_score_covectors(
    model: LensModel,
    prompt: str,
    *,
    layer: int,
    position: int,
    token_ids: Sequence[int],
    max_seq_len: int = 512,
) -> torch.Tensor:
    """Derivatives of measured next-token logits with respect to a residual edit.

    Unlike :func:`local_score_covectors`, this traces through the real
    downstream blocks from ``layer`` to the final unembedding. These covectors
    are the right basis for interventions whose success criterion is an actual
    generation-logit change rather than movement in the fitted J-lens readout.
    """
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    resolved = position + seq_len if position < 0 else position
    if not 0 <= resolved < seq_len:
        raise IndexError(f"position {position} out of range for sequence length {seq_len}")
    final_layer = model.n_layers - 1
    with ActivationRecorder(
        model.layers, at=sorted({layer, final_layer}), start_graph_at=layer
    ) as recorder:
        model.forward(input_ids)
    residuals = recorder.activations[layer]
    logits = model.unembed(recorder.activations[final_layer][0, -1].float())
    if logits.ndim != 1:
        raise ValueError("unembedding a single residual must return [vocab_size]")
    if min(token_ids) < 0 or max(token_ids) >= len(logits):
        raise ValueError("token ID out of vocabulary range")
    gradients = []
    for index, token_id in enumerate(token_ids):
        gradient = torch.autograd.grad(
            logits[token_id],
            residuals,
            retain_graph=index < len(token_ids) - 1,
        )[0][0, resolved]
        gradients.append(gradient.detach())
    return torch.stack(gradients)


def downstream_logits_and_covectors(
    model: LensModel,
    prompt: str,
    *,
    layer: int,
    position: int,
    token_ids: Sequence[int],
    base_delta: torch.Tensor | None = None,
    max_seq_len: int = 512,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    """Measured next-token logits and local covectors at an optional edit point."""
    if not token_ids:
        raise ValueError("token_ids must not be empty")
    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    resolved = position + seq_len if position < 0 else position
    if not 0 <= resolved < seq_len:
        raise IndexError(f"position {position} out of range for sequence length {seq_len}")
    final_layer = model.n_layers - 1
    captured: dict[str, torch.Tensor] = {}

    def edit_and_capture(_module, _inputs, output):
        tensor = output if torch.is_tensor(output) else output[0]
        edited = tensor.clone()
        if base_delta is not None:
            edited[0, resolved] += base_delta.to(edited.device, edited.dtype)
        edited.requires_grad_(True)
        captured["residuals"] = edited
        if torch.is_tensor(output):
            return edited
        if isinstance(output, tuple):
            return (edited, *output[1:])
        if isinstance(output, list):
            return [edited, *output[1:]]
        raise TypeError("block output must be a tensor, tuple, or list")

    handle = model.layers[layer].register_forward_hook(edit_and_capture)
    try:
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids)
    finally:
        handle.remove()
    logits = model.unembed(recorder.activations[final_layer][0, -1].float())
    if min(token_ids) < 0 or max(token_ids) >= len(logits):
        raise ValueError("token ID out of vocabulary range")
    gradients = []
    for index, token_id in enumerate(token_ids):
        gradient = torch.autograd.grad(
            logits[token_id],
            captured["residuals"],
            retain_graph=index < len(token_ids) - 1,
        )[0][0, resolved]
        gradients.append(gradient.detach())
    return logits.detach().float().cpu(), torch.stack(gradients), resolved


@dataclass(frozen=True)
class SearchPoint:
    scale: float
    passed: bool
    target_margin: float
    downstream_shift: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.scale, self.target_margin, self.downstream_shift)
        ):
            raise ValueError("search point values must be finite")


@dataclass(frozen=True)
class InterventionTrace:
    operation: str
    target_ids: tuple[int, ...]
    source_ids: tuple[int, ...]
    experimental: bool
    selected_layer: int | None
    selected_positions: tuple[int, ...]
    selected_scale: float
    normalized_cost: float
    baseline_scores: dict[str, float]
    after_scores: dict[str, float]
    baseline_top_ids: tuple[int, ...]
    after_top_ids: tuple[int, ...]
    search_points: tuple[SearchPoint, ...]
    warnings: tuple[str, ...]
    sequence_logprob_before: float | None = None
    sequence_logprob_after: float | None = None
    application_delay: int = 0
    carrier_phrase: str | None = None

    def __post_init__(self) -> None:
        scalar_values = [self.selected_scale, self.normalized_cost]
        scalar_values.extend(self.baseline_scores.values())
        scalar_values.extend(self.after_scores.values())
        scalar_values.extend(
            value
            for value in (
                self.sequence_logprob_before,
                self.sequence_logprob_after,
            )
            if value is not None
        )
        if not all(math.isfinite(value) for value in scalar_values):
            raise ValueError("intervention trace numeric values must be finite")
        if self.selected_scale < 0 or self.normalized_cost < 0:
            raise ValueError("intervention scale and cost must be non-negative")
        if self.application_delay < 0:
            raise ValueError("application delay must be non-negative")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> InterventionTrace:
        data = dict(value)
        for key in (
            "target_ids",
            "source_ids",
            "selected_positions",
            "baseline_top_ids",
            "after_top_ids",
            "warnings",
        ):
            data[key] = tuple(data[key])
        data["search_points"] = tuple(
            point if isinstance(point, SearchPoint) else SearchPoint(**point)
            for point in data["search_points"]
        )
        return cls(**data)


@dataclass(frozen=True)
class InterventionResult:
    success: bool
    trace: InterventionTrace
    delta: torch.Tensor | None
    message: str
    operator: PhraseResidualOperator | None = None
    operators: tuple[tuple[int, PhraseResidualOperator], ...] = ()


@dataclass(frozen=True)
class PhraseResidualOperator:
    """Bounded residual projection/transport for a token sequence."""

    operation: str
    source_basis: torch.Tensor | None
    target_directions: torch.Tensor | None
    alignment: torch.Tensor | None
    scale: float

    def __post_init__(self) -> None:
        if self.operation not in {"inject", "suppress", "replace"}:
            raise ValueError(f"unsupported phrase operation {self.operation!r}")
        if not math.isfinite(self.scale) or self.scale <= 0:
            raise ValueError("phrase operator scale must be finite and positive")
        if self.operation == "suppress" and self.source_basis is None:
            raise ValueError("suppression requires a source basis")
        if self.operation == "inject" and self.target_directions is None:
            raise ValueError("injection requires target directions")
        if self.operation == "replace" and (
            self.source_basis is None
            or self.target_directions is None
            or self.alignment is None
        ):
            raise ValueError("replacement requires source, target, and alignment")

    def apply(self, residual: torch.Tensor) -> torch.Tensor:
        return self.apply_at_step(residual, 0)

    def apply_at_step(self, residual: torch.Tensor, step: int | None) -> torch.Tensor:
        if residual.ndim != 1:
            raise ValueError("phrase operator expects one residual vector")
        if step is not None and step < 0:
            raise ValueError("phrase step must be non-negative")
        value = residual.float()
        if self.operation == "inject":
            targets = self.target_directions.to(value.device, value.dtype)
            target = (
                targets.mean(dim=1)
                if step is None
                else targets[:, min(step, targets.shape[1] - 1)]
            )
            target = target / target.norm().clamp_min(1e-8)
            rms = value.square().mean().sqrt()
            edited = value + self.scale * rms * target
        else:
            source = self.source_basis.to(value.device, value.dtype)
            coefficients = value @ source
            source_component = source @ coefficients
            edited = value - min(self.scale, 1.0) * source_component
            if self.operation == "replace":
                targets = self.target_directions.to(value.device, value.dtype)
                target = (
                    targets.mean(dim=1)
                    if step is None
                    else targets[:, min(step, targets.shape[1] - 1)]
                )
                target = target / target.norm().clamp_min(1e-8)
                magnitude = (
                    coefficients.norm()
                    if step in (0, None)
                    else value.square().mean().sqrt()
                )
                edited = edited + self.scale * magnitude * target
        if not torch.isfinite(edited).all():
            raise ValueError("phrase operator produced non-finite residuals")
        return edited.to(residual.dtype)

    def make_transform(
        self, *, ordered: bool = True, delay: int = 0
    ) -> PhraseResidualSchedule:
        return PhraseResidualSchedule(self, ordered=ordered, delay=delay)


@dataclass
class PhraseResidualSchedule:
    """Fresh per-generation state for ordered phrase-token transport."""

    operator: PhraseResidualOperator
    ordered: bool = True
    delay: int = 0
    step: int = 0
    calls: int = 0

    def __post_init__(self) -> None:
        if self.delay < 0:
            raise ValueError("phrase schedule delay must be non-negative")

    def __call__(self, residual: torch.Tensor) -> torch.Tensor:
        if self.calls < self.delay:
            self.calls += 1
            return residual
        self.calls += 1
        edited = self.operator.apply_at_step(
            residual, self.step if self.ordered else None
        )
        self.step += 1
        return edited


@dataclass(frozen=True)
class ReadResult:
    layer: int
    position: int
    top_ids: tuple[int, ...]
    top_scores: tuple[float, ...]


@dataclass
class _ForwardState:
    input_ids: torch.Tensor
    position: int
    residual: torch.Tensor
    lens_logits: torch.Tensor
    final_logits: torch.Tensor


@dataclass
class _Evaluation:
    scale: float
    passed: bool
    target_margin: float
    downstream_shift: float
    state: _ForwardState
    sequence_logprob: float | None


@dataclass
class _Candidate:
    result: InterventionResult
    objective: float


class InterventionEngine:
    """Read and minimally edit token-indexed J-space coordinates."""

    def __init__(self, model: LensModel, lens: JacobianLens) -> None:
        if model.d_model != lens.d_model:
            raise ValueError("model and lens disagree on d_model")
        self.model = model
        self.lens = lens
        self.resolver = ConceptResolver(model.tokenizer)

    def _capture(
        self,
        prompt: str,
        layer: int,
        position: int,
        delta: torch.Tensor | None = None,
    ) -> _ForwardState:
        if layer not in self.lens.source_layers:
            raise ValueError(
                f"layer {layer} not fitted; available layers are {self.lens.source_layers}"
            )
        input_ids = self.model.encode(prompt, max_length=512)
        seq_len = input_ids.shape[1]
        resolved = position + seq_len if position < 0 else position
        if not 0 <= resolved < seq_len:
            raise IndexError(
                f"position {position} out of range for sequence length {seq_len}"
            )
        final_layer = self.model.n_layers - 1
        edit_context = (
            nullcontext()
            if delta is None
            else ActivationEditor(
                self.model.layers,
                [ResidualEdit(layer, (resolved,), delta)],
            )
        )
        with torch.no_grad(), edit_context:
            with ActivationRecorder(
                self.model.layers, at=sorted({layer, final_layer})
            ) as recorder:
                self.model.forward(input_ids)
        residual = recorder.activations[layer][0, resolved].detach().float()
        final_residual = recorder.activations[final_layer][0, -1].detach().float()
        lens_logits = (
            self.model.unembed(self.lens.transport(residual, layer))
            .detach()
            .float()
            .cpu()
        )
        final_logits = self.model.unembed(final_residual).detach().float().cpu()
        return _ForwardState(
            input_ids=input_ids,
            position=resolved,
            residual=residual,
            lens_logits=lens_logits,
            final_logits=final_logits,
        )

    def read(
        self,
        prompt: str,
        *,
        layer: int,
        position: int = -1,
        top_n: int = 10,
    ) -> ReadResult:
        if top_n <= 0:
            raise ValueError("top_n must be positive")
        state = self._capture(prompt, layer, position)
        values, indices = state.lens_logits.topk(min(top_n, len(state.lens_logits)))
        return ReadResult(
            layer=layer,
            position=state.position,
            top_ids=tuple(int(value) for value in indices),
            top_scores=tuple(float(value) for value in values),
        )

    def apply(
        self,
        result: InterventionResult,
        *,
        once: bool = True,
        max_applications: int | None = None,
        ordered: bool = True,
        delay: int | None = None,
    ) -> ActivationEditor | ResidualTransformEditor:
        """Return a scoped hook that applies a selected edit during inference.

        ``once`` defaults every edit to the first matching forward. Phrase
        transforms may instead use an explicit positive ``max_applications``;
        ``once=False`` with no limit keeps them active for the full generation.
        """
        if not result.success or (
            result.delta is None
            and result.operator is None
            and not result.operators
        ):
            raise ValueError("only a successful intervention result can be applied")
        layer = result.trace.selected_layer
        if layer is None or not result.trace.selected_positions:
            raise ValueError("intervention result has no selected edit location")
        if max_applications is not None and max_applications <= 0:
            raise ValueError("max_applications must be positive")
        resolved_delay = result.trace.application_delay if delay is None else delay
        if resolved_delay < 0:
            raise ValueError("application delay must be non-negative")
        phrase_limit = (
            max_applications if max_applications is not None else (1 if once else None)
        )
        if result.operators:
            return ResidualTransformEditor(
                self.model.layers,
                [
                    ResidualTransform(
                        layer=operator_layer,
                        positions=result.trace.selected_positions,
                        transform=operator.make_transform(
                            ordered=ordered, delay=resolved_delay
                        ),
                        max_applications=phrase_limit,
                    )
                    for operator_layer, operator in result.operators
                ],
            )
        if result.operator is not None:
            return ResidualTransformEditor(
                self.model.layers,
                [
                    ResidualTransform(
                        layer=layer,
                        positions=result.trace.selected_positions,
                        transform=result.operator.make_transform(
                            ordered=ordered, delay=resolved_delay
                        ),
                        max_applications=phrase_limit,
                    )
                ],
            )
        return ActivationEditor(
            self.model.layers,
            [
                ResidualEdit(
                    layer=layer,
                    positions=result.trace.selected_positions,
                    delta=result.delta,
                    max_applications=1 if once else None,
                )
            ],
        )

    def phrase_inject(
        self,
        prompt: str,
        target: str | int | Sequence[int],
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] = (-1,),
        application_positions: Sequence[int] | None = None,
        application_delay: int = 0,
        carrier_phrase: str | None = None,
        maximum_scale: float = 16.0,
        effect_probe: Callable[
            [tuple[tuple[int, PhraseResidualOperator], ...], tuple[int, ...]],
            tuple[bool, float],
        ]
        | None = None,
    ) -> InterventionResult:
        return self._phrase_intervene(
            "inject",
            prompt,
            source=None,
            target=self.resolver.resolve(target),
            layers=layers,
            positions=positions,
            application_positions=application_positions,
            application_delay=application_delay,
            carrier_phrase=carrier_phrase,
            maximum_scale=maximum_scale,
            effect_probe=effect_probe,
        )

    def phrase_suppress(
        self,
        prompt: str,
        source: str | int | Sequence[int],
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] = (-1,),
        application_positions: Sequence[int] | None = None,
        application_delay: int = 0,
        carrier_phrase: str | None = None,
        maximum_scale: float = 16.0,
        effect_probe: Callable[
            [tuple[tuple[int, PhraseResidualOperator], ...], tuple[int, ...]],
            tuple[bool, float],
        ]
        | None = None,
    ) -> InterventionResult:
        return self._phrase_intervene(
            "suppress",
            prompt,
            source=self.resolver.resolve(source),
            target=None,
            layers=layers,
            positions=positions,
            application_positions=application_positions,
            application_delay=application_delay,
            carrier_phrase=carrier_phrase,
            maximum_scale=maximum_scale,
            effect_probe=effect_probe,
        )

    def phrase_replace(
        self,
        prompt: str,
        source: str | int | Sequence[int],
        target: str | int | Sequence[int],
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] = (-1,),
        application_positions: Sequence[int] | None = None,
        application_delay: int = 0,
        carrier_phrase: str | None = None,
        maximum_scale: float = 16.0,
        effect_probe: Callable[
            [tuple[tuple[int, PhraseResidualOperator], ...], tuple[int, ...]],
            tuple[bool, float],
        ]
        | None = None,
    ) -> InterventionResult:
        return self._phrase_intervene(
            "replace",
            prompt,
            source=self.resolver.resolve(source),
            target=self.resolver.resolve(target),
            layers=layers,
            positions=positions,
            application_positions=application_positions,
            application_delay=application_delay,
            carrier_phrase=carrier_phrase,
            maximum_scale=maximum_scale,
            effect_probe=effect_probe,
        )

    def _phrase_residuals(
        self, prompt: str, layers: Sequence[int], position: int
    ) -> tuple[dict[int, torch.Tensor], int]:
        unknown = set(layers) - set(self.lens.source_layers)
        if unknown:
            raise ValueError(
                f"layers {sorted(unknown)} not fitted; available layers are "
                f"{self.lens.source_layers}"
            )
        input_ids = self.model.encode(prompt, max_length=512)
        seq_len = input_ids.shape[1]
        resolved = position + seq_len if position < 0 else position
        if not 0 <= resolved < seq_len:
            raise IndexError(
                f"position {position} out of range for sequence length {seq_len}"
            )
        with torch.no_grad(), ActivationRecorder(
            self.model.layers, at=layers
        ) as recorder:
            self.model.forward(input_ids)
        return (
            {
                layer: recorder.activations[layer][0, resolved].detach().float()
                for layer in layers
            },
            resolved,
        )

    def _phrase_directions(
        self,
        residual: torch.Tensor,
        layer: int,
        token_ids: tuple[int, ...],
    ) -> torch.Tensor:
        del residual
        covectors = transported_unembedding_covectors(
            self.model, self.lens, layer, token_ids
        )
        return normalize_directions(covectors).T.contiguous()

    @staticmethod
    def _phrase_measurements(
        residual: torch.Tensor,
        edited: torch.Tensor,
        source_basis: torch.Tensor | None,
        target_directions: torch.Tensor | None,
    ) -> tuple[dict[str, float], dict[str, float]]:
        residual = residual.float()
        edited = edited.float()
        residual_norm = residual.norm().clamp_min(1e-8)
        edited_norm = edited.norm()
        residual_rms = residual.square().mean().sqrt().clamp_min(1e-8)
        edited_rms = edited.square().mean().sqrt()
        cosine = float(
            torch.nn.functional.cosine_similarity(residual, edited, dim=0)
        )
        baseline = {
            "residual_norm": float(residual_norm),
            "residual_rms": float(residual_rms),
        }
        after = {
            "residual_norm": float(edited_norm),
            "residual_rms": float(edited_rms),
            "residual_cosine": cosine,
            "norm_ratio": float(edited_norm / residual_norm),
        }
        if source_basis is not None:
            source = source_basis.to(residual.device)
            baseline["source_energy"] = float((residual @ source).norm())
            after["source_energy"] = float((edited @ source).norm())
        if target_directions is not None:
            targets = target_directions.to(residual.device)
            centroid = targets.mean(dim=1)
            centroid = centroid / centroid.norm().clamp_min(1e-8)
            baseline["target_energy"] = float(residual @ centroid)
            after["target_energy"] = float(edited @ centroid)
        return baseline, after

    @staticmethod
    def _phrase_passes(
        operation: str,
        baseline: dict[str, float],
        after: dict[str, float],
    ) -> bool:
        if after["residual_cosine"] < 0.9 or not 0.5 <= after["norm_ratio"] <= 1.5:
            return False
        if operation == "inject":
            threshold = 0.05 * max(baseline["residual_rms"], 1e-8)
            return after["target_energy"] >= baseline["target_energy"] + threshold
        source_before = baseline["source_energy"]
        if after["source_energy"] > 0.8 * source_before + 1e-7:
            return False
        if operation == "replace":
            threshold = 0.05 * max(source_before, 1e-8)
            return after["target_energy"] >= baseline["target_energy"] + threshold
        return True

    def _phrase_intervene(
        self,
        operation: str,
        prompt: str,
        *,
        source: ConceptSpec | None,
        target: ConceptSpec | None,
        layers: Sequence[int] | None,
        positions: Sequence[int],
        application_positions: Sequence[int] | None,
        application_delay: int,
        carrier_phrase: str | None,
        maximum_scale: float,
        effect_probe: Callable[
            [tuple[tuple[int, PhraseResidualOperator], ...], tuple[int, ...]],
            tuple[bool, float],
        ]
        | None,
    ) -> InterventionResult:
        if not math.isfinite(maximum_scale) or maximum_scale <= 0:
            raise ValueError("maximum_scale must be finite and positive")
        if not positions:
            raise ValueError("positions must not be empty")
        if application_positions is not None and not application_positions:
            raise ValueError("application_positions must be None or non-empty")
        if application_delay < 0:
            raise ValueError("application_delay must be non-negative")
        if layers is None:
            start = len(self.lens.source_layers) // 3
            layers = self.lens.source_layers[start:]
        layers = tuple(dict.fromkeys(int(layer) for layer in layers))
        if not layers:
            raise ValueError("layers must not be empty")
        budget = maximum_scale
        ladder = [
            value
            for value in (
                1 / 16,
                1 / 8,
                1 / 4,
                1 / 2,
                3 / 4,
                *tuple(float(value) for value in range(1, 17)),
            )
            if value <= budget
        ]
        if not ladder or ladder[-1] < budget:
            ladder.append(budget)
        best_failure = None
        for position in positions:
            selected_positions = (position,)
            if application_positions is not None:
                selected_positions = tuple(
                    dict.fromkeys(int(value) for value in application_positions)
                )
            residuals, _resolved = self._phrase_residuals(prompt, layers, position)
            geometry = []
            for layer in layers:
                residual = residuals[layer]
                source_basis = None
                target_directions = None
                alignment = None
                if source is not None:
                    source_directions = self._phrase_directions(
                        residual, layer, source.token_ids
                    )
                    source_basis = compact_basis(source_directions.T)
                if target is not None:
                    target_directions = self._phrase_directions(
                        residual, layer, target.token_ids
                    )
                if operation == "replace":
                    alignment = sequence_alignment(
                        source_basis.shape[1], target_directions.shape[1]
                    )
                geometry.append(
                    (layer, residual, source_basis, target_directions, alignment)
                )
            points = []
            for scale in ladder:
                operator_pairs = []
                baselines = []
                afters = []
                costs = []
                for (
                    layer,
                    residual,
                    source_basis,
                    target_directions,
                    alignment,
                ) in geometry:
                    operator = PhraseResidualOperator(
                        operation=operation,
                        source_basis=source_basis,
                        target_directions=target_directions,
                        alignment=alignment,
                        scale=scale,
                    )
                    edited = operator.apply(residual)
                    baseline_scores, after_scores = self._phrase_measurements(
                        residual,
                        edited,
                        source_basis,
                        target_directions,
                    )
                    operator_pairs.append((layer, operator))
                    baselines.append(baseline_scores)
                    afters.append(after_scores)
                    costs.append(
                        float(
                            (edited - residual).norm()
                            / residual.norm().clamp_min(1e-8)
                        )
                    )

                def averaged(values: list[dict[str, float]]) -> dict[str, float]:
                    keys = set.intersection(*(set(value) for value in values))
                    return {
                        key: sum(value[key] for value in values) / len(values)
                        for key in keys
                    }

                baseline_scores = averaged(baselines)
                after_scores = averaged(afters)
                locally_passed = self._phrase_passes(
                    operation, baseline_scores, after_scores
                )
                probe_safe = (
                    after_scores["residual_cosine"] >= 0.5
                    and 0.25 <= after_scores["norm_ratio"] <= 3.0
                )
                causal_shift = 1.0 - after_scores["residual_cosine"]
                causally_passed = True
                if probe_safe and effect_probe is not None:
                    causally_passed, causal_shift = effect_probe(
                        tuple(operator_pairs), selected_positions
                    )
                passed = (
                    locally_passed
                    if effect_probe is None
                    else probe_safe and causally_passed
                )
                target_margin = (
                    baseline_scores.get("source_energy", 0.0)
                    - after_scores.get("source_energy", 0.0)
                    + after_scores.get("target_energy", 0.0)
                    - baseline_scores.get("target_energy", 0.0)
                )
                selected_scale = scale
                points.append(
                    SearchPoint(
                        scale=selected_scale,
                        passed=passed,
                        target_margin=target_margin,
                        downstream_shift=causal_shift,
                    )
                )
                trace = InterventionTrace(
                    operation=operation,
                    target_ids=target.token_ids if target is not None else (),
                    source_ids=source.token_ids if source is not None else (),
                    experimental=bool(
                        (source is not None and source.experimental)
                        or (target is not None and target.experimental)
                    ),
                    selected_layer=layers[0] if passed else None,
                    selected_positions=selected_positions,
                    selected_scale=selected_scale,
                    normalized_cost=sum(costs) / len(costs),
                    baseline_scores=baseline_scores,
                    after_scores=after_scores,
                    baseline_top_ids=(),
                    after_top_ids=(),
                    search_points=tuple(points),
                    warnings=(
                        "multi-token-jspace-transform",
                        f"layer-range:{layers[0]}-{layers[-1]}",
                        *(("natural-carrier",) if carrier_phrase else ()),
                        *(
                            ("generation-causal-probe",)
                            if effect_probe is not None
                            else ()
                        ),
                    ),
                    application_delay=application_delay,
                    carrier_phrase=carrier_phrase,
                )
                result = InterventionResult(
                    success=passed,
                    trace=trace,
                    delta=None,
                    message=(
                        f"minimum effective J-space strength "
                        f"{selected_scale:.2f} across layers "
                        f"{layers[0]}-{layers[-1]}"
                        + (
                            f" after {application_delay} decode steps"
                            if application_delay
                            else ""
                        )
                        if passed
                        else "bounded J-space phrase search did not pass"
                    ),
                    operator=operator_pairs[0][1] if passed else None,
                    operators=tuple(operator_pairs) if passed else (),
                )
                if passed:
                    return result
                best_failure = result
        assert best_failure is not None
        return best_failure

    def inject(
        self,
        prompt: str,
        target: str | int | Sequence[int],
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] = (-1,),
        top_k: int = 1,
        margin: float = 0.0,
        maximum_scale: float = 16.0,
        relative_tolerance: float = 0.01,
    ) -> InterventionResult:
        return self._intervene(
            "inject",
            prompt,
            target=self.resolver.resolve(target),
            source=None,
            layers=layers,
            positions=positions,
            top_k=top_k,
            margin=margin,
            preserve_top_k=0,
            preservation_tolerance=0.0,
            maximum_scale=maximum_scale,
            relative_tolerance=relative_tolerance,
        )

    def suppress(
        self,
        prompt: str,
        target: str | int | Sequence[int],
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] = (-1,),
        top_k: int = 5,
        margin: float = 0.0,
        maximum_scale: float = 16.0,
        relative_tolerance: float = 0.01,
    ) -> InterventionResult:
        concept = self.resolver.resolve(target)
        if concept.experimental:
            raise ValueError("suppression currently requires a single token")
        return self._intervene(
            "suppress",
            prompt,
            target=concept,
            source=None,
            layers=layers,
            positions=positions,
            top_k=top_k,
            margin=margin,
            preserve_top_k=0,
            preservation_tolerance=0.0,
            maximum_scale=maximum_scale,
            relative_tolerance=relative_tolerance,
        )

    def replace(
        self,
        prompt: str,
        source: str | int | Sequence[int],
        target: str | int | Sequence[int],
        *,
        layers: Sequence[int] | None = None,
        positions: Sequence[int] = (-1,),
        margin: float = 0.125,
        preserve_top_k: int = 8,
        preservation_tolerance: float = 0.05,
        maximum_scale: float = 16.0,
        relative_tolerance: float = 0.01,
    ) -> InterventionResult:
        source_spec = self.resolver.resolve(source)
        target_spec = self.resolver.resolve(target)
        if source_spec.experimental or target_spec.experimental:
            raise ValueError("replacement currently requires single-token concepts")
        return self._intervene(
            "replace",
            prompt,
            target=target_spec,
            source=source_spec,
            layers=layers,
            positions=positions,
            top_k=1,
            margin=margin,
            preserve_top_k=preserve_top_k,
            preservation_tolerance=preservation_tolerance,
            maximum_scale=maximum_scale,
            relative_tolerance=relative_tolerance,
        )

    def _sequence_logprob(
        self,
        baseline: _ForwardState,
        target_ids: tuple[int, ...],
        layer: int,
        delta: torch.Tensor | None,
    ) -> float:
        continuation = torch.tensor(
            target_ids[:-1],
            device=baseline.input_ids.device,
            dtype=baseline.input_ids.dtype,
        ).unsqueeze(0)
        input_ids = torch.cat([baseline.input_ids, continuation], dim=1)
        edit_context = (
            nullcontext()
            if delta is None
            else ActivationEditor(
                self.model.layers,
                [ResidualEdit(layer, (baseline.position,), delta)],
            )
        )
        final_layer = self.model.n_layers - 1
        with torch.no_grad(), edit_context:
            with ActivationRecorder(self.model.layers, at=[final_layer]) as recorder:
                self.model.forward(input_ids)
        residuals = recorder.activations[final_layer][0].detach().float()
        start = baseline.input_ids.shape[1] - 1
        total = 0.0
        for offset, token_id in enumerate(target_ids):
            logits = self.model.unembed(residuals[start + offset]).float()
            total += float(logits.log_softmax(dim=-1)[token_id].detach())
        return total

    def _intervene(
        self,
        operation: str,
        prompt: str,
        *,
        target: ConceptSpec,
        source: ConceptSpec | None,
        layers: Sequence[int] | None,
        positions: Sequence[int],
        top_k: int,
        margin: float,
        preserve_top_k: int,
        preservation_tolerance: float,
        maximum_scale: float,
        relative_tolerance: float,
    ) -> InterventionResult:
        if top_k <= 0 or maximum_scale <= 0:
            raise ValueError("top_k and maximum_scale must be positive")
        if preserve_top_k < 0 or preservation_tolerance < 0:
            raise ValueError("preservation settings must be non-negative")
        if not positions:
            raise ValueError("positions must not be empty")
        if layers is None:
            start = len(self.lens.source_layers) // 3
            layers = self.lens.source_layers[start:]
        if not layers:
            raise ValueError("layers must not be empty")

        candidates = [
            self._candidate(
                operation,
                prompt,
                target,
                source,
                layer,
                position,
                top_k,
                margin,
                preserve_top_k,
                preservation_tolerance,
                maximum_scale,
                relative_tolerance,
            )
            for layer in layers
            for position in positions
        ]
        successful = [candidate for candidate in candidates if candidate.result.success]
        if successful:
            return min(successful, key=lambda candidate: candidate.objective).result
        return max(candidates, key=lambda candidate: candidate.objective).result

    def _candidate(
        self,
        operation: str,
        prompt: str,
        target: ConceptSpec,
        source: ConceptSpec | None,
        layer: int,
        position: int,
        top_k: int,
        margin: float,
        preserve_top_k: int,
        preservation_tolerance: float,
        maximum_scale: float,
        relative_tolerance: float,
    ) -> _Candidate:
        baseline = self._capture(prompt, layer, position)
        vocab_size = len(baseline.final_logits)
        controlled_ids = list(target.token_ids)
        if source is not None:
            controlled_ids.extend(source.token_ids)
        if min(controlled_ids) < 0 or max(controlled_ids) >= vocab_size:
            raise ValueError("concept token ID out of vocabulary range")

        ranking = baseline.final_logits.argsort(descending=True)
        top_k = min(top_k, vocab_size - 1)
        if operation == "inject":
            competitor_ids = []
            for target_id in target.token_ids:
                filtered = ranking[ranking != target_id]
                competitor_ids.append(int(filtered[top_k - 1]))
        elif operation == "suppress":
            target_id = target.token_ids[0]
            filtered = ranking[ranking != target_id]
            competitor_ids = [int(filtered[top_k - 1])]
        else:
            competitor_ids = [source.token_ids[0]]

        preservation_ids: list[int] = []
        if operation == "replace" and preserve_top_k:
            excluded = set(target.token_ids + source.token_ids)
            preservation_ids = [
                int(token_id) for token_id in ranking if int(token_id) not in excluded
            ][:preserve_top_k]

        gradient_ids = list(
            dict.fromkeys(controlled_ids + competitor_ids + preservation_ids)
        )
        gradients = downstream_score_covectors(
            self.model,
            prompt,
            layer=layer,
            position=position,
            token_ids=gradient_ids,
        )
        gradient = dict(zip(gradient_ids, gradients, strict=True))
        rows = []
        deficits = []
        if operation == "inject":
            for target_id, competitor_id in zip(
                target.token_ids, competitor_ids, strict=True
            ):
                rows.append(gradient[target_id] - gradient[competitor_id])
                current = float(
                    baseline.final_logits[target_id]
                    - baseline.final_logits[competitor_id]
                )
                deficits.append(margin - current)
        elif operation == "suppress":
            target_id = target.token_ids[0]
            competitor_id = competitor_ids[0]
            rows.append(gradient[competitor_id] - gradient[target_id])
            current = float(
                baseline.final_logits[competitor_id] - baseline.final_logits[target_id]
            )
            deficits.append(margin - current)
        else:
            target_id = target.token_ids[0]
            source_id = source.token_ids[0]
            rows.append(gradient[target_id] - gradient[source_id])
            current = float(
                baseline.final_logits[target_id] - baseline.final_logits[source_id]
            )
            deficits.append(margin - current)
            for preserve_id in preservation_ids:
                bound = preservation_tolerance * max(
                    abs(float(baseline.final_logits[preserve_id])), 1.0
                )
                rows.extend((gradient[preserve_id], -gradient[preserve_id]))
                deficits.extend((-bound, -bound))

        constraints = torch.stack(rows)
        deficit_tensor = torch.tensor(
            deficits, device=constraints.device, dtype=constraints.dtype
        )
        solution = minimum_cost_perturbation(
            constraints, deficit_tensor, self.lens.metric(layer)
        )
        if not solution.feasible:
            return self._failed_solution(
                operation, target, source, layer, position, baseline, solution.delta
            )

        baseline_sequence = (
            self._sequence_logprob(baseline, target.token_ids, layer, None)
            if target.experimental
            else None
        )
        evaluations: dict[float, _Evaluation] = {}

        def evaluate_edit(scale: float, edit: torch.Tensor) -> bool:
            state = self._capture(prompt, layer, position, edit)
            target_final = state.final_logits[list(target.token_ids)]
            baseline_target_final = baseline.final_logits[list(target.token_ids)]
            if operation == "inject":
                after_top = set(
                    int(value) for value in state.final_logits.topk(top_k).indices
                )
                target_margin = min(
                    float(state.final_logits[target_id] - state.final_logits[competitor])
                    for target_id, competitor in zip(
                        target.token_ids, competitor_ids, strict=True
                    )
                )
                downstream_shift = float((target_final - baseline_target_final).mean())
                passed = (
                    set(target.token_ids).issubset(after_top)
                    and target_margin >= margin
                )
            elif operation == "suppress":
                target_id = target.token_ids[0]
                target_margin = float(
                    state.final_logits[competitor_ids[0]] - state.final_logits[target_id]
                )
                downstream_shift = float(
                    baseline.final_logits[target_id] - state.final_logits[target_id]
                )
                passed = (
                    target_id not in state.final_logits.topk(top_k).indices.tolist()
                    and target_margin >= margin
                )
            else:
                target_id = target.token_ids[0]
                source_id = source.token_ids[0]
                target_margin = float(
                    state.final_logits[target_id] - state.final_logits[source_id]
                )
                downstream_shift = float(
                    (state.final_logits[target_id] - state.final_logits[source_id])
                    - (
                        baseline.final_logits[target_id]
                        - baseline.final_logits[source_id]
                    )
                )
                passed = target_margin >= margin and downstream_shift >= 0
            sequence_logprob = (
                self._sequence_logprob(baseline, target.token_ids, layer, edit)
                if target.experimental
                else None
            )
            if sequence_logprob is not None:
                passed = passed and sequence_logprob >= baseline_sequence
            evaluations[scale] = _Evaluation(
                scale,
                passed,
                target_margin,
                downstream_shift,
                state,
                sequence_logprob,
            )
            return passed

        def evaluate(scale: float) -> bool:
            return evaluate_edit(scale, solution.delta * scale)

        search = minimum_passing_scale(
            evaluate,
            initial=min(1.0, maximum_scale),
            maximum=maximum_scale,
            relative_tolerance=relative_tolerance,
        )
        selected_delta = solution.delta * search.scale
        selected = evaluations[search.scale]
        relevant_ids = list(dict.fromkeys(controlled_ids + competitor_ids))
        warnings = []
        if not self.lens.metric(layer).calibrated:
            warnings.append("uncalibrated identity geometry")
        if target.experimental:
            warnings.append("multi-token intervention is experimental")
        selected_scale = (
            float(selected_delta.norm() / solution.delta.norm().clamp_min(1e-12))
            if solution.delta.numel()
            else search.scale
        )
        trace = InterventionTrace(
            operation=operation,
            target_ids=target.token_ids,
            source_ids=() if source is None else source.token_ids,
            experimental=target.experimental,
            selected_layer=layer,
            selected_positions=(selected.state.position,),
            selected_scale=selected_scale,
            normalized_cost=selected_scale * selected_scale * solution.cost,
            baseline_scores={
                str(token_id): float(baseline.final_logits[token_id])
                for token_id in relevant_ids
            },
            after_scores={
                str(token_id): float(selected.state.final_logits[token_id])
                for token_id in relevant_ids
            },
            baseline_top_ids=tuple(int(value) for value in ranking[:32]),
            after_top_ids=tuple(
                int(value)
                for value in selected.state.final_logits.topk(
                    min(32, vocab_size)
                ).indices
            ),
            search_points=tuple(
                SearchPoint(
                    scale=point.scale,
                    passed=point.passed,
                    target_margin=evaluations[point.scale].target_margin,
                    downstream_shift=evaluations[point.scale].downstream_shift,
                )
                for point in search.evaluations
            ),
            warnings=tuple(warnings),
            sequence_logprob_before=baseline_sequence,
            sequence_logprob_after=selected.sequence_logprob,
        )
        passed = search.passed
        message = (
            "minimum passing intervention found"
            if passed
            else "bounded search found no passing intervention"
        )
        result = InterventionResult(
            success=passed,
            trace=trace,
            delta=selected_delta.detach().cpu(),
            message=message,
        )
        objective = trace.normalized_cost if passed else selected.target_margin
        return _Candidate(result, objective)

    def _failed_solution(
        self,
        operation: str,
        target: ConceptSpec,
        source: ConceptSpec | None,
        layer: int,
        position: int,
        baseline: _ForwardState,
        delta: torch.Tensor,
    ) -> _Candidate:
        ranking = baseline.final_logits.argsort(descending=True)
        trace = InterventionTrace(
            operation=operation,
            target_ids=target.token_ids,
            source_ids=() if source is None else source.token_ids,
            experimental=target.experimental,
            selected_layer=layer,
            selected_positions=(baseline.position,),
            selected_scale=0.0,
            normalized_cost=0.0,
            baseline_scores={
                str(token_id): float(baseline.final_logits[token_id])
                for token_id in target.token_ids
            },
            after_scores={},
            baseline_top_ids=tuple(int(value) for value in ranking[:32]),
            after_top_ids=tuple(int(value) for value in ranking[:32]),
            search_points=(),
            warnings=("constrained projection was infeasible",),
        )
        return _Candidate(
            InterventionResult(
                False, trace, delta.detach().cpu(), "projection infeasible"
            ),
            float("-inf"),
        )
