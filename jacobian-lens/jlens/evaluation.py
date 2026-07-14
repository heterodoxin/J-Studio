# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Held-out quality gates for progressive Jacobian-lens fitting."""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import torch

from jlens.benchmark import (
    ReadoutCase,
    find_token_position_containing,
    run_readout_benchmark,
)
from jlens.examples import EXAMPLES, resolve_prompt
from jlens.lens import JacobianLens
from jlens.protocol import LensModel


@dataclass(frozen=True)
class FitQuality:
    """Metrics used to decide whether a fitted lens can be labeled stable."""

    pass_at_10: float
    rank_overlap: float
    finite: bool
    minimum_pass_at_10: float = field(default=0.3, repr=False, kw_only=True)
    minimum_rank_overlap: float = field(default=0.5, repr=False, kw_only=True)
    stable: bool = field(init=False)
    reasons: tuple[str, ...] = field(init=False)

    def __post_init__(self) -> None:
        reasons = []
        metrics = (self.pass_at_10, self.rank_overlap)
        if not self.finite or not all(math.isfinite(value) for value in metrics):
            reasons.append("transport or quality metrics are not finite")
        if self.pass_at_10 < self.minimum_pass_at_10:
            reasons.append(
                f"held-out J-space pass@10 is below {self.minimum_pass_at_10:.2f}"
            )
        rank_overlap_low = self.rank_overlap < self.minimum_rank_overlap
        if rank_overlap_low:
            reasons.append(
                "successive-stage top-10 rank overlap is below "
                f"{self.minimum_rank_overlap:.2f}"
            )
        object.__setattr__(self, "reasons", tuple(reasons))
        object.__setattr__(
            self,
            "stable",
            self.finite and all(math.isfinite(value) for value in metrics) and not reasons,
        )


def _item_fields(item: Any) -> tuple[str, Sequence[str], int]:
    if isinstance(item, Mapping):
        return (
            str(item["prompt"]),
            tuple(str(value) for value in item["intermediates"]),
            int(item.get("position", -1)),
        )
    return str(item.prompt), tuple(item.intermediates), int(getattr(item, "position", -1))


def _variant_token_ids(model: LensModel, text: str) -> set[int]:
    """Return single-token bare/space-prefixed variants without special tokens."""
    result: set[int] = set()
    tokenizer = model.tokenizer
    for variant in (text, f" {text}"):
        ids: list[int]
        try:
            encoded = tokenizer(variant, add_special_tokens=False)
        except TypeError:
            encoded = tokenizer(variant)
        raw = encoded.input_ids if hasattr(encoded, "input_ids") else encoded["input_ids"]
        if torch.is_tensor(raw):
            raw = raw.detach().cpu().reshape(-1).tolist()
        elif raw and isinstance(raw[0], (list, tuple)):
            raw = list(raw[0])
        else:
            raw = list(raw)
        special = {
            value
            for value in (
                getattr(tokenizer, "bos_token_id", None),
                getattr(tokenizer, "eos_token_id", None),
                getattr(tokenizer, "pad_token_id", None),
            )
            if value is not None
        }
        ids = [int(value) for value in raw if int(value) not in special]
        if len(ids) == 1:
            result.add(ids[0])
    return result


def _rank(logits: torch.Tensor, token_id: int) -> int:
    return int((logits > logits[token_id]).sum().item()) + 1


def evaluate_fit_quality(
    model: LensModel,
    lens: JacobianLens,
    validation_items: Sequence[Any],
    previous: JacobianLens | None = None,
    *,
    max_seq_len: int = 512,
    minimum_rank_overlap: float = 0.5,
) -> FitQuality:
    """Measure held-out intermediate recovery using only J-space transport."""
    recovered = 0
    total = 0
    overlaps: list[float] = []
    finite = True

    for item in validation_items:
        prompt, intermediates, position = _item_fields(item)
        lens_logits, _, _ = lens.apply(
            model, prompt, positions=[position], max_seq_len=max_seq_len
        )
        previous_logits = None
        if previous is not None:
            shared_layers = sorted(set(lens.source_layers) & set(previous.source_layers))
            if shared_layers:
                previous_logits, _, _ = previous.apply(
                    model,
                    prompt,
                    layers=shared_layers,
                    positions=[position],
                    max_seq_len=max_seq_len,
                )
                for layer in shared_layers:
                    current_ids = set(
                        lens_logits[layer][0].topk(
                            min(10, lens_logits[layer].shape[-1])
                        ).indices.tolist()
                    )
                    prior_ids = set(
                        previous_logits[layer][0].topk(
                            min(10, previous_logits[layer].shape[-1])
                        ).indices.tolist()
                    )
                    overlaps.append(len(current_ids & prior_ids) / len(current_ids | prior_ids))

        finite = finite and all(
            torch.isfinite(logits).all().item() for logits in lens_logits.values()
        )
        for intermediate in intermediates:
            token_ids = _variant_token_ids(model, intermediate)
            if not token_ids:
                continue
            total += 1
            lens_rank = min(
                _rank(logits[0], token_id)
                for logits in lens_logits.values()
                for token_id in token_ids
                if token_id < logits.shape[-1]
            )
            recovered += lens_rank <= 10

    pass_at_10 = recovered / total if total else 0.0
    rank_overlap = sum(overlaps) / len(overlaps) if overlaps else 1.0
    return FitQuality(
        pass_at_10,
        rank_overlap,
        finite,
        minimum_rank_overlap=minimum_rank_overlap,
    )


def select_transport_shrinkage(
    model: LensModel,
    lens: JacobianLens,
    validation_items: Sequence[Any],
    *,
    candidates: Sequence[float] = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
    max_seq_len: int = 512,
) -> tuple[JacobianLens, FitQuality]:
    """Select correction shrinkage on held-out J-space recovery.

    Ties prefer the larger value so the fitted operator retains as much of its
    measured Jacobian correction as the held-out evidence supports.
    """
    if not validation_items:
        raise ValueError("transport shrinkage selection needs validation items")
    if not candidates:
        raise ValueError("transport shrinkage selection needs candidates")
    evaluated = []
    for value in candidates:
        candidate = lens.with_transport_shrinkage(float(value))
        quality = evaluate_fit_quality(
            model,
            candidate,
            validation_items,
            max_seq_len=max_seq_len,
        )
        evaluated.append((quality.pass_at_10, quality.rank_overlap, float(value), candidate, quality))
    _, _, _, selected, quality = max(evaluated, key=lambda row: row[:3])
    return selected, quality


def standard_readout_cases(
    model: LensModel,
    *,
    max_rank: int = 100,
) -> tuple[ReadoutCase, ...]:
    """Reference viewing checks over completed, causally ordered transcripts."""
    ascii_example = next(example for example in EXAMPLES if example.slug == "ascii-face")
    ascii_prompt = resolve_prompt(ascii_example, model.tokenizer)
    ascii_context = (
        ascii_prompt + "\n\nIt is an ASCII face with eyes, a nose, and a smile."
    )
    nose_position = find_token_position_containing(model, ascii_context, "^")

    multihop = next(example for example in EXAMPLES if example.slug == "multihop")
    multihop_prompt = resolve_prompt(multihop, model.tokenizer)
    multihop_context = multihop_prompt + " the Euro."
    boot_position = find_token_position_containing(model, multihop_context, "boot")
    euro_position = find_token_position_containing(model, multihop_context, "Euro")
    mode = "prompt+generated-response"
    return (
        ReadoutCase(
            "ascii-face nose at caret",
            ascii_context,
            "nose",
            nose_position,
            max_rank=max_rank,
            context_mode=mode,
        ),
        ReadoutCase(
            "multihop country at boot position",
            multihop_context,
            "Italy",
            boot_position,
            max_rank=max_rank,
            context_mode=mode,
        ),
        ReadoutCase(
            "multihop currency at answer boundary",
            multihop_context,
            "Euro",
            euro_position - 1,
            max_rank=max_rank,
            context_mode=mode,
        ),
    )


def select_readout_shrinkage(
    model: LensModel,
    lens: JacobianLens,
    cases: Sequence[ReadoutCase],
    *,
    candidates: Sequence[float] = (0.1, 0.2, 0.35, 0.5, 0.75, 1.0),
) -> tuple[JacobianLens, tuple[Any, ...]]:
    """Select the strongest correction that passes the most viewing checks."""
    if not cases:
        raise ValueError("readout shrinkage selection needs cases")
    evaluated = []
    for value in candidates:
        candidate = lens.with_transport_shrinkage(float(value))
        results = run_readout_benchmark(model, candidate, tuple(cases))
        passed = sum(result.success for result in results)
        worst_rank = max(result.best_rank for result in results)
        evaluated.append((passed, float(value), -worst_rank, candidate, results))
    _, _, _, selected, results = max(evaluated, key=lambda row: row[:3])
    return selected, tuple(results)
