"""Tokenizer-sequence geometry for multi-token J-space interventions."""

from __future__ import annotations

import math

import torch


def normalize_directions(directions: torch.Tensor) -> torch.Tensor:
    """Return finite row-wise unit directions without silently dropping rows."""
    if directions.ndim != 2 or not directions.shape[0] or not directions.shape[1]:
        raise ValueError("directions must have shape [tokens, d_model]")
    values = directions.detach().float()
    if not torch.isfinite(values).all():
        raise ValueError("directions must be finite")
    norms = values.norm(dim=1, keepdim=True)
    if torch.any(norms <= 1e-8):
        raise ValueError("directions contain a zero row")
    return values / norms


def compact_basis(
    directions: torch.Tensor, *, relative_tolerance: float = 1e-5
) -> torch.Tensor:
    """Orthonormal basis spanning token directions, with duplicates removed."""
    if not math.isfinite(relative_tolerance) or not 0 < relative_tolerance < 1:
        raise ValueError("relative_tolerance must lie in (0, 1)")
    normalized = normalize_directions(directions)
    _, singular_values, vh = torch.linalg.svd(normalized, full_matrices=False)
    keep = singular_values > singular_values.max() * relative_tolerance
    if not torch.any(keep):
        raise ValueError("phrase directions have rank zero")
    return vh[keep].T.contiguous()


def sequence_alignment(source_count: int, target_count: int) -> torch.Tensor:
    """Map ordered source coefficients onto an unequal-length target sequence.

    Columns correspond to source tokens and sum to one, preventing phrase
    length from multiplying the intervention norm.
    """
    if source_count <= 0 or target_count <= 0:
        raise ValueError("source_count and target_count must be positive")
    if source_count == target_count:
        return torch.eye(source_count)
    alignment = torch.zeros(target_count, source_count)
    if target_count == 1:
        alignment[0] = 1.0
        return alignment
    if source_count == 1:
        alignment[:, 0] = 1.0 / target_count
        return alignment
    for source_index in range(source_count):
        coordinate = source_index * (target_count - 1) / (source_count - 1)
        lower = int(math.floor(coordinate))
        upper = int(math.ceil(coordinate))
        if lower == upper:
            alignment[lower, source_index] = 1.0
        else:
            alignment[lower, source_index] = upper - coordinate
            alignment[upper, source_index] = coordinate - lower
    return alignment
