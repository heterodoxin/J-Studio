# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""J-space token contrasts and concept utilities."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class TokenSetContrast:
    """A verbalizable target expressed as positive and negative token sets."""

    name: str
    positive_ids: tuple[int, ...]
    negative_ids: tuple[int, ...] = ()

    def __init__(
        self,
        name: str,
        positive_ids: Sequence[int],
        negative_ids: Sequence[int] = (),
    ) -> None:
        positive = tuple(int(token_id) for token_id in positive_ids)
        negative = tuple(int(token_id) for token_id in negative_ids)
        if not name:
            raise ValueError("contrast name must not be empty")
        if not positive and not negative:
            raise ValueError("contrast needs at least one token id")
        if any(token_id < 0 for token_id in (*positive, *negative)):
            raise ValueError("token ids must be non-negative")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "positive_ids", positive)
        object.__setattr__(self, "negative_ids", negative)

    def target_vector(self, *, vocab_size: int, device=None) -> torch.Tensor:
        if vocab_size <= 0:
            raise ValueError("vocab_size must be positive")
        ids = (*self.positive_ids, *self.negative_ids)
        if ids and max(ids) >= vocab_size:
            raise ValueError("token id out of vocabulary range")
        vector = torch.zeros(vocab_size, dtype=torch.float32, device=device)
        if self.positive_ids:
            vector[list(self.positive_ids)] += 1.0 / len(self.positive_ids)
        if self.negative_ids:
            vector[list(self.negative_ids)] -= 1.0 / len(self.negative_ids)
        return vector

    def score(self, logits: torch.Tensor) -> float:
        value = logits.float()
        if value.ndim != 1:
            raise ValueError("logits must be a rank-1 tensor")
        if max((*self.positive_ids, *self.negative_ids)) >= value.shape[0]:
            raise ValueError("token id out of vocabulary range")
        score = torch.tensor(0.0, device=value.device)
        if self.positive_ids:
            score = score + value[list(self.positive_ids)].mean()
        if self.negative_ids:
            score = score - value[list(self.negative_ids)].mean()
        return float(score)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "positive_ids": list(self.positive_ids),
            "negative_ids": list(self.negative_ids),
        }
