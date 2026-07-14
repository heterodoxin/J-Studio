# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import torch

from jlens.autotune import choose_batch


class FakeRunner:
    def __init__(self, outcomes):
        self.outcomes = outcomes

    def __call__(self, candidate):
        outcome = self.outcomes[candidate]
        if isinstance(outcome, BaseException):
            raise outcome
        return outcome


def test_autotuner_selects_fastest_safe_candidate():
    runner = FakeRunner(
        {1: (4.0, 0.4), 2: (2.5, 0.6), 4: (1.8, 0.85), 8: (1.2, 0.96)}
    )
    result = choose_batch((1, 2, 4, 8), runner, memory_fraction=0.9)
    assert result.batch_size == 4


def test_autotuner_recovers_from_oom():
    runner = FakeRunner({1: (4.0, 0.4), 2: torch.OutOfMemoryError()})
    assert choose_batch((1, 2), runner, memory_fraction=0.9).batch_size == 1
