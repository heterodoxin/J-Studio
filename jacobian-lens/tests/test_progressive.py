# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import pytest

from jlens.evaluation import FitQuality
from jlens.progressive import FitStage, fit_progressive

from .tiny import TinyDecoder


def test_stable_requires_j_space_recovery():
    quality = FitQuality(
        pass_at_10=0.4,
        rank_overlap=0.9,
        finite=True,
    )
    assert quality.stable
    assert not quality.reasons


def test_low_j_space_recovery_is_not_stable():
    quality = FitQuality(
        pass_at_10=0.1,
        rank_overlap=0.9,
        finite=True,
    )

    assert not quality.stable
    assert quality.reasons == (
        "held-out J-space pass@10 is below 0.30",
    )


def test_non_finite_fit_is_not_stable():
    quality = FitQuality(
        pass_at_10=0.4,
        rank_overlap=0.9,
        finite=False,
    )

    assert not quality.stable
    assert quality.reasons == ("transport or quality metrics are not finite",)


@pytest.fixture
def tiny_progressive_inputs():
    return (
        TinyDecoder(n_layers=4, d_model=8),
        ["abcdefghij " * 5, "klmnopqrst " * 5, "uvwxyzabcd " * 5],
        [],
    )


def test_progressive_callback_receives_preview_then_stable(
    tiny_progressive_inputs,
):
    seen = []
    result = fit_progressive(
        *tiny_progressive_inputs,
        source_layers=[1],
        stages=(
            FitStage("Preview", 2, 4, 1, 32),
            FitStage("Stable", 3, 4, 2, 64),
        ),
        evaluator=lambda lens, previous: FitQuality(0.8, 1.0, True),
        on_stage=lambda stage: seen.append(stage.name),
    )
    assert seen == ["Preview", "Stable"]
    assert result.active.name == "Stable"


def test_failed_stable_retains_preview(tiny_progressive_inputs):
    qualities = iter(
        [FitQuality(0.5, 1.0, True), FitQuality(0.1, 1.0, True)]
    )
    result = fit_progressive(
        *tiny_progressive_inputs,
        source_layers=[1],
        stages=(
            FitStage("Preview", 2, 4, 1, 32),
            FitStage("Stable", 3, 4, 1, 64),
        ),
        evaluator=lambda lens, previous: next(qualities),
    )
    assert result.active.name == "Preview"
    assert result.stages[-1].quality.stable is False
