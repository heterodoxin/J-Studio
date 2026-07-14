from types import SimpleNamespace

import torch

import jlens.evaluation as evaluation
from jlens.evaluation import (
    FitQuality,
    select_readout_shrinkage,
    select_transport_shrinkage,
)
from jlens.lens import JacobianLens


def test_select_transport_shrinkage_uses_held_out_quality_and_keeps_largest_tie(
    monkeypatch,
):
    lens = JacobianLens({1: 2 * torch.eye(4)}, n_prompts=4, d_model=4)
    scores = {0.1: 0.4, 0.25: 0.8, 0.5: 0.8, 1.0: 0.2}

    def evaluate(model, candidate, validation_items, **kwargs):
        alpha = float(candidate.metadata["transport_shrinkage"])
        return FitQuality(
            scores[alpha],
            1.0,
            True,
            minimum_pass_at_10=0.0,
            minimum_rank_overlap=0.0,
        )

    monkeypatch.setattr(evaluation, "evaluate_fit_quality", evaluate)

    selected, quality = select_transport_shrinkage(
        object(), lens, [object()], candidates=(0.1, 0.25, 0.5, 1.0)
    )

    assert selected is not lens
    assert selected.metadata["transport_shrinkage"] == "0.5"
    assert quality.pass_at_10 == 0.8
    assert "transport_shrinkage" not in lens.metadata


def test_select_readout_shrinkage_prefers_largest_full_pass_candidate(monkeypatch):
    lens = JacobianLens({1: 2 * torch.eye(4)}, n_prompts=4, d_model=4)
    ranks = {
        0.2: (20, 30, 2),
        0.5: (40, 50, 3),
        0.75: (80, 90, 4),
        1.0: (150, 40, 1),
    }

    def run(model, candidate, cases):
        alpha = float(candidate.metadata["transport_shrinkage"])
        return tuple(
            SimpleNamespace(success=rank < 100, best_rank=rank)
            for rank in ranks[alpha]
        )

    monkeypatch.setattr(evaluation, "run_readout_benchmark", run)

    selected, results = select_readout_shrinkage(
        object(),
        lens,
        (object(), object(), object()),
        candidates=(0.2, 0.5, 0.75, 1.0),
    )

    assert selected.metadata["transport_shrinkage"] == "0.75"
    assert all(result.success for result in results)
