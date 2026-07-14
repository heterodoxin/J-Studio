# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import torch

from jlens import calibrate_geometry
from jlens.geometry import CovarianceMetric
from jlens.lens import JacobianLens
from tests.tiny import TinyDecoder


def test_dense_transport_shrinkage_scales_only_the_fitted_correction():
    lens = JacobianLens(
        {1: 2.0 * torch.eye(4)},
        n_prompts=4,
        d_model=4,
        metadata={"transport_shrinkage": "0.25"},
    )
    residual = torch.tensor([[1.0, -2.0, 3.0, -4.0]])

    transported = lens.transport(residual, 1)

    torch.testing.assert_close(transported, 1.25 * residual)


def test_legacy_checkpoint_loads_with_uncalibrated_identity(tmp_path):
    path = tmp_path / "legacy.pt"
    torch.save(
        {"J": {0: torch.eye(3)}, "n_prompts": 1, "d_model": 3},
        path,
    )

    lens = JacobianLens.load(str(path))

    metric = lens.metric(0)
    assert not metric.calibrated
    torch.testing.assert_close(metric.diagonal, torch.ones(3))


def test_calibrated_geometry_round_trip(tmp_path):
    metric = CovarianceMetric(
        diagonal=torch.tensor([1.0, 2.0, 3.0]),
        factors=torch.ones(1, 3) * 0.1,
    )
    lens = JacobianLens(
        {0: torch.eye(3)},
        n_prompts=2,
        d_model=3,
        geometry={0: metric},
        metadata={"model": "tiny"},
    )
    path = tmp_path / "lens.pt"

    lens.save(str(path))
    loaded = JacobianLens.load(str(path))

    assert loaded.metadata == {"model": "tiny"}
    assert loaded.metric(0).calibrated
    torch.testing.assert_close(
        loaded.metric(0).diagonal, metric.diagonal, atol=2e-3, rtol=0
    )
    torch.testing.assert_close(
        loaded.metric(0).factors, metric.factors, atol=2e-3, rtol=0
    )


def test_constructor_rejects_geometry_shape_mismatch():
    metric = CovarianceMetric.identity(4, calibrated=True)

    try:
        JacobianLens({0: torch.eye(3)}, n_prompts=1, d_model=3, geometry={0: metric})
    except ValueError as exc:
        assert "geometry" in str(exc)
    else:
        raise AssertionError("shape mismatch was accepted")


def test_calibrate_geometry_produces_positive_bounded_rank_metric():
    model = TinyDecoder(n_layers=4, d_model=8)
    lens = JacobianLens({0: torch.eye(8), 2: torch.eye(8)}, n_prompts=1, d_model=8)
    prompts = ["abcdefghij " * 3, "klmnopqrst " * 3]

    calibrated = calibrate_geometry(
        model,
        lens,
        prompts,
        max_seq_len=32,
        rank=2,
        shrinkage=0.1,
    )

    assert calibrated.metadata["geometry_prompts"] == "2"
    for layer in lens.source_layers:
        metric = calibrated.metric(layer)
        assert metric.calibrated
        assert (metric.diagonal > 0).all()
        assert metric.factors is not None
        assert metric.factors.shape[0] <= 2
        probe = torch.randn(8)
        assert torch.dot(probe, metric.apply(probe)) > 0
