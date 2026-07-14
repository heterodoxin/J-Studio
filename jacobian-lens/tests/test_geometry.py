# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import torch

from jlens.geometry import (
    CovarianceMetric,
    minimum_cost_perturbation,
    minimum_passing_scale,
)


def test_single_constraint_matches_closed_form():
    sigma = torch.tensor([[2.0, 0.3], [0.3, 1.0]], dtype=torch.float64)
    metric = CovarianceMetric.from_dense(sigma)
    constraints = torch.tensor([[1.0, -0.5]], dtype=torch.float64)
    deficits = torch.tensor([0.7], dtype=torch.float64)

    result = minimum_cost_perturbation(constraints, deficits, metric)

    expected = (
        deficits[0]
        / (constraints @ sigma @ constraints.T)[0, 0]
        * (sigma @ constraints[0])
    )
    assert result.feasible
    torch.testing.assert_close(result.delta, expected, atol=1e-7, rtol=1e-7)


def test_collinear_constraints_stay_finite_and_feasible():
    constraints = torch.tensor([[1.0, 0.0], [2.0, 0.0]])
    deficits = torch.tensor([1.0, 2.0])

    result = minimum_cost_perturbation(
        constraints, deficits, CovarianceMetric.identity(2)
    )

    assert result.feasible
    assert torch.isfinite(result.delta).all()
    assert torch.all(constraints @ result.delta >= deficits - 1e-5)


def test_negative_preservation_constraint_can_become_active():
    # x >= 1 would choose (1, 0). The preservation constraint x + y <= 0.2
    # must remain in the solve even though it is satisfied at delta=0.
    constraints = torch.tensor([[1.0, 0.0], [-1.0, -1.0]])
    deficits = torch.tensor([1.0, -0.2])

    result = minimum_cost_perturbation(
        constraints, deficits, CovarianceMetric.identity(2)
    )

    assert result.feasible
    assert result.delta[0] >= 1.0 - 1e-5
    assert result.delta.sum() <= 0.2 + 1e-5


def test_metric_applies_diagonal_plus_low_rank_covariance():
    metric = CovarianceMetric(
        diagonal=torch.tensor([2.0, 3.0]),
        factors=torch.tensor([[1.0, -1.0]]),
    )
    value = torch.tensor([2.0, 4.0])
    expected_covariance = (
        torch.diag(metric.diagonal) + metric.factors.T @ metric.factors
    )

    torch.testing.assert_close(metric.apply(value), value @ expected_covariance)


def test_scale_search_returns_lower_passing_boundary():
    result = minimum_passing_scale(lambda scale: scale >= 0.37, relative_tolerance=1e-3)

    assert result.passed
    assert 0.37 <= result.scale <= 0.371
    assert result.evaluations[0].scale == 0.0


def test_scale_search_reports_bounded_failure():
    result = minimum_passing_scale(lambda _scale: False, maximum=4.0)

    assert not result.passed
    assert result.scale == 4.0
    assert result.evaluations[-1].scale == 4.0
