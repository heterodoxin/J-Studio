# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import json

import torch

from jlens.intervention import (
    CausalInterventionSolver,
    InterventionProposal,
    measure_dose_response,
    noop_control,
    norm_matched_random_control,
    shuffled_target_control,
)
from jlens.jspace import TokenSetContrast
from jlens.lens import JacobianLens
from jlens.operator import AdaptiveCausalOperator, OperatorConfig
from jlens.trace import CausalTrace


def test_global_only_operator_matches_lens_transport_and_reports_fallback():
    jacobian = torch.tensor(
        [
            [1.0, 0.2, 0.0],
            [0.0, 0.7, 0.1],
            [0.3, 0.0, 1.2],
        ]
    )
    lens = JacobianLens({1: jacobian}, n_prompts=3, d_model=3)
    operator = AdaptiveCausalOperator(
        lens, OperatorConfig(model_id="tiny", operator_id="op-global")
    )
    residual = torch.tensor([0.4, -0.2, 0.8])

    torch.testing.assert_close(
        operator.matvec(layer=1, residual=residual),
        lens.transport(residual, 1),
    )
    torch.testing.assert_close(
        operator.rmatvec(layer=1, covector=residual),
        jacobian.T @ residual,
    )
    diagnostics = operator.diagnostics(layer=1, context="prompt")
    assert diagnostics.local_status == "global-only"
    assert diagnostics.alpha == 0.0
    assert diagnostics.evidence == "predicted"


def test_token_set_contrast_scores_and_target_vector():
    contrast = TokenSetContrast(
        name="correct-vs-wrong",
        positive_ids=(2, 4),
        negative_ids=(1,),
    )
    logits = torch.tensor([0.0, 0.5, 2.0, 0.0, 4.0])

    assert contrast.score(logits) == 2.5
    torch.testing.assert_close(
        contrast.target_vector(vocab_size=5),
        torch.tensor([0.0, -1.0, 0.5, 0.0, 0.5]),
    )


def test_regularized_solver_moves_target_while_preserving_protected_contrast():
    lens = JacobianLens({0: torch.eye(4)}, n_prompts=1, d_model=4)
    operator = AdaptiveCausalOperator(
        lens, OperatorConfig(model_id="tiny", operator_id="op-global")
    )
    readout = torch.eye(4)
    solver = CausalInterventionSolver(operator, readout)
    target = TokenSetContrast("target", positive_ids=(2,), negative_ids=(0,))
    protected = TokenSetContrast("protected", positive_ids=(1,), negative_ids=())

    proposal = solver.solve(
        layer=0,
        target=target,
        protected=(protected,),
        lambda_reg=0.05,
        protected_weight=100.0,
        max_iter=64,
    )
    effect = readout @ operator.matvec(layer=0, residual=proposal.delta)

    assert proposal.converged
    assert target.score(effect) > 0.5
    assert abs(protected.score(effect)) < 0.05
    assert proposal.condition_estimate >= 1.0
    assert proposal.norm > 0.0


def test_dose_response_records_measured_effects_and_controls():
    proposal = InterventionProposal(
        layer=2,
        delta=torch.tensor([3.0, 4.0]),
        predicted_effect=1.0,
        norm=5.0,
        converged=True,
        iterations=3,
        residual_error=0.01,
        condition_estimate=2.0,
    )

    trace = CausalTrace(
        model_id="tiny",
        prompt="2+2=",
        tokenization=(10, 11, 12),
        operator_id="op-global",
        layer=2,
        position=-1,
        seed=123,
        strength=1.0,
        target={"positive_ids": [2], "negative_ids": [0]},
        protected=[],
    )
    trace = measure_dose_response(
        trace,
        proposal,
        strengths=(0.0, 0.5, 1.0),
        measure=lambda delta: float(delta[0] - delta[1]),
        controls=(
            noop_control(),
            norm_matched_random_control(seed=7),
            shuffled_target_control(seed=8),
        ),
    )

    assert [point.strength for point in trace.dose_response] == [0.0, 0.5, 1.0]
    assert trace.dose_response[-1].observed_effect == -1.0
    assert {control.name for control in trace.controls} == {
        "noop",
        "norm-matched-random",
        "shuffled-target",
    }
    assert next(control for control in trace.controls if control.name == "noop").norm == 0.0

    decoded = CausalTrace.from_dict(json.loads(json.dumps(trace.to_dict())))
    assert decoded == trace
