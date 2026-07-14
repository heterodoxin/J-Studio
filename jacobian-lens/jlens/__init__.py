# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Jacobian lens: fit and apply the average input-output Jacobian as a readout
of decoder-transformer residuals."""

from jlens._logging import configure_logging
from jlens.baking import (
    ProjectionBake,
    ProjectionBakeRule,
    bake_projection,
    save_projection_bake,
)
from jlens.benchmark import (
    BenchmarkCase,
    BenchmarkResult,
    LayerReadoutRank,
    ReadoutBenchmarkResult,
    ReadoutCase,
    TokenRank,
    find_token_position_containing,
    ranked_default_cases,
    render_readout_html_report,
    run_benchmark,
    run_case,
    run_readout_benchmark,
    run_readout_case,
)
from jlens.benchmark import (
    render_html_report as render_benchmark_html,
)
from jlens.evaluation import (
    FitQuality,
    evaluate_fit_quality,
    select_readout_shrinkage,
    select_transport_shrinkage,
    standard_readout_cases,
)
from jlens.fitting import (
    calibrate_geometry,
    fit,
    fit_sketch,
    jacobian_for_prompt,
    sketched_jacobian_for_prompt,
)
from jlens.hf import HFLensModel, Layout, from_hf
from jlens.hooks import ActivationRecorder
from jlens.intervention import (
    CausalInterventionSolver,
    InterventionProposal,
    measure_dose_response,
    noop_control,
    norm_matched_random_control,
    shuffled_target_control,
)
from jlens.interventions import (
    ConceptResolver,
    ConceptSpec,
    InterventionEngine,
    InterventionResult,
    InterventionTrace,
    PhraseResidualOperator,
    ReadResult,
)
from jlens.jspace import TokenSetContrast
from jlens.lens import JacobianLens, SketchedJacobian
from jlens.operator import AdaptiveCausalOperator, OperatorConfig
from jlens.progressive import (
    DEFAULT_STAGES,
    FitStage,
    ProgressiveFitResult,
    StageResult,
    fit_progressive,
)
from jlens.protocol import LensModel

__all__ = [
    "ActivationRecorder",
    "AdaptiveCausalOperator",
    "BenchmarkCase",
    "BenchmarkResult",
    "CausalInterventionSolver",
    "ConceptResolver",
    "ConceptSpec",
    "HFLensModel",
    "FitQuality",
    "FitStage",
    "InterventionEngine",
    "InterventionProposal",
    "InterventionResult",
    "InterventionTrace",
    "JacobianLens",
    "LayerReadoutRank",
    "Layout",
    "LensModel",
    "OperatorConfig",
    "PhraseResidualOperator",
    "ProjectionBake",
    "ProjectionBakeRule",
    "ReadResult",
    "ReadoutBenchmarkResult",
    "ReadoutCase",
    "ProgressiveFitResult",
    "SketchedJacobian",
    "StageResult",
    "DEFAULT_STAGES",
    "TokenRank",
    "TokenSetContrast",
    "calibrate_geometry",
    "bake_projection",
    "save_projection_bake",
    "configure_logging",
    "fit",
    "fit_sketch",
    "fit_progressive",
    "find_token_position_containing",
    "from_hf",
    "jacobian_for_prompt",
    "measure_dose_response",
    "noop_control",
    "norm_matched_random_control",
    "ranked_default_cases",
    "render_benchmark_html",
    "render_readout_html_report",
    "evaluate_fit_quality",
    "select_transport_shrinkage",
    "select_readout_shrinkage",
    "standard_readout_cases",
    "run_benchmark",
    "run_case",
    "run_readout_benchmark",
    "run_readout_case",
    "shuffled_target_control",
    "sketched_jacobian_for_prompt",
]
