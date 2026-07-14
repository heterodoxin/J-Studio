# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Progressive, quality-gated Jacobian-lens fitting."""

from __future__ import annotations

import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from jlens.evaluation import FitQuality, evaluate_fit_quality
from jlens.fitting import fit_sketch
from jlens.lens import JacobianLens
from jlens.protocol import LensModel


@dataclass(frozen=True)
class FitStage:
    name: str
    prompts: int
    sketch_rank: int
    probe_blocks: int
    max_seq_len: int


DEFAULT_STAGES = (
    FitStage("Preview", 8, 32, 1, 64),
    FitStage("Stable", 32, 64, 2, 128),
)


@dataclass(frozen=True)
class StageResult:
    stage: FitStage
    lens: JacobianLens
    quality: FitQuality
    elapsed_seconds: float
    checkpoint_path: str | None

    @property
    def name(self) -> str:
        return self.stage.name


@dataclass(frozen=True)
class ProgressiveFitResult:
    stages: tuple[StageResult, ...]
    active: StageResult


def fit_progressive(
    model: LensModel,
    prompts: Sequence[str],
    validation_items: Sequence[Any],
    *,
    stages: Sequence[FitStage] = DEFAULT_STAGES,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 4,
    seed: int = 0,
    skip_first: int = 16,
    checkpoint_dir: str | None = None,
    evaluator: Callable[[JacobianLens, JacobianLens | None], FitQuality] | None = None,
    on_stage: Callable[[StageResult], None] | None = None,
    on_progress: Callable[[FitStage, dict[str, int]], None] | None = None,
    operation_context: Callable[[str], AbstractContextManager] | None = None,
) -> ProgressiveFitResult:
    """Fit Preview then Stable, retaining Preview when Stable misses its gates."""
    if not stages:
        raise ValueError("at least one fit stage is required")
    if not prompts:
        raise ValueError("at least one fitting prompt is required")
    target = model.n_layers - 2 if target_layer is None else target_layer
    root = Path(checkpoint_dir) if checkpoint_dir else None
    if root is not None:
        root.mkdir(parents=True, exist_ok=True)
    evaluate = evaluator or (
        lambda lens, previous: evaluate_fit_quality(
            model, lens, validation_items, previous
        )
    )

    completed: list[StageResult] = []
    active: StageResult | None = None
    previous_lens: JacobianLens | None = None
    for stage_index, stage in enumerate(stages):
        started = time.perf_counter()
        slug = stage.name.lower().replace(" ", "-")
        fit_checkpoint = str(root / f"{slug}.fit.pt") if root else None
        lens_path = str(root / f"{slug}.lens.pt") if root else None
        lens = fit_sketch(
            model,
            prompts[: stage.prompts],
            source_layers=source_layers,
            sketch_rank=min(stage.sketch_rank, model.d_model),
            probe_blocks=stage.probe_blocks,
            target_layer=target,
            dim_batch=dim_batch,
            max_seq_len=stage.max_seq_len,
            skip_first=skip_first,
            seed=seed,
            checkpoint_path=fit_checkpoint,
            on_progress=(
                None
                if on_progress is None
                else lambda progress, stage=stage: on_progress(stage, progress)
            ),
            operation_context=operation_context,
        )
        quality = evaluate(lens, previous_lens)
        if lens_path:
            lens.save(lens_path)
        result = StageResult(
            stage=stage,
            lens=lens,
            quality=quality,
            elapsed_seconds=time.perf_counter() - started,
            checkpoint_path=lens_path,
        )
        completed.append(result)
        if stage_index == 0 or quality.stable:
            active = result
        if on_stage is not None:
            on_stage(result)
        previous_lens = lens

    assert active is not None
    return ProgressiveFitResult(tuple(completed), active)
