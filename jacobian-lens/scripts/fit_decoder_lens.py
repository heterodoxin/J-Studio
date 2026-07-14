#!/usr/bin/env python3
"""Fit a quality-gated progressive Jacobian lens for an HF decoder."""

from __future__ import annotations

import argparse
import json
import logging
import os
from collections.abc import Sequence
from dataclasses import asdict, replace
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import jlens
from jlens.autotune import autotune_vjp_batch
from jlens.examples import load_wikitext_prompts


def evaluation_items(root: Path) -> list[dict]:
    items = []
    for path in sorted((root / "data" / "evaluations").glob("*.json")):
        value = json.loads(path.read_text(encoding="utf-8"))
        items.extend(item for item in value.get("items", ()) if "prompt" in item)
    return items


def select_source_layers(
    target_layer: int, *, limit: int = 25, layer_step: int | None = None
) -> list[int]:
    """Select at most ``limit`` evenly spaced layers below the target."""
    if target_layer <= 0:
        raise ValueError("target_layer must leave at least one source layer")
    if layer_step is not None:
        if layer_step <= 0:
            raise ValueError("layer_step must be positive")
        layers = list(range(0, target_layer, layer_step))
        if target_layer - 1 not in layers:
            layers.append(target_layer - 1)
        return layers[-limit:]
    count = min(limit, target_layer)
    if count == 1:
        return [target_layer - 1]
    return sorted(
        {
            round(index * (target_layer - 1) / (count - 1))
            for index in range(count)
        }
    )


def atomic_json(path: Path | None, value: object) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="heterodoxin/qwen3-8b-apostate")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--stage", choices=("preview", "stable", "refined"), default="stable")
    parser.add_argument(
        "--prompt-source",
        choices=("evaluations", "wikitext"),
        default="evaluations",
        help="Fitting prompt source; wikitext better matches the original lens recipe.",
    )
    parser.add_argument("--prompts", type=int)
    parser.add_argument("--rank", type=int)
    parser.add_argument("--dim-batch", type=int)
    parser.add_argument("--max-seq-len", type=int)
    parser.add_argument("--skip-first", type=int, default=16)
    parser.add_argument("--layer-step", type=int)
    parser.add_argument("--quality-json", type=Path)
    parser.add_argument("--progress-json", type=Path)
    parser.add_argument("--allow-download", action="store_true")
    return parser.parse_args(argv)


def _requested_stages(args: argparse.Namespace) -> tuple[jlens.FitStage, ...]:
    stages = list(jlens.DEFAULT_STAGES)
    if args.stage == "preview":
        stages = stages[:1]
    elif args.stage == "refined":
        stages.append(jlens.FitStage("Refined", 64, 96, 3, 128))
    final = stages[-1]
    stages[-1] = replace(
        final,
        prompts=args.prompts or final.prompts,
        sketch_rank=args.rank or final.sketch_rank,
        max_seq_len=args.max_seq_len or final.max_seq_len,
    )
    return tuple(stages)


def main(argv: Sequence[str] | None = None) -> None:
    args = arguments(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    if not torch.cuda.is_available() or torch.version.hip is None:
        raise SystemExit("ROCm PyTorch is required")
    tokenizer = AutoTokenizer.from_pretrained(
        args.model, local_files_only=not args.allow_download
    )
    hf_model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        local_files_only=not args.allow_download,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    ).to("cuda:0")
    model = jlens.from_hf(hf_model, tokenizer)
    target_layer = model.n_layers - 2
    layers = select_source_layers(
        target_layer, layer_step=args.layer_step
    )
    stages = _requested_stages(args)

    all_items = evaluation_items(Path(__file__).parents[1])
    prompt_count = max(stage.prompts for stage in stages)
    if len(all_items) <= prompt_count:
        raise SystemExit("not enough evaluation prompts for disjoint fit and validation")
    if args.prompt_source == "wikitext":
        prompts = load_wikitext_prompts(prompt_count)
    else:
        prompts = [item["prompt"] for item in all_items[:prompt_count]]
    validation_items = all_items[prompt_count : prompt_count + 32]
    dim_batch = args.dim_batch
    if dim_batch is None:
        tuning = autotune_vjp_batch(
            model,
            prompts[0],
            layers,
            max_seq_len=stages[-1].max_seq_len,
            target_layer=target_layer,
        )
        dim_batch = tuning.batch_size

    args.output.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = args.output.parent / f".{args.output.stem}.fit"
    atomic_json(
        args.progress_json,
        {"status": "fitting", "stage": stages[0].name, "completed_stages": 0},
    )

    quality_rows: list[dict] = []

    def stage_complete(result: jlens.StageResult) -> None:
        quality_rows.append(
            {
                "stage": result.name,
                "elapsed_seconds": result.elapsed_seconds,
                **asdict(result.quality),
            }
        )
        atomic_json(args.quality_json, {"stages": quality_rows})
        atomic_json(
            args.progress_json,
            {
                "status": "fitting" if result.name != stages[-1].name else "complete",
                "stage": result.name,
                "completed_stages": len(quality_rows),
                "total_stages": len(stages),
            },
        )

    def fit_progress(stage: jlens.FitStage, progress: dict[str, int]) -> None:
        atomic_json(
            args.progress_json,
            {
                "status": "fitting",
                "stage": stage.name,
                "completed_stages": len(quality_rows),
                **progress,
            },
        )
        atomic_json(args.quality_json, {"stages": quality_rows})

    result = jlens.fit_progressive(
        model,
        prompts,
        validation_items,
        stages=stages,
        source_layers=layers,
        target_layer=target_layer,
        dim_batch=dim_batch,
        skip_first=args.skip_first,
        checkpoint_dir=str(checkpoint_dir),
        on_stage=stage_complete,
        on_progress=fit_progress,
    )
    lens = result.active.lens
    lens.metadata.update(
        {
            "model": args.model,
            "precision": "BF16",
            "backend": "ROCm",
            "quality_stage": result.active.name,
            "quality_gate_version": "jspace-v1",
            "fit_quality_pass_at_10": f"{result.active.quality.pass_at_10:.6g}",
            "fit_quality_rank_overlap": f"{result.active.quality.rank_overlap:.6g}",
        }
    )
    lens = jlens.calibrate_geometry(
        model,
        lens,
        prompts[: min(16, len(prompts))],
        max_seq_len=result.active.stage.max_seq_len,
        rank=min(16, args.rank or result.active.stage.sketch_rank),
    )
    lens.save(str(args.output))
    print(args.output)


if __name__ == "__main__":
    main()
