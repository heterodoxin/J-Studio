#!/usr/bin/env python3.11
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Run a strict J-space intervention benchmark and write an HTML report."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch
import transformers

import jlens


def _arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, help="Hugging Face model id/path")
    parser.add_argument("--lens", required=True, help="Path to a JacobianLens .pt file")
    parser.add_argument(
        "--out",
        required=True,
        help="Output directory for index.html and results.json",
    )
    parser.add_argument(
        "--local-files-only",
        action="store_true",
        help="Load model/tokenizer from the local Hugging Face cache only",
    )
    parser.add_argument(
        "--case-set",
        choices=("default",),
        default="default",
        help="Benchmark case set to run",
    )
    parser.add_argument(
        "--max-scale",
        type=float,
        default=16.0,
        help="Maximum residual/J-space search budget per case",
    )
    return parser.parse_args()


def _load_model(model_id: str, *, local_files_only: bool):
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    hf_model = transformers.AutoModelForCausalLM.from_pretrained(
        model_id,
        dtype=dtype,
        local_files_only=local_files_only,
        low_cpu_mem_usage=True,
        attn_implementation="sdpa",
    )
    if torch.cuda.is_available():
        hf_model = hf_model.to("cuda:0")
    hf_model.eval()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        model_id,
        local_files_only=local_files_only,
    )
    return jlens.from_hf(hf_model, tokenizer)


def main() -> int:
    args = _arguments()
    model = _load_model(args.model, local_files_only=args.local_files_only)
    lens = jlens.JacobianLens.load(args.lens)
    cases = jlens.ranked_default_cases(model, maximum_scale=args.max_scale)
    results = jlens.run_benchmark(model, lens, cases)
    passed = sum(1 for result in results if result.success)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps([result.to_dict() for result in results], indent=2),
        encoding="utf-8",
    )
    (out_dir / "index.html").write_text(
        jlens.render_benchmark_html(
            results,
            title=f"J-Lens Intervention Benchmark — {args.model}",
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'index.html'}")
    print(f"passed {passed}/{len(results)} strict intervention cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
