#!/usr/bin/env python3.11
# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Run prompt-context J-lens viewing benchmarks and write an HTML report."""

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
    parser.add_argument("--out", required=True, help="Output directory")
    parser.add_argument("--local-files-only", action="store_true")
    parser.add_argument(
        "--max-rank",
        type=int,
        default=100,
        help="Pass threshold: target must appear above this rank in some layer",
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


def _default_cases(model, max_rank: int) -> tuple[jlens.ReadoutCase, ...]:
    return jlens.standard_readout_cases(model, max_rank=max_rank)


def main() -> int:
    args = _arguments()
    model = _load_model(args.model, local_files_only=args.local_files_only)
    lens = jlens.JacobianLens.load(args.lens)
    cases = _default_cases(model, args.max_rank)
    results = jlens.run_readout_benchmark(model, lens, cases)
    passed = sum(1 for result in results if result.success)

    out_dir = Path(args.out).expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "results.json").write_text(
        json.dumps([result.to_dict() for result in results], indent=2),
        encoding="utf-8",
    )
    (out_dir / "index.html").write_text(
        jlens.render_readout_html_report(
            results,
            title=f"J-Lens Viewing Benchmark — {args.model}",
        ),
        encoding="utf-8",
    )
    print(f"wrote {out_dir / 'index.html'}")
    print(f"passed {passed}/{len(results)} viewing cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
