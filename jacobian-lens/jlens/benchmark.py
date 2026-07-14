# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Strict causal benchmarks for J-space interventions.

The benchmark never generates text and never applies fallback edits. It runs one
forward pass over the supplied context, asks :class:`InterventionEngine` for an
edit, and applies that edit only when the engine reports success.
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import asdict, dataclass
from html import escape
from typing import Literal

import torch

from jlens.hooks import ActivationRecorder
from jlens.interventions import ConceptResolver, InterventionEngine
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

Operation = Literal["inject", "replace", "suppress"]


@dataclass(frozen=True)
class TokenRank:
    token_id: int
    token: str
    rank: int
    logit: float


@dataclass(frozen=True)
class BenchmarkCase:
    name: str
    operation: Operation
    prompt: str
    target: str | int
    source: str | int | None = None
    layers: tuple[int, ...] = ()
    position: int = -1
    top_k: int = 1
    maximum_scale: float = 16.0
    context_mode: str = "prompt-only"


@dataclass(frozen=True)
class BenchmarkResult:
    case: BenchmarkCase
    success: bool
    applied: bool
    target_before: TokenRank
    target_after: TokenRank
    source_before: TokenRank | None
    source_after: TokenRank | None
    top_before: tuple[TokenRank, ...]
    top_after: tuple[TokenRank, ...]
    selected_layer: int | None
    selected_scale: float
    normalized_cost: float
    trace: dict

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class ReadoutCase:
    name: str
    prompt: str
    target: str | int
    position: int
    layers: tuple[int, ...] = ()
    max_rank: int = 100
    context_mode: str = "prompt-only"


@dataclass(frozen=True)
class LayerReadoutRank:
    layer: int
    token_id: int
    token: str
    rank: int
    logit: float


@dataclass(frozen=True)
class ReadoutBenchmarkResult:
    case: ReadoutCase
    success: bool
    target_id: int
    target_ids: tuple[int, ...]
    target_token: str
    resolved_position: int
    best_rank: int
    best_layer: int
    layer_ranks: tuple[LayerReadoutRank, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def next_token_logits(model: LensModel, prompt: str, editor=None) -> torch.Tensor:
    """Return final-layer next-token logits for a supplied context."""
    final_layer = model.n_layers - 1
    input_ids = model.encode(prompt, max_length=512)
    edit_context = editor if editor is not None else nullcontext()
    with torch.no_grad(), edit_context:
        with ActivationRecorder(model.layers, at=[final_layer]) as recorder:
            model.forward(input_ids)
    residual = recorder.activations[final_layer][0, -1].detach().float()
    return model.unembed(residual).detach().float().cpu()


def rank_token(logits: torch.Tensor, token_id: int, tokenizer=None) -> TokenRank:
    if logits.ndim != 1:
        raise ValueError("logits must have shape [vocab_size]")
    token_id = int(token_id)
    if not 0 <= token_id < logits.shape[0]:
        raise ValueError("token_id out of vocabulary range")
    rank = int((logits > logits[token_id]).sum().item())
    token = str(token_id) if tokenizer is None else tokenizer.decode([token_id])
    return TokenRank(
        token_id=token_id,
        token=token,
        rank=rank,
        logit=float(logits[token_id]),
    )


def _top(logits: torch.Tensor, tokenizer, *, n: int = 8) -> tuple[TokenRank, ...]:
    values, indices = logits.topk(min(n, logits.shape[0]))
    del values
    return tuple(rank_token(logits, int(token_id), tokenizer) for token_id in indices)


def _resolve_single(resolver: ConceptResolver, value: str | int) -> int:
    spec = resolver.resolve(value)
    if len(spec.token_ids) != 1:
        raise ValueError("benchmark concepts must resolve to one token")
    return spec.token_ids[0]


def _resolve_single_token_variants(
    resolver: ConceptResolver, value: str | int
) -> tuple[int, ...]:
    if isinstance(value, int):
        return (int(value),)
    spec = resolver.resolve(value)
    variants = tuple(
        dict.fromkeys(variant[0] for variant in spec.variants if len(variant) == 1)
    )
    if not variants:
        raise ValueError("benchmark concepts must have at least one token variant")
    return variants


def _benchmark_success(
    case: BenchmarkCase,
    *,
    engine_success: bool,
    target_before: TokenRank,
    target_after: TokenRank,
    source_before: TokenRank | None,
    source_after: TokenRank | None,
) -> tuple[bool, str]:
    if not engine_success:
        return False, "engine did not find a passing residual/J-space edit"
    if case.operation == "inject":
        if target_before.rank < case.top_k:
            return False, "target already met injection criterion before intervention"
        if target_after.rank < case.top_k:
            return True, "target entered requested top-k"
        return False, "target did not enter requested top-k after intervention"
    if case.operation == "replace":
        if source_before is None or source_after is None:
            return False, "replacement requires source rank measurements"
        if source_before.rank >= target_before.rank:
            return False, "source must outrank target before replacement"
        if target_after.rank < source_after.rank:
            return True, "target overtook source"
        return False, "target did not overtake source after intervention"
    if case.operation == "suppress":
        if target_after.rank >= case.top_k:
            return True, "target left requested top-k"
        return False, "target remained inside requested top-k"
    return False, f"unknown operation {case.operation!r}"


def run_case(
    model: LensModel,
    lens: JacobianLens,
    case: BenchmarkCase,
) -> BenchmarkResult:
    """Run one strict intervention case and measure actual next-token ranks."""
    engine = InterventionEngine(model, lens)
    resolver = engine.resolver
    target_id = _resolve_single(resolver, case.target)
    source_id = _resolve_single(resolver, case.source) if case.source is not None else None
    layers = case.layers or tuple(lens.source_layers)
    options = {
        "layers": layers,
        "positions": (case.position,),
        "maximum_scale": case.maximum_scale,
    }

    baseline_logits = next_token_logits(model, case.prompt)
    if case.operation == "inject":
        result = engine.inject(case.prompt, case.target, top_k=case.top_k, **options)
    elif case.operation == "replace":
        if case.source is None:
            raise ValueError("replace benchmark requires source")
        result = engine.replace(case.prompt, case.source, case.target, **options)
    elif case.operation == "suppress":
        result = engine.suppress(case.prompt, case.target, top_k=case.top_k, **options)
    else:
        raise ValueError(f"unknown operation {case.operation!r}")

    editor = engine.apply(result) if result.success else nullcontext()
    edited_logits = next_token_logits(model, case.prompt, editor)

    target_before = rank_token(baseline_logits, target_id, model.tokenizer)
    target_after = rank_token(edited_logits, target_id, model.tokenizer)
    source_before = (
        None
        if source_id is None
        else rank_token(baseline_logits, source_id, model.tokenizer)
    )
    source_after = (
        None
        if source_id is None
        else rank_token(edited_logits, source_id, model.tokenizer)
    )
    benchmark_success, benchmark_reason = _benchmark_success(
        case,
        engine_success=result.success,
        target_before=target_before,
        target_after=target_after,
        source_before=source_before,
        source_after=source_after,
    )

    trace = result.trace.to_dict()
    trace["message"] = result.message
    trace["benchmark_success"] = benchmark_success
    trace["benchmark_reason"] = benchmark_reason
    return BenchmarkResult(
        case=case,
        success=benchmark_success,
        applied=result.success,
        target_before=target_before,
        target_after=target_after,
        source_before=source_before,
        source_after=source_after,
        top_before=_top(baseline_logits, model.tokenizer),
        top_after=_top(edited_logits, model.tokenizer),
        selected_layer=result.trace.selected_layer,
        selected_scale=result.trace.selected_scale,
        normalized_cost=result.trace.normalized_cost,
        trace=trace,
    )


def run_benchmark(
    model: LensModel,
    lens: JacobianLens,
    cases: tuple[BenchmarkCase, ...],
) -> tuple[BenchmarkResult, ...]:
    return tuple(run_case(model, lens, case) for case in cases)


def find_token_position_containing(
    model: LensModel,
    prompt: str,
    substring: str,
    *,
    occurrence: int = 0,
    max_seq_len: int = 512,
) -> int:
    """Return the token position whose decoded token contains ``substring``."""
    if not substring:
        raise ValueError("substring must be non-empty")
    input_ids = model.encode(prompt, max_length=max_seq_len)[0].tolist()
    seen = 0
    for index, token_id in enumerate(input_ids):
        decoded = model.tokenizer.decode(
            [token_id], clean_up_tokenization_spaces=False
        )
        if substring in decoded:
            if seen == occurrence:
                return index
            seen += 1
    raise ValueError(f"substring {substring!r} not found in decoded context tokens")


def run_readout_case(
    model: LensModel,
    lens: JacobianLens,
    case: ReadoutCase,
) -> ReadoutBenchmarkResult:
    resolver = ConceptResolver(model.tokenizer)
    target_ids = _resolve_single_token_variants(resolver, case.target)
    layers = case.layers or tuple(lens.source_layers)
    input_ids = model.encode(case.prompt, max_length=512)
    seq_len = input_ids.shape[1]
    position = case.position + seq_len if case.position < 0 else case.position
    if not 0 <= position < seq_len:
        raise IndexError(f"position {case.position} out of range for length {seq_len}")

    logits_by_layer, _, _ = lens.apply(
        model,
        case.prompt,
        layers=layers,
        positions=(position,),
    )
    layer_rows = []
    for layer, logits in sorted(logits_by_layer.items()):
        best_token = min(
            (rank_token(logits[0], target_id, model.tokenizer) for target_id in target_ids),
            key=lambda item: item.rank,
        )
        layer_rows.append(
            LayerReadoutRank(
                layer=layer,
                token_id=best_token.token_id,
                token=best_token.token,
                rank=best_token.rank,
                logit=best_token.logit,
            )
    )
    layer_ranks = tuple(layer_rows)
    best = min(layer_ranks, key=lambda item: item.rank)
    return ReadoutBenchmarkResult(
        case=case,
        success=best.rank < case.max_rank,
        target_id=best.token_id,
        target_ids=target_ids,
        target_token=best.token,
        resolved_position=position,
        best_rank=best.rank,
        best_layer=best.layer,
        layer_ranks=layer_ranks,
    )


def run_readout_benchmark(
    model: LensModel,
    lens: JacobianLens,
    cases: tuple[ReadoutCase, ...],
) -> tuple[ReadoutBenchmarkResult, ...]:
    return tuple(run_readout_case(model, lens, case) for case in cases)


def ranked_default_cases(
    model: LensModel,
    *,
    prompt: str = "Complete with exactly one word: The secret word is",
    maximum_scale: float = 16.0,
) -> tuple[BenchmarkCase, ...]:
    """Choose non-trivial default cases from the prompt's baseline ranks.

    This avoids hard-coded token strings whose rank depends strongly on whether
    the context is raw text, chat-formatted text, or a transcript.
    """
    logits = next_token_logits(model, prompt)
    top_ids = [int(token_id) for token_id in logits.topk(min(6, logits.numel())).indices]
    if len(top_ids) < 3:
        raise ValueError("model vocabulary is too small for ranked default cases")
    target = top_ids[min(4, len(top_ids) - 1)]
    second_target = top_ids[min(5, len(top_ids) - 1)]
    if second_target == target:
        second_target = top_ids[-2]
    return (
        BenchmarkCase(
            name=f"inject baseline-rank-{top_ids.index(target) + 1}",
            operation="inject",
            prompt=prompt,
            target=target,
            maximum_scale=maximum_scale,
        ),
        BenchmarkCase(
            name="replace top-1 with baseline lower-ranked token",
            operation="replace",
            prompt=prompt,
            source=top_ids[0],
            target=target,
            maximum_scale=maximum_scale,
        ),
        BenchmarkCase(
            name="replace top-2 with baseline lower-ranked token",
            operation="replace",
            prompt=prompt,
            source=top_ids[1],
            target=second_target,
            maximum_scale=maximum_scale,
        ),
    )


def _rank_text(rank: TokenRank | None) -> str:
    if rank is None:
        return "—"
    return f"#{rank.rank + 1} {escape(rank.token)!r} ({rank.logit:.3g})"


def _rank_bar(before: TokenRank, after: TokenRank) -> str:
    before_width = max(2, min(100, 100 - before.rank))
    after_width = max(2, min(100, 100 - after.rank))
    color = "#16a34a" if after.rank < before.rank else "#dc2626"
    return f"""
      <svg viewBox="0 0 120 28" role="img" aria-label="Target rank before and after">
        <rect x="0" y="3" width="{before_width}" height="8" rx="2" fill="#94a3b8"/>
        <rect x="0" y="17" width="{after_width}" height="8" rx="2" fill="{color}"/>
        <text x="104" y="10" font-size="7" fill="#475569">before</text>
        <text x="104" y="24" font-size="7" fill="#475569">after</text>
      </svg>
    """


def _top_tokens(tokens: tuple[TokenRank, ...]) -> str:
    return ", ".join(
        f"{escape(token.token)!r} #{token.rank + 1}" for token in tokens[:8]
    )


def render_html_report(
    results: tuple[BenchmarkResult, ...],
    *,
    title: str = "J-Lens Intervention Benchmark",
) -> str:
    """Render a standalone benchmark report."""
    total = len(results)
    passed = sum(1 for result in results if result.success)
    pass_rate = 0.0 if total == 0 else passed / total * 100.0
    rows = []
    for result in results:
        case = result.case
        status_class = "pass" if result.success else "fail"
        source_rank = (
            "—"
            if result.source_before is None
            else f"{_rank_text(result.source_before)} → {_rank_text(result.source_after)}"
        )
        rows.append(
            f"""
            <tr class="{status_class}">
              <td>{escape(case.name)}</td>
              <td>{escape(case.operation)}</td>
              <td>{escape(case.context_mode)}</td>
              <td>{'pass' if result.success else 'fail'}</td>
              <td>{'yes' if result.applied else 'no'}</td>
              <td>{_rank_text(result.target_before)} → {_rank_text(result.target_after)}</td>
              <td>{source_rank}</td>
              <td>L{result.selected_layer if result.selected_layer is not None else '—'}</td>
              <td>{result.selected_scale:.4g}</td>
              <td>{result.normalized_cost:.4g}</td>
              <td>{_rank_bar(result.target_before, result.target_after)}</td>
              <td>{escape(str(result.trace.get('message', '')))}</td>
            </tr>
            <tr class="details">
              <td colspan="12">
                <details>
                  <summary>Prompt, top tokens, and trace</summary>
                  <pre>{escape(case.prompt)}</pre>
                  <p><b>Before:</b> {_top_tokens(result.top_before)}</p>
                  <p><b>After:</b> {_top_tokens(result.top_after)}</p>
                  <pre>{escape(str(result.to_dict()))}</pre>
                </details>
              </td>
            </tr>
            """
        )
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #0f172a; }}
    h1 {{ margin-bottom: 4px; }}
    .summary {{ display: flex; gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #cbd5e1; border-radius: 10px; padding: 12px 16px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px; vertical-align: top; }}
    th {{ text-align: left; background: #f8fafc; position: sticky; top: 0; }}
    tr.pass {{ background: #f0fdf4; }}
    tr.fail {{ background: #fef2f2; }}
    tr.details {{ background: white; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; padding: 8px; border-radius: 6px; }}
    svg {{ width: 160px; height: 38px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p>Strict residual/J-space benchmark. No logit fallback, lexical fallback, or corrective residual fallback.</p>
  <div class="summary">
    <div class="card"><div>Cases</div><div class="value">{total}</div></div>
    <div class="card"><div>Passed</div><div class="value">{passed}</div></div>
    <div class="card"><div>Pass rate</div><div class="value">{pass_rate:.1f}%</div></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Case</th><th>Operation</th><th>Context</th><th>Status</th>
        <th>Applied</th><th>Target rank</th><th>Source rank</th>
        <th>Layer</th><th>Scale</th><th>Cost</th><th>Rank visual</th><th>Message</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""


def _layer_rank_sparkline(result: ReadoutBenchmarkResult) -> str:
    if not result.layer_ranks:
        return ""
    max_rank = max(max(item.rank for item in result.layer_ranks), result.case.max_rank)
    bars = []
    for item in result.layer_ranks:
        height = max(2, 50 - min(48, int(item.rank / max_rank * 48)))
        color = "#16a34a" if item.rank < result.case.max_rank else "#dc2626"
        bars.append(
            f'<rect x="{len(bars) * 8}" y="{54 - height}" '
            f'width="5" height="{height}" fill="{color}"><title>'
            f"L{item.layer}: rank #{item.rank + 1}</title></rect>"
        )
    width = max(80, len(bars) * 8)
    return f'<svg viewBox="0 0 {width} 56">{"".join(bars)}</svg>'


def render_readout_html_report(
    results: tuple[ReadoutBenchmarkResult, ...],
    *,
    title: str = "J-Lens Viewing Benchmark",
) -> str:
    total = len(results)
    passed = sum(1 for result in results if result.success)
    rows = []
    for result in results:
        rows.append(
            f"""
            <tr class="{'pass' if result.success else 'fail'}">
              <td>{escape(result.case.name)}</td>
              <td>{escape(result.case.context_mode)}</td>
              <td>{escape(result.target_token)!r}</td>
              <td>P{result.resolved_position}</td>
              <td>L{result.best_layer}</td>
              <td>#{result.best_rank + 1}</td>
              <td>top {result.case.max_rank}</td>
              <td>{'pass' if result.success else 'fail'}</td>
              <td>{_layer_rank_sparkline(result)}</td>
            </tr>
            <tr class="details">
              <td colspan="9">
                <details>
                  <summary>Prompt and layer ranks</summary>
                  <pre>{escape(result.case.prompt)}</pre>
                  <pre>{escape(str(result.to_dict()))}</pre>
                </details>
              </td>
            </tr>
            """
        )
    pass_rate = 0.0 if total == 0 else passed / total * 100.0
    return f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <title>{escape(title)}</title>
  <style>
    body {{ font-family: system-ui, sans-serif; margin: 24px; color: #0f172a; }}
    .summary {{ display: flex; gap: 12px; margin: 18px 0; }}
    .card {{ border: 1px solid #cbd5e1; border-radius: 10px; padding: 12px 16px; }}
    .value {{ font-size: 24px; font-weight: 700; }}
    table {{ border-collapse: collapse; width: 100%; font-size: 13px; }}
    th, td {{ border-bottom: 1px solid #e2e8f0; padding: 8px; vertical-align: top; }}
    th {{ text-align: left; background: #f8fafc; }}
    tr.pass {{ background: #f0fdf4; }}
    tr.fail {{ background: #fef2f2; }}
    tr.details {{ background: white; }}
    pre {{ white-space: pre-wrap; background: #f8fafc; padding: 8px; border-radius: 6px; }}
    svg {{ width: 220px; height: 72px; }}
  </style>
</head>
<body>
  <h1>{escape(title)}</h1>
  <p>Prompt-context J-lens readout benchmark. It runs one forward pass over the supplied context, matching the original notebook viewing path.</p>
  <div class="summary">
    <div class="card"><div>Cases</div><div class="value">{total}</div></div>
    <div class="card"><div>Passed</div><div class="value">{passed}</div></div>
    <div class="card"><div>Pass rate</div><div class="value">{pass_rate:.1f}%</div></div>
  </div>
  <table>
    <thead>
      <tr>
        <th>Case</th><th>Context</th><th>Target</th><th>Position</th>
        <th>Best layer</th><th>Best rank</th>
        <th>Criterion</th><th>Status</th><th>Layer ranks</th>
      </tr>
    </thead>
    <tbody>
      {''.join(rows)}
    </tbody>
  </table>
</body>
</html>
"""
