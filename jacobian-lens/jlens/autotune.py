# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Device-local VJP batch autotuning with a persistent fingerprinted cache."""

from __future__ import annotations

import hashlib
import json
import os
import time
from collections.abc import Callable, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from jlens.fitting import sketched_jacobian_for_prompt
from jlens.protocol import LensModel


@dataclass(frozen=True)
class AutotuneResult:
    batch_size: int
    elapsed_seconds: float
    peak_memory_fraction: float
    cached: bool = False


def choose_batch(
    candidates: Sequence[int],
    runner: Callable[[int], tuple[float, float]],
    *,
    memory_fraction: float = 0.9,
) -> AutotuneResult:
    """Choose the fastest measured candidate below the memory ceiling."""
    safe: list[AutotuneResult] = []
    for candidate in candidates:
        try:
            elapsed, peak = runner(candidate)
        except torch.OutOfMemoryError:
            break
        if peak <= memory_fraction:
            safe.append(AutotuneResult(candidate, elapsed, peak))
    if not safe:
        raise torch.OutOfMemoryError(
            "no VJP batch size fits below the configured memory ceiling"
        )
    return min(safe, key=lambda result: result.elapsed_seconds)


def _fingerprint(model: LensModel, layers: Sequence[int], seq_len: int) -> str:
    device = "cpu"
    if torch.cuda.is_available():
        device = torch.cuda.get_device_name(torch.cuda.current_device())
    value = (
        f"{type(model).__module__}.{type(model).__qualname__}|{model.n_layers}|"
        f"{model.d_model}|{tuple(layers)}|{device}|seq{2 ** (seq_len - 1).bit_length()}"
    )
    return hashlib.sha256(value.encode()).hexdigest()[:24]


def _read_cache(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return {}


def _write_cache(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True), encoding="utf-8")
    os.replace(temporary, path)


def autotune_vjp_batch(
    model: LensModel,
    prompt: str,
    layers: Sequence[int],
    *,
    candidates: Sequence[int] = (1, 2, 4, 8, 16),
    memory_fraction: float = 0.9,
    max_seq_len: int = 128,
    target_layer: int | None = None,
    cache_path: str | None = None,
) -> AutotuneResult:
    """Benchmark a small probe slice and cache the fastest safe batch size."""
    cache_file = Path(cache_path or "~/.cache/jlens/autotune.json").expanduser()
    seq_len = model.encode(prompt, max_length=max_seq_len).shape[1]
    key = _fingerprint(model, layers, seq_len)
    cache = _read_cache(cache_file)
    if key in cache:
        stored = cache[key]
        return AutotuneResult(
            batch_size=int(stored["batch_size"]),
            elapsed_seconds=float(stored["elapsed_seconds"]),
            peak_memory_fraction=float(stored["peak_memory_fraction"]),
            cached=True,
        )

    total_memory = 1
    if torch.cuda.is_available():
        total_memory = torch.cuda.get_device_properties(0).total_memory

    def run(candidate: int) -> tuple[float, float]:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.synchronize()
        started = time.perf_counter()
        sketched_jacobian_for_prompt(
            model,
            prompt,
            layers,
            sketch_rank=min(16, model.d_model),
            target_layer=target_layer,
            dim_batch=candidate,
            max_seq_len=max_seq_len,
            vjp_backend="batched",
        )
        if torch.cuda.is_available():
            torch.cuda.synchronize()
            peak = torch.cuda.max_memory_allocated() / total_memory
        else:
            peak = 0.0
        return time.perf_counter() - started, peak

    result = choose_batch(candidates, run, memory_fraction=memory_fraction)
    cache[key] = asdict(result)
    _write_cache(cache_file, cache)
    return result
