# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Fitting the Jacobian lens.

The lens reads out an early-layer residual ``h_l`` by linearly transporting it
into the final-layer basis with the average input-output Jacobian, then
decoding with the model's own unembedding::

    lens_l(h) = unembed( J_l @ h )

Estimator (:func:`jacobian_for_prompt`): for each output dimension, inject a
one-hot cotangent at *every valid target position at once* and backprop. The
gradient at source position ``p`` is then ``sum_{p' >= p} dh_final[p'] / dh_l[p]``,
the sum over later target positions; we take the mean over source positions
``p``. This is the reduction used in the paper. A per-position estimator
(``dh_final[p] / dh_l[p]`` averaged over ``p``) gives a slightly different
``J_l``; both work as a lens.

Cost: one forward pass and ``ceil(d_model / dim_batch)`` backward passes per
prompt. Shard across machines by running :func:`fit` on disjoint prompt
slices and merging with :meth:`jlens.lens.JacobianLens.merge`.
"""

from __future__ import annotations

import logging
import math
import os
import time
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager, nullcontext
from typing import Literal

import torch

from jlens.geometry import CovarianceMetric
from jlens.hooks import ActivationRecorder
from jlens.lens import JacobianLens, SketchedJacobian
from jlens.protocol import LensModel

logger = logging.getLogger(__name__)

#: Positions before this index are excluded from the Jacobian average; early
#: positions act as attention sinks and have atypical residual statistics.
SKIP_FIRST_N_POSITIONS = 16


def valid_position_mask(
    seq_len: int, *, skip_first: int = SKIP_FIRST_N_POSITIONS
) -> torch.Tensor:
    """Boolean mask over sequence positions to include in the Jacobian average.

    Early positions are dominated by attention-sink behaviour and the final
    position has no next-token target, so both are excluded.

    Args:
        seq_len: Length of the tokenized prompt.
        skip_first: Number of leading positions to exclude.

    Returns:
        Boolean tensor of shape ``[seq_len]``.

    Raises:
        ValueError: If ``skip_first`` is negative or the prompt is too short to
            leave any valid positions.
    """
    if skip_first < 0:
        raise ValueError(f"skip_first must be >= 0, got {skip_first}")
    mask = torch.zeros(seq_len, dtype=torch.bool)
    mask[skip_first : seq_len - 1] = True
    if mask.sum() == 0:
        raise ValueError(
            f"prompt too short: seq_len={seq_len}, need > {skip_first + 1} tokens"
        )
    return mask


def _check_layer_indices(
    source_layers: Sequence[int] | None, target_layer: int | None, n_layers: int
) -> tuple[list[int], int]:
    """Resolve None/negative layer indices, bounds-check, enforce source < target."""
    target = n_layers - 1 if target_layer is None else target_layer
    if target < 0:
        target += n_layers
    if not 0 <= target < n_layers:
        raise ValueError(
            f"target_layer={target_layer} out of range for {n_layers} layers"
        )
    if source_layers is None:
        return list(range(target)), target
    sources = sorted({l + n_layers if l < 0 else l for l in source_layers})
    if not sources or sources[0] < 0 or sources[-1] >= n_layers:
        raise ValueError(
            f"source_layers {sorted(source_layers)} out of range for {n_layers} layers"
        )
    if sources[-1] >= target:
        raise ValueError(
            f"source_layers must all be < target_layer={target}; got max={sources[-1]}"
        )
    return sources, target


def jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
) -> tuple[dict[int, torch.Tensor], int, int]:
    """Compute the per-layer Jacobian estimator ``J_l`` for one prompt.

    Runs one forward pass on the prompt replicated ``dim_batch`` times along
    the batch axis, retains the graph, then runs ``ceil(d_model / dim_batch)``
    backward passes against it. Each backward computes ``dim_batch`` rows of
    ``J_l`` at once: batch element ``b`` carries a one-hot cotangent at output
    dimension ``dim_start + b``, set at every valid target position. See the
    module docstring for the resulting estimator and how it relates to
    a strict per-position Jacobian.

    Args:
        model: The model to compute Jacobians for.
        prompt: Input text.
        source_layers: Layer indices ``l`` to compute ``J_l`` at.
        target_layer: Layer to take gradients with respect to. Defaults to the
            final layer; negative indices count from the end. In some cases,
            targeting the penultimate layer can give a better-conditioned
            ``J_l``.
        dim_batch: Output dimensions computed per backward pass. Higher uses
            more GPU memory (the prompt is replicated this many times); total
            backward FLOPs are unchanged.
        max_seq_len: Truncate the prompt to this many tokens.
        skip_first: Leading positions to exclude; see :func:`valid_position_mask`.

    Returns:
        ``(jacobians, seq_len, n_valid_positions)``. ``jacobians`` maps each
        source layer to a ``[d_model, d_model]`` fp32 CPU tensor.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())

    jacobians = {
        layer: torch.zeros(d_model, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(d_model / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        # One forward on the prompt replicated dim_batch times. The retained
        # graph is reused for every backward pass below.
        replicated_ids = input_ids.expand(dim_batch, -1)
        model.forward(replicated_ids)
        target_activation = recorder.activations[
            target_layer
        ]  # [dim_batch, seq_len, d_model]
        source_activations = [recorder.activations[layer] for layer in source_layers]

        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        batch_indices = torch.arange(dim_batch, device=target_activation.device)
        cotangent = torch.zeros_like(target_activation)

        for pass_idx, dim_start in enumerate(range(0, d_model, dim_batch)):
            n_dims_this_pass = min(dim_batch, d_model - dim_start)
            # One-hot cotangent at dim (dim_start + b) for batch element b,
            # at every valid target position. Yields rows dim_start..+n of J_l.
            cotangent.zero_()
            cotangent[
                batch_indices[:n_dims_this_pass, None],
                valid_positions[None, :],
                dim_start + batch_indices[:n_dims_this_pass, None],
            ] = 1.0
            grads = torch.autograd.grad(
                outputs=target_activation,
                inputs=source_activations,
                grad_outputs=cotangent,
                retain_graph=(pass_idx < n_passes - 1),
            )
            for layer, grad in zip(source_layers, grads, strict=True):
                # grad: [dim_batch, seq_len, d_model] on whatever device this
                # layer lives on; mean over the valid positions -> dim_batch rows.
                positions_on_device = valid_positions.to(grad.device, non_blocking=True)
                rows = (
                    grad[:n_dims_this_pass, positions_on_device, :].float().mean(dim=1)
                )
                jacobians[layer][dim_start : dim_start + n_dims_this_pass, :] = (
                    rows.cpu()
                )
            del grads
            if pass_idx % 100 == 0 or pass_idx == n_passes - 1:
                logger.debug(
                    "    pass %d/%d (dims %d-%d)",
                    pass_idx + 1,
                    n_passes,
                    dim_start,
                    dim_start + n_dims_this_pass,
                )

    return jacobians, seq_len, n_valid_positions


def _orthogonal_probes(
    d_model: int, rank: int, seed: int, *, block_index: int = 0
) -> torch.Tensor:
    if not 1 <= rank <= d_model:
        raise ValueError(f"sketch_rank must lie in 1..{d_model}, got {rank}")
    blocks_per_cycle = math.ceil(d_model / rank)
    cycle, block = divmod(block_index, blocks_per_cycle)
    generator = torch.Generator(device="cpu").manual_seed(seed + cycle)
    order = torch.randperm(d_model, generator=generator)
    start = block * rank
    row_indices = order[torch.arange(start, start + rank) % d_model].float()
    dimensions = torch.arange(d_model, dtype=torch.float32)
    angles = math.pi / d_model * row_indices[:, None] * (dimensions[None, :] + 0.5)
    probes = math.sqrt(2.0) * torch.cos(angles)
    probes[row_indices == 0] = 1.0
    signs = torch.randint(0, 2, (d_model,), generator=generator).float() * 2 - 1
    return (probes * signs).contiguous()


def _batched_vjp(
    target: torch.Tensor,
    sources: Sequence[torch.Tensor],
    cotangents: torch.Tensor,
    *,
    retain_graph: bool,
) -> tuple[torch.Tensor, ...]:
    """Evaluate several vector-Jacobian products on one retained graph."""
    return torch.autograd.grad(
        outputs=target,
        inputs=sources,
        grad_outputs=cotangents,
        is_grads_batched=True,
        retain_graph=retain_graph,
    )


def sketched_jacobian_for_prompt(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    sketch_rank: int = 64,
    target_layer: int | None = None,
    dim_batch: int = 4,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    seed: int = 0,
    _probes: torch.Tensor | None = None,
    vjp_backend: Literal["auto", "batched", "replicated"] = "auto",
) -> tuple[dict[int, SketchedJacobian], int, int]:
    """Estimate each Jacobian with a batched VJP and safe compatibility fallback."""
    if vjp_backend not in {"auto", "batched", "replicated"}:
        raise ValueError(
            "vjp_backend must be 'auto', 'batched', or 'replicated', "
            f"got {vjp_backend!r}"
        )
    kwargs = dict(
        sketch_rank=sketch_rank,
        target_layer=target_layer,
        dim_batch=dim_batch,
        max_seq_len=max_seq_len,
        skip_first=skip_first,
        seed=seed,
        _probes=_probes,
    )
    if vjp_backend != "auto":
        return _sketched_jacobian_for_prompt_impl(
            model, prompt, source_layers, vjp_backend=vjp_backend, **kwargs
        )
    try:
        return _sketched_jacobian_for_prompt_impl(
            model, prompt, source_layers, vjp_backend="batched", **kwargs
        )
    except torch.OutOfMemoryError:
        raise
    except RuntimeError as exc:
        logger.warning("batched VJP unavailable; using replicated fallback: %s", exc)
        return _sketched_jacobian_for_prompt_impl(
            model, prompt, source_layers, vjp_backend="replicated", **kwargs
        )


def _sketched_jacobian_for_prompt_impl(
    model: LensModel,
    prompt: str,
    source_layers: Sequence[int],
    *,
    sketch_rank: int,
    target_layer: int | None,
    dim_batch: int,
    max_seq_len: int,
    skip_first: int,
    seed: int,
    _probes: torch.Tensor | None,
    vjp_backend: Literal["batched", "replicated"],
) -> tuple[dict[int, SketchedJacobian], int, int]:
    """Estimate each Jacobian as an identity plus randomized low-rank update.

    Each reverse pass evaluates ``J.T @ z`` for an orthogonal random probe.
    Subtracting ``z`` exactly preserves the residual identity path and sketches
    only the transformer's correction. The estimator is unbiased over random
    orthogonal probe subspaces and becomes exact when ``sketch_rank=d_model``.
    """
    if dim_batch <= 0:
        raise ValueError("dim_batch must be positive")
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )
    probes = (
        _orthogonal_probes(d_model, sketch_rank, seed)
        if _probes is None
        else _probes.detach().float().cpu()
    )
    if probes.shape != (sketch_rank, d_model):
        raise ValueError(
            f"probes must have shape {(sketch_rank, d_model)}, got {tuple(probes.shape)}"
        )

    input_ids = model.encode(prompt, max_length=max_seq_len)
    seq_len = input_ids.shape[1]
    position_mask = valid_position_mask(seq_len, skip_first=skip_first)
    n_valid_positions = int(position_mask.sum())
    corrections = {
        layer: torch.zeros(sketch_rank, d_model, dtype=torch.float32)
        for layer in source_layers
    }
    n_passes = math.ceil(sketch_rank / dim_batch)

    with (
        ActivationRecorder(
            model.layers,
            at=[*source_layers, target_layer],
            start_graph_at=min(source_layers),
        ) as recorder,
        torch.enable_grad(),
    ):
        forward_ids = (
            input_ids
            if vjp_backend == "batched"
            else input_ids.expand(dim_batch, -1)
        )
        model.forward(forward_ids)
        target_activation = recorder.activations[target_layer]
        source_activations = [recorder.activations[layer] for layer in source_layers]
        valid_positions = position_mask.nonzero(as_tuple=True)[0].to(
            target_activation.device
        )
        cotangent = (
            None
            if vjp_backend == "batched"
            else torch.zeros_like(target_activation)
        )

        for pass_index, start in enumerate(range(0, sketch_rank, dim_batch)):
            count = min(dim_batch, sketch_rank - start)
            probe_batch = probes[start : start + count].to(
                target_activation.device, dtype=target_activation.dtype
            )
            retain_graph = pass_index < n_passes - 1
            if vjp_backend == "batched":
                batched_cotangents = torch.zeros(
                    (count, *target_activation.shape),
                    device=target_activation.device,
                    dtype=target_activation.dtype,
                )
                batched_cotangents[:, 0, valid_positions, :] = probe_batch[:, None, :]
                grads = _batched_vjp(
                    target_activation,
                    source_activations,
                    batched_cotangents,
                    retain_graph=retain_graph,
                )
            else:
                assert cotangent is not None
                cotangent.zero_()
                cotangent[:count, valid_positions, :] = probe_batch[:, None, :]
                grads = torch.autograd.grad(
                    outputs=target_activation,
                    inputs=source_activations,
                    grad_outputs=cotangent,
                    retain_graph=retain_graph,
                )
            for layer, grad in zip(source_layers, grads, strict=True):
                positions = valid_positions.to(grad.device, non_blocking=True)
                if vjp_backend == "batched":
                    transported = grad[:, 0, positions, :].float().mean(dim=1)
                else:
                    transported = grad[:count, positions, :].float().mean(dim=1)
                identity = probes[start : start + count].to(grad.device)
                corrections[layer][start : start + count] = (
                    transported - identity
                ).cpu()
            del grads

    return (
        {
            layer: SketchedJacobian(probes=probes, corrections=correction)
            for layer, correction in corrections.items()
        },
        seq_len,
        n_valid_positions,
    )


def fit_sketch(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    sketch_rank: int = 64,
    target_layer: int | None = None,
    dim_batch: int = 4,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    seed: int = 0,
    probe_blocks: int = 1,
    vjp_backend: Literal["auto", "batched", "replicated"] = "auto",
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
    on_progress: Callable[[dict[str, int]], None] | None = None,
    operation_context: Callable[[str], AbstractContextManager] | None = None,
) -> JacobianLens:
    """Fit a prompt-averaged randomized Jacobian transport.

    Every fixed probe block is averaged over the prompt corpus before
    independent blocks are combined. This keeps corpus sampling variance
    separate from randomized projection variance.
    """
    if probe_blocks <= 0:
        raise ValueError("probe_blocks must be positive")
    sources, target = _check_layer_indices(source_layers, target_layer, model.n_layers)
    probes_by_block = [
        _orthogonal_probes(model.d_model, sketch_rank, seed, block_index=index)
        for index in range(probe_blocks)
    ]
    correction_sums = {
        layer: [
            torch.zeros(sketch_rank, model.d_model, dtype=torch.float32)
            for _ in range(probe_blocks)
        ]
        for layer in sources
    }
    prompt_counts = [0 for _ in range(probe_blocks)]
    next_block = 0
    next_prompt = 0
    if resume and checkpoint_path and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        if state.get("format") != "prompt-averaged-probe-blocks-v2":
            raise ValueError(
                "incompatible sketch checkpoint; restart progressive fitting"
            )
        expected = (sources, target, sketch_rank, seed, skip_first, probe_blocks)
        found = (
            state["source_layers"],
            state["target_layer"],
            state["sketch_rank"],
            state["seed"],
            state["skip_first"],
            state.get("probe_blocks"),
        )
        if found != expected:
            raise ValueError("sketch checkpoint configuration does not match")
        probes_by_block = [value.float() for value in state["probes_by_block"]]
        correction_sums = {
            int(layer): [value.float() for value in values]
            for layer, values in state["correction_sums"].items()
        }
        prompt_counts = list(state["prompt_counts"])
        next_block = state["next_block"]
        next_prompt = state["next_prompt"]

    def write_checkpoint() -> None:
        if checkpoint_path:
            _atomic_save(
                {
                    "source_layers": sources,
                    "target_layer": target,
                    "sketch_rank": sketch_rank,
                    "seed": seed,
                    "skip_first": skip_first,
                    "probe_blocks": probe_blocks,
                    "format": "prompt-averaged-probe-blocks-v2",
                    "probes_by_block": probes_by_block,
                    "correction_sums": correction_sums,
                    "prompt_counts": prompt_counts,
                    "next_block": next_block,
                    "next_prompt": next_prompt,
                },
                checkpoint_path,
            )

    for block_index in range(next_block, probe_blocks):
        start_prompt = next_prompt if block_index == next_block else 0
        probes = probes_by_block[block_index]
        for prompt_index in range(start_prompt, len(prompts)):
            prompt = prompts[prompt_index]
            started = time.perf_counter()
            try:
                operation = (
                    operation_context("lens-fit-prompt")
                    if operation_context is not None
                    else nullcontext()
                )
                with operation:
                    estimates, seq_len, n_valid = sketched_jacobian_for_prompt(
                        model,
                        prompt,
                        sources,
                        sketch_rank=sketch_rank,
                        target_layer=target,
                        dim_batch=dim_batch,
                        max_seq_len=max_seq_len,
                        skip_first=skip_first,
                        seed=seed,
                        _probes=probes,
                        vjp_backend=vjp_backend,
                    )
            except ValueError as exc:
                logger.warning("  skipping prompt %d: %s", prompt_index, exc)
            else:
                for layer in sources:
                    correction_sums[layer][block_index] += estimates[layer].corrections
                prompt_counts[block_index] += 1
                logger.info(
                    "  block %d/%d prompt %d/%d seq_len=%d n_valid=%d %.1fs",
                    block_index + 1,
                    probe_blocks,
                    prompt_index + 1,
                    len(prompts),
                    seq_len,
                    n_valid,
                    time.perf_counter() - started,
                )
            next_block = block_index
            next_prompt = prompt_index + 1
            if checkpoint_every and next_prompt % checkpoint_every == 0:
                write_checkpoint()
            if on_progress is not None:
                on_progress(
                    {
                        "block": block_index + 1,
                        "blocks": probe_blocks,
                        "prompt": prompt_index + 1,
                        "prompts": len(prompts),
                        "successful_prompts": prompt_counts[block_index],
                    }
                )
        next_block = block_index + 1
        next_prompt = 0
        write_checkpoint()
    write_checkpoint()
    if not prompt_counts or min(prompt_counts) == 0:
        raise ValueError("no prompts were long enough to fit on")
    combined_probes = torch.cat(probes_by_block, dim=0)
    sketches = {
        layer: SketchedJacobian(
            probes=combined_probes,
            corrections=torch.cat(
                [
                    correction_sums[layer][index] / prompt_counts[index]
                    for index in range(probe_blocks)
                ],
                dim=0,
            ),
        )
        for layer in sources
    }
    return JacobianLens(
        sketches=sketches,
        n_prompts=min(prompt_counts),
        d_model=model.d_model,
        metadata={
            "estimator": "prompt-averaged-orthogonal-sketch-v2",
            "sketch_rank": str(sketch_rank),
            "probe_blocks": str(probe_blocks),
            "effective_rank": str(sketch_rank * probe_blocks),
            "sketch_seed": str(seed),
            "target_layer": str(target),
            "source_layers": ",".join(str(layer) for layer in sources),
        },
    )


def _atomic_save(obj: object, path: str) -> None:
    """``torch.save`` to a temp file then ``os.replace`` so a crash never
    leaves a half-written checkpoint."""
    tmp_path = f"{path}.tmp.{os.getpid()}"
    torch.save(obj, tmp_path)
    os.replace(tmp_path, path)


def fit(
    model: LensModel,
    prompts: Sequence[str],
    *,
    source_layers: Sequence[int] | None = None,
    target_layer: int | None = None,
    dim_batch: int = 8,
    max_seq_len: int = 128,
    skip_first: int = SKIP_FIRST_N_POSITIONS,
    checkpoint_path: str | None = None,
    checkpoint_every: int | None = 1,
    resume: bool = True,
) -> JacobianLens:
    """Fit ``J_l`` over a list of prompts and return a :class:`JacobianLens`.

    Per-prompt Jacobians from :func:`jacobian_for_prompt` are accumulated as a
    running mean. If ``checkpoint_path`` is set, the running sum is written
    every ``checkpoint_every`` prompts (atomic) and resumed from on restart.

    Args:
        model: The model to fit on.
        prompts: Text prompts to average over. See the README for guidance on
            corpus size and distribution.
        source_layers: Layers to fit at. Defaults to every layer below
            ``target_layer``; negative indices count from the end.
        target_layer: See :func:`jacobian_for_prompt`. Defaults to the final
            layer; negative indices count from the end.
        dim_batch: See :func:`jacobian_for_prompt`.
        max_seq_len: Truncate each prompt to this many tokens.
        skip_first: See :func:`jacobian_for_prompt`.
        checkpoint_path: If set, write a resumable checkpoint here.
        checkpoint_every: Write the checkpoint every N prompts (default 1).
            ``None`` skips per-iteration writes and saves once at the end; the
            checkpoint can be large (``len(source_layers) * d_model**2 * 4``
            bytes), so raise this for large models.
        resume: If ``True`` and ``checkpoint_path`` exists, resume from it.

    Returns:
        The fitted :class:`JacobianLens`.
    """
    n_layers, d_model = model.n_layers, model.d_model
    source_layers, target_layer = _check_layer_indices(
        source_layers, target_layer, n_layers
    )

    logger.info(
        "fit: n_layers=%d d_model=%d, fitting %d source layers "
        "(target=L%d) on %d prompts",
        n_layers,
        d_model,
        len(source_layers),
        target_layer,
        len(prompts),
    )

    # Running state: sum of per-prompt Jacobians, success count, and the list
    # index to resume from. ``next_idx`` is tracked separately from ``n_done``
    # so a too-short prompt that was skipped is not re-processed on resume.
    jacobian_sum: dict[int, torch.Tensor]
    n_done: int
    next_idx: int
    if resume and checkpoint_path is not None and os.path.exists(checkpoint_path):
        state = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        for key, expected in (
            ("source_layers", source_layers),
            ("target_layer", target_layer),
            ("skip_first", skip_first),
        ):
            if key in state and state[key] != expected:
                raise ValueError(
                    f"checkpoint at {checkpoint_path} was fitted with {key}="
                    f"{state[key]!r}, not {expected!r}; pass resume=False to discard it"
                )
        jacobian_sum, n_done, next_idx = (
            state["jacobian_sum"],
            state["n_done"],
            state["next_idx"],
        )
        logger.info(
            "  resuming from checkpoint: %d/%d prompts processed",
            next_idx,
            len(prompts),
        )
    else:
        jacobian_sum = {
            layer: torch.zeros(d_model, d_model, dtype=torch.float32)
            for layer in source_layers
        }
        n_done = 0
        next_idx = 0

    def write_checkpoint() -> None:
        if checkpoint_path is not None:
            _atomic_save(
                {
                    "jacobian_sum": jacobian_sum,
                    "n_done": n_done,
                    "next_idx": next_idx,
                    "source_layers": source_layers,
                    "target_layer": target_layer,
                    "skip_first": skip_first,
                },
                checkpoint_path,
            )

    sqrt_d = math.sqrt(d_model)
    for prompt_idx, prompt in enumerate(prompts):
        if prompt_idx < next_idx:
            continue
        start_time = time.perf_counter()
        try:
            per_prompt_J, seq_len, n_valid = jacobian_for_prompt(
                model,
                prompt,
                source_layers,
                target_layer=target_layer,
                dim_batch=dim_batch,
                max_seq_len=max_seq_len,
                skip_first=skip_first,
            )
        except ValueError as exc:
            logger.warning("  skipping prompt %d: %s", prompt_idx, exc)
            next_idx = prompt_idx + 1
            continue

        # Per-prompt diagnostics, max over source layers: the prompt's own
        # Jacobian norm flags heavy-tailed outliers, and the relative shift
        # in the running mean tracks convergence (falls ~1/n once settled).
        prompt_norm = max(per_prompt_J[l].norm().item() for l in source_layers) / sqrt_d
        if n_done > 0:
            mean_rel_change = max(
                (
                    (per_prompt_J[l] - jacobian_sum[l] / n_done).norm()
                    / ((n_done + 1) * (jacobian_sum[l] / n_done).norm())
                ).item()
                for l in source_layers
            )
        else:
            mean_rel_change = float("nan")

        for layer in source_layers:
            jacobian_sum[layer] += per_prompt_J[layer]
        n_done += 1
        next_idx = prompt_idx + 1

        logger.info(
            "  prompt %d/%d  seq_len=%d n_valid=%d  %.0fs  "
            "max||J||/sqrt(d)=%.3f  max_d_mean=%.2e",
            prompt_idx + 1,
            len(prompts),
            seq_len,
            n_valid,
            time.perf_counter() - start_time,
            prompt_norm,
            mean_rel_change,
        )
        if checkpoint_every is not None and next_idx % checkpoint_every == 0:
            write_checkpoint()

    write_checkpoint()
    if n_done == 0:
        raise ValueError("no prompts were long enough to fit on")
    jacobian_mean = {layer: jacobian_sum[layer] / n_done for layer in source_layers}
    logger.info("fit: done, %d prompts", n_done)
    return JacobianLens(jacobians=jacobian_mean, n_prompts=n_done, d_model=d_model)


class _FrequentDirections:
    """Bounded-memory sketch of centered activation rows."""

    def __init__(self, rank: int, width: int) -> None:
        self.rank = min(rank, width)
        self.width = width
        self.buffer = torch.zeros(2 * self.rank, width, dtype=torch.float32)
        self.size = 0

    def _compress(self) -> None:
        if self.size <= self.rank:
            return
        _, singular, vh = torch.linalg.svd(
            self.buffer[: self.size], full_matrices=False
        )
        kept = min(self.rank, len(singular))
        threshold = singular[kept].square() if kept < len(singular) else 0.0
        shrunk = torch.sqrt(torch.clamp(singular[:kept].square() - threshold, min=0))
        self.buffer.zero_()
        self.buffer[:kept] = shrunk[:, None] * vh[:kept]
        self.size = kept

    def add(self, rows: torch.Tensor) -> None:
        rows = rows.detach().float().cpu().reshape(-1, self.width)
        offset = 0
        while offset < len(rows):
            if self.size == len(self.buffer):
                self._compress()
            available = len(self.buffer) - self.size
            take = min(available, len(rows) - offset)
            self.buffer[self.size : self.size + take] = rows[offset : offset + take]
            self.size += take
            offset += take

    def factors(self, denominator: int) -> torch.Tensor:
        self._compress()
        rows = self.buffer[: self.size]
        if len(rows) > self.rank:
            _, singular, vh = torch.linalg.svd(rows, full_matrices=False)
            rows = singular[: self.rank, None] * vh[: self.rank]
        return rows / math.sqrt(max(denominator, 1))


@torch.no_grad()
def calibrate_geometry(
    model: LensModel,
    lens: JacobianLens,
    prompts: Sequence[str],
    *,
    max_seq_len: int = 128,
    rank: int = 16,
    shrinkage: float = 0.05,
) -> JacobianLens:
    """Fit bounded-rank residual covariance geometry for intervention costs.

    This uses two inference-only passes: Welford statistics establish exact
    means and diagonal variances, then Frequent Directions sketches centered
    residual rows without materializing a dense covariance matrix.
    """
    if not prompts:
        raise ValueError("calibration needs at least one prompt")
    if rank <= 0:
        raise ValueError("rank must be positive")
    if not 0 <= shrinkage <= 1:
        raise ValueError("shrinkage must lie in [0, 1]")
    if lens.d_model != model.d_model:
        raise ValueError("lens and model disagree on d_model")
    layers = lens.source_layers

    counts = {layer: 0 for layer in layers}
    means = {layer: torch.zeros(model.d_model, dtype=torch.float64) for layer in layers}
    m2s = {layer: torch.zeros(model.d_model, dtype=torch.float64) for layer in layers}

    def activation_rows(prompt: str) -> dict[int, torch.Tensor]:
        input_ids = model.encode(prompt, max_length=max_seq_len)
        with ActivationRecorder(model.layers, at=layers) as recorder:
            model.forward(input_ids)
        return {
            layer: recorder.activations[layer]
            .detach()
            .float()
            .reshape(-1, model.d_model)
            .cpu()
            for layer in layers
        }

    for prompt in prompts:
        for layer, rows in activation_rows(prompt).items():
            batch_count = len(rows)
            if batch_count == 0:
                continue
            batch_mean = rows.double().mean(dim=0)
            batch_m2 = (rows.double() - batch_mean).square().sum(dim=0)
            old_count = counts[layer]
            new_count = old_count + batch_count
            difference = batch_mean - means[layer]
            means[layer] += difference * (batch_count / new_count)
            m2s[layer] += (
                batch_m2 + difference.square() * old_count * batch_count / new_count
            )
            counts[layer] = new_count

    sketches = {layer: _FrequentDirections(rank, model.d_model) for layer in layers}
    for prompt in prompts:
        for layer, rows in activation_rows(prompt).items():
            sketches[layer].add(rows - means[layer].float())

    geometry = {}
    for layer in layers:
        if counts[layer] < 2:
            raise ValueError(f"not enough activation rows to calibrate layer {layer}")
        variance = (m2s[layer] / (counts[layer] - 1)).float()
        factors = sketches[layer].factors(counts[layer] - 1)
        represented_diagonal = factors.square().sum(dim=0)
        mean_variance = variance.mean().clamp_min(1e-8)
        floor = mean_variance * 1e-6
        residual_diagonal = (variance - represented_diagonal).clamp_min(floor)
        diagonal = (1 - shrinkage) * residual_diagonal + shrinkage * mean_variance
        factors = factors * math.sqrt(1 - shrinkage)
        geometry[layer] = CovarianceMetric(
            diagonal=diagonal,
            factors=factors,
            calibrated=True,
        )

    metadata = dict(lens.metadata)
    metadata.update(
        {
            "geometry_prompts": str(len(prompts)),
            "geometry_rank": str(rank),
            "geometry_shrinkage": str(shrinkage),
        }
    )
    return JacobianLens(
        jacobians=lens.jacobians,
        sketches=lens.sketches,
        n_prompts=lens.n_prompts,
        d_model=lens.d_model,
        geometry=geometry,
        metadata=metadata,
    )
