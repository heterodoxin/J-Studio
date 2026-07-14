# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0
"""Forward-hook context manager for capturing the residual stream."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class ResidualEdit:
    """An additive edit to selected positions in one residual block output."""

    layer: int
    positions: tuple[int, ...]
    delta: torch.Tensor
    batch_indices: tuple[int, ...] | None = None
    max_applications: int | None = None

    def __post_init__(self) -> None:
        if not self.positions:
            raise ValueError("positions must not be empty")
        if self.delta.ndim != 1:
            raise ValueError("delta must have shape [d_model]")
        if not torch.isfinite(self.delta).all():
            raise ValueError("delta must be finite")
        if self.batch_indices is not None and not self.batch_indices:
            raise ValueError("batch_indices must be None or non-empty")
        if self.max_applications is not None and self.max_applications <= 0:
            raise ValueError("max_applications must be positive or None")


@dataclass(frozen=True)
class ResidualTransform:
    """A state-dependent residual transform applied at selected positions."""

    layer: int
    positions: tuple[int, ...]
    transform: Callable[[torch.Tensor], torch.Tensor]
    batch_indices: tuple[int, ...] | None = None
    max_applications: int | None = None

    def __post_init__(self) -> None:
        if not self.positions:
            raise ValueError("positions must not be empty")
        if not callable(self.transform):
            raise TypeError("transform must be callable")
        if self.batch_indices is not None and not self.batch_indices:
            raise ValueError("batch_indices must be None or non-empty")
        if self.max_applications is not None and self.max_applications <= 0:
            raise ValueError("max_applications must be positive or None")


class ActivationEditor:
    """Scoped forward hooks that add residual edits without mutating inputs.

    Multiple edits targeting one layer are combined in registration order. The
    first element of tuple/list block outputs is replaced while auxiliary cache
    values are preserved.
    """

    def __init__(
        self, blocks: Sequence[nn.Module], edits: Sequence[ResidualEdit]
    ) -> None:
        self._blocks = blocks
        self._edits_by_layer: dict[int, list[ResidualEdit]] = {}
        for edit in edits:
            if not 0 <= edit.layer < len(blocks):
                raise ValueError(
                    f"edit layer {edit.layer} out of range for {len(blocks)} layers"
                )
            self._edits_by_layer.setdefault(edit.layer, []).append(edit)
        if not self._edits_by_layer:
            raise ValueError("at least one residual edit is required")
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    @staticmethod
    def _residual(output) -> torch.Tensor:
        tensor = output if torch.is_tensor(output) else output[0]
        if not torch.is_tensor(tensor) or tensor.ndim != 3:
            raise ValueError(
                "block residual output must have shape [batch, seq, d_model]"
            )
        return tensor

    @staticmethod
    def _replace_residual(output, residual: torch.Tensor):
        if torch.is_tensor(output):
            return residual
        if isinstance(output, tuple):
            return (residual, *output[1:])
        if isinstance(output, list):
            return [residual, *output[1:]]
        raise TypeError("block output must be a tensor, tuple, or list")

    def _make_hook(self, edits: Sequence[ResidualEdit]) -> Callable[..., object]:
        remaining = [edit.max_applications for edit in edits]

        def hook(module: nn.Module, inputs, output):
            source = self._residual(output)
            edited = source.clone()
            batch_size, seq_len, d_model = edited.shape
            for edit_index, edit in enumerate(edits):
                if remaining[edit_index] == 0:
                    continue
                if edit.delta.shape[0] != d_model:
                    raise ValueError(
                        f"edit width {edit.delta.shape[0]} does not match d_model={d_model}"
                    )
                positions = []
                for position in edit.positions:
                    resolved = position + seq_len if position < 0 else position
                    if not 0 <= resolved < seq_len:
                        raise IndexError(
                            f"position {position} out of range for sequence length {seq_len}"
                        )
                    positions.append(resolved)
                batches = (
                    range(batch_size)
                    if edit.batch_indices is None
                    else edit.batch_indices
                )
                delta = edit.delta.to(device=edited.device, dtype=edited.dtype)
                for batch in batches:
                    resolved_batch = batch + batch_size if batch < 0 else batch
                    if not 0 <= resolved_batch < batch_size:
                        raise IndexError(
                            f"batch index {batch} out of range for batch size {batch_size}"
                        )
                    for position in positions:
                        edited[resolved_batch, position] += delta
                if remaining[edit_index] is not None:
                    remaining[edit_index] -= 1
            return self._replace_residual(output, edited)

        return hook

    def __enter__(self) -> ActivationEditor:
        try:
            for layer, edits in self._edits_by_layer.items():
                self._handles.append(
                    self._blocks[layer].register_forward_hook(self._make_hook(edits))
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ResidualTransformEditor:
    """Scoped hooks that compose transforms over the current residual state."""

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        transforms: Sequence[ResidualTransform],
    ) -> None:
        self._blocks = blocks
        self._transforms_by_layer: dict[int, list[ResidualTransform]] = {}
        for transform in transforms:
            if not 0 <= transform.layer < len(blocks):
                raise ValueError(
                    f"transform layer {transform.layer} out of range for "
                    f"{len(blocks)} layers"
                )
            self._transforms_by_layer.setdefault(transform.layer, []).append(transform)
        if not self._transforms_by_layer:
            raise ValueError("at least one residual transform is required")
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(
        self, transforms: Sequence[ResidualTransform]
    ) -> Callable[..., object]:
        remaining = [transform.max_applications for transform in transforms]

        def hook(module: nn.Module, inputs, output):
            source = ActivationEditor._residual(output)
            edited = source.clone()
            batch_size, seq_len, _ = edited.shape
            for transform_index, transform in enumerate(transforms):
                if remaining[transform_index] == 0:
                    continue
                positions = []
                for position in transform.positions:
                    resolved = position + seq_len if position < 0 else position
                    if not 0 <= resolved < seq_len:
                        raise IndexError(
                            f"position {position} out of range for sequence "
                            f"length {seq_len}"
                        )
                    positions.append(resolved)
                batches = (
                    range(batch_size)
                    if transform.batch_indices is None
                    else transform.batch_indices
                )
                for batch in batches:
                    resolved_batch = batch + batch_size if batch < 0 else batch
                    if not 0 <= resolved_batch < batch_size:
                        raise IndexError(
                            f"batch index {batch} out of range for batch size "
                            f"{batch_size}"
                        )
                    for position in positions:
                        current = edited[resolved_batch, position]
                        transformed = transform.transform(current)
                        if transformed.shape != current.shape:
                            raise ValueError(
                                "residual transform must preserve the residual shape"
                            )
                        if not torch.isfinite(transformed).all():
                            raise ValueError("residual transform output must be finite")
                        edited[resolved_batch, position] = transformed.to(
                            device=edited.device, dtype=edited.dtype
                        )
                if remaining[transform_index] is not None:
                    remaining[transform_index] -= 1
            return ActivationEditor._replace_residual(output, edited)

        return hook

    def __enter__(self) -> ResidualTransformEditor:
        try:
            for layer, transforms in self._transforms_by_layer.items():
                self._handles.append(
                    self._blocks[layer].register_forward_hook(
                        self._make_hook(transforms)
                    )
                )
        except Exception:
            self.__exit__()
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []


class ActivationRecorder:
    """Captures residual-stream tensors at the given block indices.

    Registers a forward hook on each requested block on ``__enter__`` and
    removes them on ``__exit__``. On the next forward pass each block's output
    is stored in :attr:`activations`, keyed by block index. Stored tensors are
    not detached, so they can be passed straight to :func:`torch.autograd.grad`.

    Args:
        blocks: The sequence of residual blocks (e.g. ``model.layers``).
        at: Block indices to record at.
        start_graph_at: If given, the captured tensor at this index is marked
            ``requires_grad_(True)`` before downstream blocks see it. When the
            model's parameters all have ``requires_grad=False``, this makes the
            captured residual the leaf that roots the autograd graph, so the
            retained graph spans only this block onward.
    """

    def __init__(
        self,
        blocks: Sequence[nn.Module],
        at: Iterable[int],
        *,
        start_graph_at: int | None = None,
    ) -> None:
        self._blocks = blocks
        self._indices = sorted(set(at))
        self._start_graph_at = start_graph_at
        if start_graph_at is not None and start_graph_at not in self._indices:
            self._indices = sorted({*self._indices, start_graph_at})
        self.activations: dict[int, torch.Tensor] = {}
        self._handles: list[torch.utils.hooks.RemovableHandle] = []

    def _make_hook(self, index: int) -> Callable[..., None]:
        is_graph_root = index == self._start_graph_at

        def hook(module: nn.Module, inputs, output) -> None:
            # Some HF blocks return a tuple (hidden, present_kv, ...).
            tensor = output if torch.is_tensor(output) else output[0]
            if is_graph_root:
                tensor.requires_grad_(True)
            self.activations[index] = tensor

        return hook

    def __enter__(self) -> ActivationRecorder:
        try:
            for index in self._indices:
                self._handles.append(
                    self._blocks[index].register_forward_hook(self._make_hook(index))
                )
        except Exception:
            for handle in self._handles:
                handle.remove()
            self._handles = []
            raise
        return self

    def __exit__(self, *exc) -> None:
        for handle in self._handles:
            handle.remove()
        self._handles = []
