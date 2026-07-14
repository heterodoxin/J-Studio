# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
from torch import nn

from jlens.hooks import (
    ActivationEditor,
    ResidualEdit,
    ResidualTransform,
    ResidualTransformEditor,
)


def test_editor_changes_only_selected_position_and_batch():
    block = nn.Identity()
    value = torch.zeros(2, 4, 3)
    edit = ResidualEdit(
        layer=0,
        positions=(1,),
        delta=torch.tensor([1.0, 2.0, 3.0]),
        batch_indices=(0,),
    )

    with ActivationEditor([block], [edit]):
        output = block(value)

    torch.testing.assert_close(output[0, 1], edit.delta)
    assert torch.count_nonzero(output[0, [0, 2, 3]]) == 0
    assert torch.count_nonzero(output[1]) == 0
    assert torch.count_nonzero(value) == 0


class _TupleBlock(nn.Module):
    def forward(self, value):
        return value, "cache"


def test_editor_preserves_tuple_auxiliary_outputs():
    block = _TupleBlock()
    edit = ResidualEdit(layer=0, positions=(-1,), delta=torch.ones(3))

    with ActivationEditor([block], [edit]):
        residual, cache = block(torch.zeros(1, 2, 3))

    torch.testing.assert_close(residual[0, -1], torch.ones(3))
    assert cache == "cache"


def test_editor_combines_multiple_edits_on_one_layer():
    block = nn.Identity()
    edits = [
        ResidualEdit(0, (0,), torch.tensor([1.0, 0.0])),
        ResidualEdit(0, (0,), torch.tensor([0.0, 2.0])),
    ]

    with ActivationEditor([block], edits):
        output = block(torch.zeros(1, 2, 2))

    torch.testing.assert_close(output[0, 0], torch.tensor([1.0, 2.0]))


def test_editor_can_apply_edit_only_once_for_generation_prefill():
    block = nn.Identity()
    edit = ResidualEdit(
        layer=0,
        positions=(-1,),
        delta=torch.ones(2),
        max_applications=1,
    )

    with ActivationEditor([block], [edit]):
        prefill = block(torch.zeros(1, 3, 2))
        decode = block(torch.zeros(1, 1, 2))

    torch.testing.assert_close(prefill[0, -1], torch.ones(2))
    assert torch.count_nonzero(decode) == 0


def test_editor_restores_hooks_after_exception():
    block = nn.Identity()

    with pytest.raises(RuntimeError, match="boom"):
        with ActivationEditor([block], [ResidualEdit(0, (-1,), torch.ones(3))]):
            raise RuntimeError("boom")

    assert len(block._forward_hooks) == 0


def test_editor_rejects_invalid_position_during_forward():
    block = nn.Identity()

    with ActivationEditor([block], [ResidualEdit(0, (3,), torch.ones(2))]):
        with pytest.raises(IndexError, match="position"):
            block(torch.zeros(1, 2, 2))


def test_editor_rejects_invalid_layer_before_registering_hooks():
    block = nn.Identity()

    with pytest.raises(ValueError, match="layer"):
        ActivationEditor([block], [ResidualEdit(2, (0,), torch.ones(2))])

    assert len(block._forward_hooks) == 0


def test_transform_editor_recomputes_last_position_for_cached_decode():
    block = nn.Identity()
    transform = ResidualTransform(0, (-1,), lambda residual: residual * 0)

    with ResidualTransformEditor([block], [transform]):
        prefill = block(torch.ones(1, 3, 4))
        decode = block(torch.ones(1, 1, 4))

    assert torch.count_nonzero(prefill[0, -1]) == 0
    assert torch.count_nonzero(decode[0, -1]) == 0
    torch.testing.assert_close(prefill[0, 0], torch.ones(4))


def test_transform_editor_composes_transforms_in_stack_order():
    block = nn.Identity()
    transforms = (
        ResidualTransform(0, (0,), lambda residual: residual + 1),
        ResidualTransform(0, (0,), lambda residual: residual * 3),
    )

    with ResidualTransformEditor([block], transforms):
        output = block(torch.ones(1, 1, 2))

    torch.testing.assert_close(output, torch.full((1, 1, 2), 6.0))


def test_transform_editor_restores_hooks_after_exception():
    block = nn.Identity()
    transform = ResidualTransform(0, (-1,), lambda residual: residual)

    with pytest.raises(RuntimeError, match="boom"):
        with ResidualTransformEditor([block], [transform]):
            raise RuntimeError("boom")

    assert len(block._forward_hooks) == 0


def test_transform_editor_rejects_nonfinite_output():
    block = nn.Identity()
    transform = ResidualTransform(
        0, (-1,), lambda residual: torch.full_like(residual, float("nan"))
    )

    with ResidualTransformEditor([block], [transform]):
        with pytest.raises(ValueError, match="finite"):
            block(torch.ones(1, 1, 2))
