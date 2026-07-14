import pytest
import torch

import jlens
from jlens.concepts import compact_basis, normalize_directions, sequence_alignment


def test_compact_basis_removes_duplicate_directions():
    directions = torch.tensor(
        [[1.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    )

    basis = compact_basis(directions)

    assert basis.shape == (3, 2)
    torch.testing.assert_close(basis.T @ basis, torch.eye(2))


def test_normalize_directions_rejects_nonfinite_and_zero_rows():
    with pytest.raises(ValueError, match="finite"):
        normalize_directions(torch.tensor([[float("nan"), 0.0]]))
    with pytest.raises(ValueError, match="zero"):
        normalize_directions(torch.zeros(1, 2))


@pytest.mark.parametrize("source_count,target_count", [(1, 3), (2, 4), (4, 2)])
def test_alignment_normalizes_unequal_length_columns(source_count, target_count):
    alignment = sequence_alignment(source_count, target_count)

    assert alignment.shape == (target_count, source_count)
    torch.testing.assert_close(alignment.sum(0), torch.ones(source_count))
    assert torch.all(alignment >= 0)


def test_alignment_preserves_equal_length_token_order():
    torch.testing.assert_close(sequence_alignment(3, 3), torch.eye(3))


def test_alignment_rejects_empty_sequences():
    with pytest.raises(ValueError, match="positive"):
        sequence_alignment(0, 2)


def test_phrase_operator_is_part_of_public_api():
    assert jlens.PhraseResidualOperator.__name__ == "PhraseResidualOperator"
