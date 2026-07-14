# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from contextlib import contextmanager

import pytest
import torch

from jlens import fitting
from jlens.fitting import (
    fit,
    fit_sketch,
    jacobian_for_prompt,
    sketched_jacobian_for_prompt,
    valid_position_mask,
)
from jlens.lens import JacobianLens

from .tiny import TinyDecoder


def test_valid_position_mask_basic():
    mask = valid_position_mask(32, skip_first=4)
    assert mask.dtype == torch.bool
    assert mask[:4].sum() == 0  # leading attention-sinks excluded
    assert not mask[-1]  # final position excluded
    assert mask[4:-1].all()
    assert mask.sum() == 32 - 4 - 1


def test_valid_position_mask_too_short():
    with pytest.raises(ValueError, match="too short"):
        valid_position_mask(5, skip_first=8)


def test_jacobian_for_prompt_tiny():
    """End-to-end on a 4-layer CPU model: shapes + late-layer diag ~= 1.

    Run with all parameters ``requires_grad=False``: the recorder's
    ``start_graph_at`` must root the autograd graph itself.
    """
    model = TinyDecoder(n_layers=4, d_model=8)
    for param in model.parameters():
        param.requires_grad_(False)
    prompt = "the quick brown fox " * 4  # > SKIP_FIRST_N_POSITIONS chars
    jacobians, seq_len, n_valid = jacobian_for_prompt(
        model, prompt, source_layers=[0, 1, 2], dim_batch=4, max_seq_len=64
    )
    assert set(jacobians) == {0, 1, 2}
    for J in jacobians.values():
        assert J.shape == (8, 8) and J.dtype == torch.float32
    assert n_valid > 0 and seq_len > n_valid
    # Residual block is h + 0.1*W*h, so J_{n_layers-2} = I + 0.1*W -> diag ~= 1.
    diag_late = jacobians[2].diag()
    assert (diag_late - 1.0).abs().max() < 0.2
    # Earlier layers compound through more blocks -> further from identity.
    assert (jacobians[0] - torch.eye(8)).norm() > (jacobians[2] - torch.eye(8)).norm()
    # Block 3 is h + W_3 h, so J_2 == I + W_3 exactly — pins orientation/indexing.
    expected_J2 = torch.eye(8) + model.layers[3].linear.weight.detach()
    torch.testing.assert_close(jacobians[2], expected_J2, rtol=0, atol=1e-5)


def test_full_rank_sketch_matches_dense_jacobian_transport():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompt = "the quick brown fox " * 4
    dense, _, _ = jacobian_for_prompt(
        model, prompt, source_layers=[0, 2], dim_batch=4, max_seq_len=64
    )
    sketches, _, _ = sketched_jacobian_for_prompt(
        model,
        prompt,
        source_layers=[0, 2],
        sketch_rank=8,
        dim_batch=4,
        max_seq_len=64,
        seed=7,
    )
    residual = torch.randn(3, 8)

    for layer in (0, 2):
        expected = residual @ dense[layer].T
        actual = sketches[layer].apply(residual)
        torch.testing.assert_close(actual, expected, rtol=2e-5, atol=2e-5)


@pytest.mark.parametrize("dim_batch", [1, 2, 4])
def test_batched_vjp_matches_replicated(dim_batch):
    model = TinyDecoder(n_layers=4, d_model=8)
    kwargs = dict(
        source_layers=[0, 2],
        sketch_rank=8,
        dim_batch=dim_batch,
        max_seq_len=64,
        seed=3,
    )
    batched, _, _ = sketched_jacobian_for_prompt(
        model, "abcdefghij " * 5, vjp_backend="batched", **kwargs
    )
    replicated, _, _ = sketched_jacobian_for_prompt(
        model, "abcdefghij " * 5, vjp_backend="replicated", **kwargs
    )
    for layer in kwargs["source_layers"]:
        torch.testing.assert_close(
            batched[layer].corrections,
            replicated[layer].corrections,
            rtol=2e-5,
            atol=2e-5,
        )


def test_auto_vjp_falls_back_to_replicated(monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("Batching rule not implemented")

    monkeypatch.setattr(fitting, "_batched_vjp", fail)
    result, _, _ = sketched_jacobian_for_prompt(
        TinyDecoder(),
        "abcdefghij " * 5,
        [1],
        sketch_rank=4,
        dim_batch=2,
        max_seq_len=64,
        vjp_backend="auto",
    )
    assert result[1].rank == 4


def test_fit_sketch_round_trip_and_merge(tmp_path):
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    lens = fit_sketch(
        model,
        prompts,
        source_layers=[0, 2],
        sketch_rank=4,
        dim_batch=2,
        max_seq_len=64,
        seed=11,
    )
    assert lens.source_layers == [0, 2]
    assert lens.jacobians == {}
    assert set(lens.sketches) == {0, 2}
    assert lens.metadata["target_layer"] == "3"
    assert lens.metadata["source_layers"] == "0,2"

    path = tmp_path / "sketched-lens.pt"
    lens.save(str(path))
    loaded = JacobianLens.load(str(path))
    residual = torch.randn(2, 8)
    for layer in (0, 2):
        torch.testing.assert_close(
            loaded.transport(residual, layer),
            lens.transport(residual, layer),
            rtol=0,
            atol=3e-3,
        )

    merged = JacobianLens.merge([lens, loaded])
    for layer in (0, 2):
        torch.testing.assert_close(
            merged.transport(residual, layer),
            lens.transport(residual, layer),
            rtol=0,
            atol=3e-3,
        )


def test_probe_blocks_average_every_prompt_before_combining(monkeypatch):
    model = TinyDecoder(n_layers=4, d_model=8)
    seen = []
    real = fitting.sketched_jacobian_for_prompt

    def record(model, prompt, layers, **kwargs):
        seen.append((prompt, kwargs["_probes"].clone()))
        return real(model, prompt, layers, **kwargs)

    monkeypatch.setattr(fitting, "sketched_jacobian_for_prompt", record)
    fit_sketch(
        model,
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[2],
        sketch_rank=4,
        probe_blocks=2,
        dim_batch=2,
        max_seq_len=64,
    )

    assert len(seen) == 4
    assert torch.equal(seen[0][1], seen[1][1])
    assert not torch.equal(seen[1][1], seen[2][1])
    assert torch.equal(seen[2][1], seen[3][1])


def test_fit_sketch_reports_each_prompt_block():
    seen = []
    fit_sketch(
        TinyDecoder(n_layers=4, d_model=8),
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[2],
        sketch_rank=4,
        probe_blocks=2,
        dim_batch=2,
        on_progress=seen.append,
    )
    assert [(row["block"], row["prompt"]) for row in seen] == [
        (1, 1),
        (1, 2),
        (2, 1),
        (2, 2),
    ]


def test_fit_sketch_releases_operation_between_prompts():
    entered = []

    @contextmanager
    def operation(name):
        entered.append(name)
        yield

    fit_sketch(
        TinyDecoder(n_layers=4, d_model=8),
        ["abcdefghij " * 5, "klmnopqrst " * 5],
        source_layers=[2],
        sketch_rank=4,
        probe_blocks=2,
        dim_batch=2,
        operation_context=operation,
    )
    assert entered == ["lens-fit-prompt"] * 4


def test_fit_sketch_resume_is_deterministic(tmp_path, monkeypatch):
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    checkpoint = tmp_path / "sketch-checkpoint.pt"
    real = fitting.sketched_jacobian_for_prompt
    calls = 0

    def interrupt_after_first(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise RuntimeError("simulated interruption")
        return real(*args, **kwargs)

    monkeypatch.setattr(fitting, "sketched_jacobian_for_prompt", interrupt_after_first)
    with pytest.raises(RuntimeError, match="simulated interruption"):
        fit_sketch(
            model,
            prompts,
            source_layers=[2],
            sketch_rank=4,
            probe_blocks=2,
            dim_batch=2,
            max_seq_len=64,
            checkpoint_path=str(checkpoint),
        )

    monkeypatch.setattr(fitting, "sketched_jacobian_for_prompt", real)
    resumed = fit_sketch(
        model,
        prompts,
        source_layers=[2],
        sketch_rank=4,
        probe_blocks=2,
        dim_batch=2,
        max_seq_len=64,
        checkpoint_path=str(checkpoint),
    )
    fresh = fit_sketch(
        model,
        prompts,
        source_layers=[2],
        sketch_rank=4,
        probe_blocks=2,
        dim_batch=2,
        max_seq_len=64,
        resume=False,
    )
    torch.testing.assert_close(
        resumed.sketches[2].corrections, fresh.sketches[2].corrections
    )


def test_fit_sketch_rejects_legacy_checkpoint(tmp_path):
    checkpoint = tmp_path / "legacy.pt"
    torch.save({"format": "independent-probe-blocks-v1"}, checkpoint)

    with pytest.raises(ValueError, match="incompatible sketch checkpoint"):
        fit_sketch(
            TinyDecoder(n_layers=4, d_model=8),
            ["abcdefghij " * 5],
            source_layers=[2],
            sketch_rank=4,
            checkpoint_path=str(checkpoint),
        )


def test_fit_and_apply_tiny(tmp_path):
    """fit() -> JacobianLens -> save/load -> apply() round-trip."""
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5]
    lens = fit(model, prompts, source_layers=[0, 1, 2], dim_batch=4, max_seq_len=64)
    assert lens.n_prompts == 2
    assert lens.source_layers == [0, 1, 2]
    assert lens.d_model == 8

    path = tmp_path / "lens.pt"
    lens.save(str(path))
    reloaded = JacobianLens.load(str(path))
    assert reloaded.source_layers == [0, 1, 2]
    assert reloaded.n_prompts == 2
    for layer in [0, 1, 2]:
        torch.testing.assert_close(
            reloaded.jacobians[layer], lens.jacobians[layer], rtol=0, atol=2e-3
        )  # fp16 round-trip

    lens_logits, model_logits, input_ids = reloaded.apply(
        model, "the quick brown fox jumps", layers=[0, 2]
    )
    assert set(lens_logits) == {0, 2}
    vocab_size = model.lm_head.out_features
    seq_len = input_ids.shape[1]
    # positions=None -> every position.
    assert model_logits.shape == (seq_len, vocab_size)
    for tensor in lens_logits.values():
        assert tensor.shape == (seq_len, vocab_size)
    # Exactly linear lens map: transported readout == model logits (atol: fp16 save).
    torch.testing.assert_close(lens_logits[2], model_logits, rtol=0, atol=1e-2)

    # Explicit positions (negative indices allowed) -> that many rows, in order.
    sub_logits, sub_model, _ = reloaded.apply(
        model, "the quick brown fox jumps", layers=[0, 2], positions=[0, -1]
    )
    assert sub_model.shape == (2, vocab_size)
    torch.testing.assert_close(sub_model[1], model_logits[-1])
    for layer in [0, 2]:
        assert sub_logits[layer].shape == (2, vocab_size)
        torch.testing.assert_close(sub_logits[layer][0], lens_logits[layer][0])

    # Logit-lens baseline path (use_jacobian=False) also works.
    baseline, _, _ = reloaded.apply(
        model, "hello world test", layers=[1], positions=[-1], use_jacobian=False
    )
    assert baseline[1].shape == (1, vocab_size)
    # Unfitted layer is rejected.
    with pytest.raises(ValueError, match="not in source_layers"):
        reloaded.apply(model, "x" * 30, layers=[3])
    # Out-of-range layers are rejected on the baseline path too.
    with pytest.raises(ValueError, match="out of range"):
        reloaded.apply(model, "x" * 30, layers=[99], use_jacobian=False)


def test_from_pretrained_local(tmp_path):
    """``from_pretrained`` resolves a file, a directory, and a directory with
    the lens at a subpath (Hub-repo style layout)."""
    lens = JacobianLens(
        jacobians={0: torch.randn(6, 6), 1: torch.randn(6, 6)}, n_prompts=3, d_model=6
    )
    # File path -> load() directly.
    single = tmp_path / "single.pt"
    lens.save(str(single))
    for layer in [0, 1]:
        torch.testing.assert_close(
            JacobianLens.from_pretrained(str(single)).jacobians[layer],
            lens.jacobians[layer],
            rtol=0,
            atol=2e-3,
        )  # fp16 round-trip
    # Directory containing lens.pt.
    one_dir = tmp_path / "one"
    one_dir.mkdir()
    lens.save(str(one_dir / "lens.pt"))
    assert JacobianLens.from_pretrained(str(one_dir)).n_prompts == 3
    # filename= may be a subpath inside the directory.
    deep = tmp_path / "hubrepo"
    sub = deep / "gemma-2-27b" / "jlens" / "wikitext"
    sub.mkdir(parents=True)
    lens.save(str(sub / "lens.pt"))
    reloaded = JacobianLens.from_pretrained(
        str(deep), filename="gemma-2-27b/jlens/wikitext/lens.pt"
    )
    assert reloaded.n_prompts == 3 and reloaded.d_model == 6


def test_merge_weighted_mean():
    """merge() is the n_prompts-weighted mean."""
    d_model = 4
    lens_a = JacobianLens(
        jacobians={
            0: torch.full((d_model, d_model), 1.0),
            1: torch.full((d_model, d_model), 2.0),
        },
        n_prompts=2,
        d_model=d_model,
    )
    lens_b = JacobianLens(
        jacobians={
            0: torch.full((d_model, d_model), 4.0),
            1: torch.full((d_model, d_model), 8.0),
        },
        n_prompts=6,
        d_model=d_model,
    )
    merged = JacobianLens.merge([lens_a, lens_b])
    assert merged.n_prompts == 8
    # (1*2 + 4*6) / 8 = 26/8 = 3.25
    torch.testing.assert_close(
        merged.jacobians[0], torch.full((d_model, d_model), 3.25)
    )
    torch.testing.assert_close(merged.jacobians[1], torch.full((d_model, d_model), 6.5))


def test_merge_mismatch_raises():
    a = JacobianLens(jacobians={0: torch.eye(4)}, n_prompts=1, d_model=4)
    b = JacobianLens(jacobians={1: torch.eye(4)}, n_prompts=1, d_model=4)
    with pytest.raises(ValueError, match="disagree"):
        JacobianLens.merge([a, b])
    with pytest.raises(ValueError, match="at least one"):
        JacobianLens.merge([])


def test_fit_checkpoint_resume(tmp_path):
    """fit() resumes from a checkpoint and produces the same result."""
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5, "klmnopqrst " * 5, "uvwxyzabcd " * 5]
    checkpoint = str(tmp_path / "ckpt.pt")

    full = fit(
        model,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    # Second call with the same checkpoint should resume past all 3 and be a no-op.
    resumed = fit(
        model,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    assert resumed.n_prompts == full.n_prompts == 3
    for layer in [0, 2]:
        torch.testing.assert_close(resumed.jacobians[layer], full.jacobians[layer])


def test_fit_resume_after_skip(tmp_path):
    """Resume after a too-short prompt was skipped must not double-count.

    Regression test: a skipped prompt used to desync the success-count from the
    list-position, so the prompt after a skip was processed again on resume.
    """
    model = TinyDecoder(n_layers=4, d_model=8)
    long_a = "abcdefghij " * 5
    short = "x"  # tokenizes to 2 tokens -> ValueError -> skip
    long_b = "klmnopqrst " * 5
    prompts = [long_a, short, long_b]
    checkpoint = str(tmp_path / "ckpt.pt")

    reference = fit(model, prompts, source_layers=[0, 2], dim_batch=4, max_seq_len=64)
    assert reference.n_prompts == 2  # short was skipped

    fit(
        model,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    resumed = fit(
        model,
        prompts,
        source_layers=[0, 2],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    assert resumed.n_prompts == 2
    for layer in [0, 2]:
        torch.testing.assert_close(resumed.jacobians[layer], reference.jacobians[layer])


def test_fit_resume_mismatched_source_layers(tmp_path):
    model = TinyDecoder(n_layers=4, d_model=8)
    prompts = ["abcdefghij " * 5]
    checkpoint = str(tmp_path / "ckpt.pt")
    fit(
        model,
        prompts,
        source_layers=[0, 1],
        dim_batch=4,
        max_seq_len=64,
        checkpoint_path=checkpoint,
    )
    with pytest.raises(ValueError, match="source_layers"):
        fit(
            model,
            prompts,
            source_layers=[0, 2],
            dim_batch=4,
            max_seq_len=64,
            checkpoint_path=checkpoint,
        )


def test_negative_layer_indices_normalized():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompt = "the quick brown fox " * 4
    jac_neg, _, _ = jacobian_for_prompt(
        model,
        prompt,
        source_layers=[-4, -3],
        target_layer=-1,
        dim_batch=4,
        max_seq_len=64,
    )
    jac_pos, _, _ = jacobian_for_prompt(
        model, prompt, source_layers=[0, 1], target_layer=3, dim_batch=4, max_seq_len=64
    )
    assert set(jac_neg) == {0, 1}
    for layer in (0, 1):
        torch.testing.assert_close(jac_neg[layer], jac_pos[layer])

    lens = fit(model, [prompt], source_layers=[-4], dim_batch=4, max_seq_len=64)
    assert lens.source_layers == [0]


def test_out_of_range_layers_rejected():
    model = TinyDecoder(n_layers=4, d_model=8)
    prompt = "the quick brown fox " * 4
    with pytest.raises(ValueError, match="out of range"):
        fit(model, [prompt], source_layers=[0, 7], dim_batch=4, max_seq_len=64)
    with pytest.raises(ValueError, match="must all be < target_layer"):
        fit(model, [prompt], source_layers=[-1], dim_batch=4, max_seq_len=64)
    with pytest.raises(ValueError, match="target_layer"):
        jacobian_for_prompt(
            model,
            prompt,
            source_layers=[0],
            target_layer=9,
            dim_batch=4,
            max_seq_len=64,
        )
