# Copyright 2026 Anthropic PBC
# SPDX-License-Identifier: Apache-2.0

from scripts.fit_decoder_lens import arguments, select_source_layers


def test_progressive_cli_defaults(tmp_path):
    args = arguments(["--output", str(tmp_path / "lens.pt")])
    assert args.stage == "stable"
    assert args.prompt_source == "evaluations"
    assert args.prompts is None
    assert args.rank is None
    assert args.dim_batch is None


def test_source_layers_are_evenly_bounded():
    layers = select_source_layers(target_layer=34, limit=25)
    assert layers == sorted(set(layers))
    assert len(layers) <= 25
    assert layers[0] == 0
    assert layers[-1] == 33
