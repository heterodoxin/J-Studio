from types import SimpleNamespace

import pytest
import torch
from torch import nn

from jlens.baking import ProjectionBakeRule, bake_projection, save_projection_bake
from jlens.lens import JacobianLens


class Block(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.self_attn = SimpleNamespace(o_proj=nn.Linear(d_model, d_model, bias=False))
        self.mlp = SimpleNamespace(down_proj=nn.Linear(d_model, d_model, bias=False))


class Model:
    def __init__(self):
        self.layers = nn.ModuleList([Block(4), Block(4)])
        self._lm_head = nn.Linear(4, 8, bias=False)
        self.layout = SimpleNamespace(path="model")
        self.tokenizer = SimpleNamespace(decode=lambda ids: str(ids[0]))


def test_suppress_bake_exports_projection_without_mutating_model():
    torch.manual_seed(3)
    model = Model()
    lens = JacobianLens({1: torch.eye(4)}, n_prompts=4, d_model=4)
    original = model.layers[1].self_attn.o_proj.weight.detach().clone()

    baked = bake_projection(
        model,
        lens,
        (ProjectionBakeRule("suppress", source=2, strength=0.5, layers=(1,)),),
    )

    key = "model.layers.1.self_attn.o_proj.weight"
    assert key in baked.tensors
    assert not torch.equal(baked.tensors[key], original)
    torch.testing.assert_close(model.layers[1].self_attn.o_proj.weight, original)
    assert baked.manifest["method"] == "jspace-residual-projection-v1"
    assert baked.manifest["modified_parameters"] == 2


def test_projection_bake_rejects_additive_injection():
    model = Model()
    lens = JacobianLens({1: torch.eye(4)}, n_prompts=4, d_model=4)

    with pytest.raises(ValueError, match="cannot be faithfully baked"):
        bake_projection(
            model,
            lens,
            (ProjectionBakeRule("inject", target=2, layers=(1,)),),
        )


def test_projection_bake_saves_weights_and_manifest(tmp_path):
    model = Model()
    lens = JacobianLens({1: torch.eye(4)}, n_prompts=4, d_model=4)
    baked = bake_projection(
        model,
        lens,
        (ProjectionBakeRule("suppress", source=2, layers=(1,)),),
    )

    weights, manifest = save_projection_bake(tmp_path / "edit.safetensors", baked)

    assert weights.is_file()
    assert manifest.is_file()
    assert "jspace-residual-projection-v1" in manifest.read_text(encoding="utf-8")
