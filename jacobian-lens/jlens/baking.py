"""Pure-weight exports for global J-space residual projections."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Literal

import torch

from jlens.interventions import ConceptResolver
from jlens.lens import JacobianLens
from jlens.protocol import LensModel

BakeOperation = Literal["inject", "replace", "suppress"]
_RESIDUAL_WRITE_PATHS = (
    "self_attn.o_proj",
    "mlp.down_proj",
    "attn.c_proj",
    "mlp.c_proj",
    "attention.dense",
    "mlp.dense_4h_to_h",
    "self_attn.dense",
    "mlp.fc2",
)


@dataclass(frozen=True)
class ProjectionBakeRule:
    operation: BakeOperation
    source: str | int | None = None
    target: str | int | None = None
    strength: float = 1.0
    layers: tuple[int, ...] = ()

    def __post_init__(self) -> None:
        if self.operation not in {"inject", "replace", "suppress"}:
            raise ValueError(f"unsupported bake operation {self.operation!r}")
        if not 0.0 < self.strength <= 1.0:
            raise ValueError("bake strength must lie in (0, 1]")
        if self.operation in {"replace", "suppress"} and self.source is None:
            raise ValueError(f"{self.operation} bake requires a source")
        if self.operation in {"inject", "replace"} and self.target is None:
            raise ValueError(f"{self.operation} bake requires a target")


@dataclass(frozen=True)
class ProjectionBake:
    tensors: dict[str, torch.Tensor]
    manifest: dict


def _resolve_module(root, path: str):
    current = root
    for part in path.split("."):
        current = getattr(current, part, None)
        if current is None:
            return None
    return current


def _concept_direction(
    model: LensModel,
    lens: JacobianLens,
    concept: str | int,
    layer: int,
) -> torch.Tensor:
    if layer not in lens.jacobians:
        raise ValueError("projection baking currently requires a dense J operator")
    spec = ConceptResolver(model.tokenizer).resolve(concept)
    rows = model._lm_head.weight.detach().float()[list(spec.token_ids)]
    final_direction = rows.mean(dim=0)
    final_direction = final_direction / final_direction.norm().clamp_min(1e-8)
    jacobian = lens.jacobians[layer].to(final_direction.device)
    pulled = jacobian.T @ final_direction
    alpha = lens.transport_shrinkage
    pulled = final_direction + alpha * (pulled - final_direction)
    return pulled / pulled.norm().clamp_min(1e-8)


def _project_weight(
    weight: torch.Tensor,
    source: torch.Tensor,
    destination: torch.Tensor,
    strength: float,
) -> torch.Tensor:
    value = weight.detach().float()
    if value.shape[0] != source.numel():
        raise ValueError("residual write output dimension does not match the lens")
    row = source.to(value.device) @ value
    delta = destination.to(value.device) - source.to(value.device)
    return (value + strength * torch.outer(delta, row)).cpu()


def bake_projection(
    model: LensModel,
    lens: JacobianLens,
    rules: tuple[ProjectionBakeRule, ...],
) -> ProjectionBake:
    """Create modified residual-write tensors without mutating ``model``."""
    if not rules:
        raise ValueError("projection bake needs at least one rule")
    if any(rule.operation == "inject" for rule in rules):
        raise ValueError(
            "additive injection cannot be faithfully baked into ordinary linear weights"
        )
    tensors: dict[str, torch.Tensor] = {}
    applied_rules = []
    prefix = getattr(getattr(model, "layout", None), "path", "model")
    for rule in rules:
        selected_layers = rule.layers or tuple(lens.source_layers)
        unknown = sorted(set(selected_layers) - set(lens.source_layers))
        if unknown:
            raise ValueError(f"bake layers are not fitted: {unknown}")
        for layer in selected_layers:
            source = _concept_direction(model, lens, rule.source, layer)
            destination = (
                torch.zeros_like(source)
                if rule.operation == "suppress"
                else _concept_direction(model, lens, rule.target, layer)
            )
            block = model.layers[layer]
            found = 0
            for module_path in _RESIDUAL_WRITE_PATHS:
                module = _resolve_module(block, module_path)
                weight = getattr(module, "weight", None)
                if weight is None or weight.ndim != 2 or weight.shape[0] != lens.d_model:
                    continue
                key = f"{prefix}.layers.{layer}.{module_path}.weight"
                base = tensors.get(key, weight.detach().float().cpu())
                tensors[key] = _project_weight(
                    base, source.cpu(), destination.cpu(), rule.strength
                )
                found += 1
            if not found:
                raise ValueError(
                    f"no supported residual-write modules found at layer {layer}"
                )
        applied_rules.append(asdict(rule))
    return ProjectionBake(
        tensors=tensors,
        manifest={
            "method": "jspace-residual-projection-v1",
            "modified_parameters": len(tensors),
            "transport_shrinkage": lens.transport_shrinkage,
            "rules": applied_rules,
            "note": (
                "Global suppress/replace projection derived from the fitted "
                "J operator; model weights were not mutated during export."
            ),
        },
    )


def save_projection_bake(
    path: str | Path,
    bake: ProjectionBake,
    *,
    dtype: torch.dtype = torch.float16,
) -> tuple[Path, Path]:
    """Atomically save modified weights plus a neighboring JSON manifest."""
    from safetensors.torch import save_file

    destination = Path(path).expanduser()
    if destination.suffix != ".safetensors":
        destination = destination.with_suffix(".safetensors")
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    tensors = {
        name: value.detach().to(dtype).contiguous()
        for name, value in bake.tensors.items()
    }
    save_file(tensors, str(temporary))
    temporary.replace(destination)
    manifest = destination.with_suffix(".json")
    manifest_tmp = manifest.with_name(f".{manifest.name}.tmp")
    manifest_tmp.write_text(
        json.dumps(bake.manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    manifest_tmp.replace(manifest)
    return destination, manifest
