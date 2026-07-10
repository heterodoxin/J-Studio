"""Lightweight Hugging Face cache and J Studio lens discovery."""

from __future__ import annotations

import os
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from jstudio.domain import DENSE_LENS_ESTIMATORS


class LensState(StrEnum):
    STABLE = "stable"
    NEEDS_FIT = "needs-fit"
    MISSING = "missing"
    UNREADABLE = "unreadable"


@dataclass(frozen=True, slots=True)
class LensInspection:
    state: LensState
    path: Path | None = None
    detail: str = ""


@dataclass(frozen=True, slots=True)
class CachedModel:
    model_id: str
    cache_path: Path
    lens_state: LensState
    lens_path: Path | None = None
    lens_detail: str = ""


def safe_model_name(model_id: str) -> str:
    return model_id.replace("/", "--")


def model_id_from_cache_dir(path: Path) -> str | None:
    name = path.name
    if not path.is_dir() or not name.startswith("models--"):
        return None
    repo = name.removeprefix("models--")
    if "--" not in repo:
        return None
    return repo.replace("--", "/")


def hf_cache_roots() -> tuple[Path, ...]:
    roots: list[Path] = []
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE"):
        value = os.environ.get(key)
        if value:
            roots.append(Path(value).expanduser())
    hf_home = Path(os.environ.get("HF_HOME", "~/.cache/huggingface")).expanduser()
    roots.append(hf_home / "hub")
    roots.append(Path("~/.cache/huggingface/hub").expanduser())
    return tuple(dict.fromkeys(roots))


def jstudio_lens_roots() -> tuple[Path, ...]:
    workspace_root = Path(__file__).resolve().parents[3]
    return (
        workspace_root / "lenses",
        Path("~/.cache/jstudio/lenses").expanduser(),
    )


def lens_candidates(
    model_id: str,
    *,
    lens_roots: tuple[Path, ...] | None = None,
) -> tuple[Path, ...]:
    roots = lens_roots if lens_roots is not None else jstudio_lens_roots()
    model_dir_name = safe_model_name(model_id)
    candidates: list[Path] = []
    for root in roots:
        model_root = root / model_dir_name
        candidates.extend(
            (
                model_root / "lens.pt",
                model_root / "stable.lens.pt",
                model_root / ".fit" / "stable.lens.pt",
            )
        )
    return tuple(dict.fromkeys(candidates))


def inspect_lens_file(path: Path) -> LensInspection:
    if not path.is_file():
        return LensInspection(LensState.MISSING)
    try:
        import torch

        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    except Exception as exc:  # pragma: no cover - depends on optional torch formats
        return LensInspection(LensState.UNREADABLE, path, f"Unreadable lens: {exc}")
    if not isinstance(checkpoint, dict):
        return LensInspection(LensState.UNREADABLE, path, "Lens file is not a dict")
    metadata = checkpoint.get("metadata", {})
    if not isinstance(metadata, dict):
        metadata = {}
    stage = metadata.get("quality_stage")
    estimator = metadata.get("estimator")
    geometry = checkpoint.get("geometry", {})
    calibrated = True
    if isinstance(geometry, dict):
        calibrated = all(
            bool(metric.get("calibrated", True))
            for metric in geometry.values()
            if isinstance(metric, dict)
        )
    # Treat dense-family lenses as usable regardless of the sketch-only pass@10 gate.
    dense_family = estimator in DENSE_LENS_ESTIMATORS
    if calibrated and dense_family:
        pass_at_10 = metadata.get("fit_quality_pass_at_10")
        suffix = f" · pass@10 {pass_at_10}" if pass_at_10 else ""
        return LensInspection(LensState.STABLE, path, f"Dense calibrated lens{suffix}")
    if stage == "Stable" and calibrated:
        rank = metadata.get("effective_rank") or metadata.get("sketch_rank")
        suffix = f" · rank {rank}" if rank else ""
        return LensInspection(LensState.STABLE, path, f"Stable calibrated lens{suffix}")
    if estimator:
        return LensInspection(
            LensState.NEEDS_FIT,
            path,
            f"Lens exists but is not a calibrated Stable lens ({stage or 'unknown'})",
        )
    return LensInspection(
        LensState.NEEDS_FIT,
        path,
        "Lens exists but lacks Stable J-space metadata",
    )


def inspect_model_lens(
    model_id: str,
    *,
    lens_roots: tuple[Path, ...] | None = None,
    explicit_lens_path: Path | None = None,
) -> LensInspection:
    candidates = []
    if explicit_lens_path is not None:
        candidates.append(explicit_lens_path)
    candidates.extend(lens_candidates(model_id, lens_roots=lens_roots))
    first_non_stable: LensInspection | None = None
    for candidate in dict.fromkeys(candidates):
        if not candidate.is_file():
            continue
        inspection = inspect_lens_file(candidate)
        if inspection.state is LensState.STABLE:
            return inspection
        if first_non_stable is None:
            first_non_stable = inspection
    return first_non_stable or LensInspection(LensState.MISSING, None, "No J Studio lens")


def scan_hf_cache(
    *,
    cache_roots: tuple[Path, ...] | None = None,
    lens_roots: tuple[Path, ...] | None = None,
) -> tuple[CachedModel, ...]:
    roots = cache_roots if cache_roots is not None else hf_cache_roots()
    discovered: dict[str, Path] = {}
    for root in roots:
        if not root.is_dir():
            continue
        for child in root.iterdir():
            model_id = model_id_from_cache_dir(child)
            if model_id is not None:
                discovered.setdefault(model_id, child)
    models: list[CachedModel] = []
    sorted_models = sorted(discovered.items(), key=lambda item: item[0].lower())
    for model_id, cache_path in sorted_models:
        lens = inspect_model_lens(model_id, lens_roots=lens_roots)
        models.append(
            CachedModel(
                model_id=model_id,
                cache_path=cache_path,
                lens_state=lens.state,
                lens_path=lens.path,
                lens_detail=lens.detail,
            )
        )
    return tuple(models)
