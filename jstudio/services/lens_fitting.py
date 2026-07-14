"""Shared-model coordination and progressive lens fitting lifecycle."""

from __future__ import annotations

import json
import shutil
import threading
import time
from collections.abc import Callable
from contextlib import contextmanager
from pathlib import Path
from typing import Any

from jstudio.domain import LensFitState, LensFitStatus


class GPUCoordinator:
    """Reentrant ownership boundary for operations using the shared model."""

    def __init__(self) -> None:
        self._lock = threading.RLock()
        self._condition = threading.Condition(threading.RLock())
        self._active_generations = 0

    @contextmanager
    def exclusive(self, operation: str):
        if operation.startswith("lens-fit"):
            with self._condition:
                while self._active_generations:
                    self._condition.wait(timeout=0.1)
        with self._lock:
            yield

    @contextmanager
    def generation(self):
        with self._condition:
            self._active_generations += 1
        try:
            with self._lock:
                yield
        finally:
            with self._condition:
                self._active_generations -= 1
                self._condition.notify_all()


class _FitCancelled(Exception):
    pass


class RuntimeProgressiveFitter:
    """Adapter from the app runtime to the backend progressive fitter."""

    def __init__(self, runtime, checkpoint_dir: str | Path) -> None:
        self.runtime = runtime
        self.checkpoint_dir = Path(checkpoint_dir)

    @staticmethod
    def _source_layers(target: int, limit: int = 25) -> list[int]:
        count = min(limit, target)
        if count == 1:
            return [target - 1]
        return sorted(
            {
                round(index * (target - 1) / (count - 1))
                for index in range(count)
            }
        )

    #: Bounds on the auto-fit wall-clock: prompts fitted and residual directions probed.
    FIT_PROMPTS = 16
    PROJECTION_RANK = 1024
    MAX_SEQ_LEN = 128
    SKIP_FIRST = 16

    @staticmethod
    def _readout_quality(results):
        from jlens.evaluation import FitQuality

        total = len(results)
        passed = sum(bool(result.success) for result in results)
        finite = bool(total) and all(
            isinstance(result.best_rank, int) and result.best_rank >= 0
            for result in results
        )
        return FitQuality(
            passed / total if total else 0.0,
            1.0,
            finite,
            minimum_pass_at_10=1.0,
            minimum_rank_overlap=0.0,
        )

    @staticmethod
    def _items(jlens_module) -> list[dict]:
        root = Path(jlens_module.__file__).parent.parent / "data" / "evaluations"
        items = []
        for path in sorted(root.glob("*.json")):
            items.extend(json.loads(path.read_text(encoding="utf-8")).get("items", ()))
        return items

    def _fit_layers(self, target: int) -> list[int]:
        """Deep band of source layers; early layers read out poorly and only
        add cost, so the auto-fit drops the shallowest third."""
        selected = [
            layer for layer in self._source_layers(target) if layer >= int(0.3 * target)
        ]
        return selected or self._source_layers(target)

    def run(self, *, on_stage, on_progress, cancel_event) -> None:
        import jlens
        import torch
        from jlens.autotune import autotune_vjp_batch
        from jlens.evaluation import (
            select_readout_shrinkage,
            standard_readout_cases,
        )
        from jlens.fitting import sketched_jacobian_for_prompt, valid_position_mask
        from jlens.hooks import ActivationRecorder
        from jlens.lens import JacobianLens
        from jlens.progressive import FitStage, StageResult

        model = self.runtime.lens_model
        items = self._items(jlens)
        if len(items) < 8:
            raise RuntimeError("lens fitting needs at least 8 local evaluation items")
        fit_count = min(self.FIT_PROMPTS, max(2, len(items) - 2))
        prompts = [item["prompt"] for item in items[:fit_count]]
        target = model.n_layers - 1
        layers = self._fit_layers(target)
        rank = min(self.PROJECTION_RANK, model.d_model)

        def guard() -> None:
            if cancel_event.is_set():
                raise _FitCancelled

        with self.runtime.coordinator.exclusive("lens-fit"):
            tuning = autotune_vjp_batch(
                model, prompts[0], layers, target_layer=target,
                max_seq_len=self.MAX_SEQ_LEN,
            )
        dim_batch = tuning.batch_size
        # Fit the transport only within the subspace real residuals occupy.
        rows = []
        for prompt in prompts:
            with self.runtime.coordinator.exclusive("lens-fit"):
                input_ids = model.encode(prompt, max_length=self.MAX_SEQ_LEN)
                mask = valid_position_mask(input_ids.shape[1], skip_first=self.SKIP_FIRST)
                with torch.no_grad(), ActivationRecorder(model.layers, at=[target]) as rec:
                    model.forward(input_ids)
                rows.append(rec.activations[target][0, mask].detach().float().cpu())
            guard()
        _, _, right = torch.linalg.svd(torch.cat(rows), full_matrices=False)
        basis = right[: min(rank, right.shape[0])].contiguous()

        half_sums = {
            layer: [torch.zeros(basis.shape[0], model.d_model) for _ in range(2)]
            for layer in layers
        }
        half_counts = [0, 0]
        started = time.perf_counter()
        try:
            for index, prompt in enumerate(prompts):
                with self.runtime.coordinator.exclusive("lens-fit"):
                    estimates, _, _ = sketched_jacobian_for_prompt(
                        model, prompt, layers, sketch_rank=basis.shape[0],
                        target_layer=target, dim_batch=dim_batch,
                        max_seq_len=self.MAX_SEQ_LEN, skip_first=self.SKIP_FIRST,
                        _probes=basis, vjp_backend="auto",
                    )
                half = index % 2
                for layer in layers:
                    half_sums[layer][half] += estimates[layer].corrections
                half_counts[half] += 1
                on_progress("Stable", {
                    "block": 1, "blocks": 1, "prompt": index + 1,
                    "prompts": len(prompts), "successful_prompts": index + 1,
                })
                guard()
        except _FitCancelled:
            return
        if min(half_counts) == 0:
            raise RuntimeError("need at least two usable prompts to fit a lens")

        device = model.input_device if hasattr(model, "input_device") else "cpu"
        identity = torch.eye(model.d_model)

        def cross_clean(directions: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
            """Keep singular directions selected on one prompt half but scored on
            the other, so only reproducible correction is retained."""
            u, s, vh = torch.linalg.svd(directions.to(device), full_matrices=False)
            cross = ((u.T @ values.to(device)) * vh).sum(dim=1)
            return ((u * cross.clamp(min=0.0).minimum(s)) @ vh).cpu()

        jacobians = {}
        for layer in layers:
            first = half_sums[layer][0] / half_counts[0]
            second = half_sums[layer][1] / half_counts[1]
            cleaned = (cross_clean(first, second) + cross_clean(second, first)) / 2
            jacobians[layer] = identity + basis.T @ cleaned
        lens = JacobianLens(
            jacobians=jacobians, n_prompts=sum(half_counts), d_model=model.d_model,
        )
        lens.metadata.update({
            "estimator": "projected-dense-jacobian-v2",
            "projection_rank": str(basis.shape[0]),
            "shrinkage": "split-half-cross-validated",
            "model": self.runtime.model_id,
            "target_layer": str(target),
            "source_layers": ",".join(str(layer) for layer in layers),
        })
        with self.runtime.coordinator.exclusive("lens-calibration"):
            lens = jlens.calibrate_geometry(
                model, lens, prompts[: min(16, len(prompts))],
                max_seq_len=self.MAX_SEQ_LEN, rank=16,
            )
        with self.runtime.coordinator.exclusive("lens-validation"):
            cases = standard_readout_cases(model, max_rank=100)
            lens, readout_results = select_readout_shrinkage(model, lens, cases)
        quality = self._readout_quality(readout_results)
        lens.metadata.update({
            "fit_quality_metric": "reference-viewing-pass-rate-v2",
            "viewing_passed": str(sum(result.success for result in readout_results)),
            "viewing_total": str(len(readout_results)),
            "viewing_best_ranks": ",".join(
                str(result.best_rank) for result in readout_results
            ),
        })
        lens.metadata["quality_stage"] = "Stable"
        on_stage(StageResult(
            stage=FitStage("Stable", len(prompts), basis.shape[0], 1, self.MAX_SEQ_LEN),
            lens=lens, quality=quality,
            elapsed_seconds=time.perf_counter() - started, checkpoint_path=None,
        ))


class ProgressiveLensController:
    """Run a progressive fitter in the background and atomically activate stages."""

    def __init__(self, runtime, fitter, cache_dir: str | Path) -> None:
        self.runtime = runtime
        self.fitter = fitter
        self.cache_dir = Path(cache_dir)
        self._status = LensFitStatus(LensFitState.MISSING, "", 0, 0)
        self._lock = threading.Lock()
        self._callbacks: list[Callable[[LensFitStatus], None]] = []
        self._cancel = threading.Event()
        self._thread: threading.Thread | None = None
        self._started_at = 0.0

    def status(self) -> LensFitStatus:
        with self._lock:
            return self._status

    def subscribe(
        self, callback: Callable[[LensFitStatus], None]
    ) -> Callable[[], None]:
        with self._lock:
            self._callbacks.append(callback)

        def unsubscribe() -> None:
            with self._lock:
                if callback in self._callbacks:
                    self._callbacks.remove(callback)

        return unsubscribe

    def _publish(self, status: LensFitStatus) -> None:
        with self._lock:
            self._status = status
            callbacks = tuple(self._callbacks)
        for callback in callbacks:
            callback(status)

    def start(self, *, force: bool = False) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            if force:
                checkpoint_dir = getattr(self.fitter, "checkpoint_dir", None)
                if checkpoint_dir is not None:
                    shutil.rmtree(checkpoint_dir, ignore_errors=True)
            self._cancel.clear()
            self._started_at = time.perf_counter()
            self._thread = threading.Thread(
                target=self._run, name="jstudio-lens-fit", daemon=True
            )
            thread = self._thread
        self._publish(LensFitStatus(LensFitState.WAITING, "Preview", 0, 8))
        thread.start()

    def cancel(self) -> None:
        self._cancel.set()
        current = self.status()
        self._publish(
            LensFitStatus(
                LensFitState.WAITING,
                current.stage,
                current.completed,
                current.total,
                current.elapsed_seconds,
                current.quality,
                "Fitting paused at the next prompt boundary",
            )
        )

    def join(self, timeout: float | None = None) -> None:
        with self._lock:
            thread = self._thread
        if thread is not None:
            thread.join(timeout)

    def _elapsed(self) -> float:
        return max(0.0, time.perf_counter() - self._started_at)

    def _on_progress(self, stage: str, progress: dict[str, int]) -> None:
        blocks = progress.get("blocks", 1)
        prompts = progress.get("prompts", 1)
        block = progress.get("block", 1)
        prompt = progress.get("prompt", 0)
        completed = min((block - 1) * prompts + prompt, blocks * prompts)
        state = LensFitState.REFINING if stage != "Preview" else LensFitState.WAITING
        self._publish(
            LensFitStatus(
                state,
                stage,
                completed,
                blocks * prompts,
                self._elapsed(),
            )
        )

    def _on_stage(self, result: Any) -> None:
        is_preview = result.name == "Preview"
        accepted = result.name == "Stable" and bool(result.quality.stable)
        result.lens.metadata.update(
            {
                "quality_gate_version": (
                    "jspace-viewing-v2"
                    if result.lens.metadata.get("fit_quality_metric")
                    == "reference-viewing-pass-rate-v2"
                    else "jspace-v1"
                ),
                "fit_quality_pass_at_10": f"{result.quality.pass_at_10:.6g}",
                "fit_quality_rank_overlap": f"{result.quality.rank_overlap:.6g}",
            }
        )
        if is_preview or accepted:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
            path = self.cache_dir / f"{result.name.lower()}.lens.pt"
            if accepted and hasattr(self.runtime, "stable_lens_path"):
                path = self.runtime.stable_lens_path
            self.runtime.activate_lens(result.lens, path, result.name)
        state = (
            LensFitState.WAITING
            if is_preview
            else LensFitState.STABLE
            if accepted
            else LensFitState.REFINING
        )
        detail = "" if accepted else "; ".join(result.quality.reasons)
        quality = (
            "passed"
            if result.quality.stable
            else "stable-required"
            if is_preview
            else "failed"
        )
        self._publish(
            LensFitStatus(
                state,
                result.name,
                result.stage.prompts,
                result.stage.prompts,
                self._elapsed(),
                quality,
                detail,
            )
        )

    def _run(self) -> None:
        try:
            self.fitter.run(
                on_stage=self._on_stage,
                on_progress=self._on_progress,
                cancel_event=self._cancel,
            )
            current = self.status()
            if self._cancel.is_set() or current.state in {
                LensFitState.STABLE,
            }:
                return
            self._publish(
                LensFitStatus(
                    LensFitState.FAILED,
                    current.stage,
                    current.completed,
                    current.total,
                    self._elapsed(),
                    current.quality,
                    current.detail or "Stable quality gates did not pass",
                )
            )
        except Exception as exc:
            current = self.status()
            self._publish(
                LensFitStatus(
                    LensFitState.FAILED,
                    current.stage,
                    current.completed,
                    current.total,
                    self._elapsed(),
                    "failed",
                    str(exc),
                )
            )
