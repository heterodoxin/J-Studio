"""Optional adapter for the separately installed :mod:`jlens` backend.

The desktop package intentionally does not depend on PyTorch or Transformers.
This boundary imports the research backend lazily and performs its blocking work
on a small executor so the presentation layer never needs those implementation
details.
"""

from __future__ import annotations

import importlib
from concurrent.futures import Future, ThreadPoolExecutor
from types import ModuleType
from typing import Any


class JLENSUnavailableError(RuntimeError):
    """Raised when an operation requires an unavailable J-Lens installation."""


class JLensAdapter:
    """Lazy, backend-neutral bridge to the calibrated ``jlens`` package."""

    def __init__(self, module_name: str = "jlens", *, max_workers: int = 1) -> None:
        self.module_name = module_name
        self._module: ModuleType | None = None
        self._model: Any | None = None
        self._lens: Any | None = None
        self._executor = ThreadPoolExecutor(
            max_workers=max_workers, thread_name_prefix="jstudio-jlens"
        )
        try:
            self._module = importlib.import_module(module_name)
            self.unavailable_reason: str | None = None
        except (ImportError, OSError) as exc:
            self.unavailable_reason = f"Could not import {module_name}: {exc}"

    @property
    def available(self) -> bool:
        return self._module is not None

    @property
    def bound(self) -> bool:
        return self.available and self._model is not None and self._lens is not None

    def require_available(self) -> ModuleType:
        if self._module is None:
            raise JLENSUnavailableError(
                self.unavailable_reason or f"{self.module_name} is unavailable"
            )
        return self._module

    def bind(self, model: Any, lens: Any) -> None:
        self.require_available()
        self._model = model
        self._lens = lens

    def require_bound(self) -> tuple[Any, Any]:
        self.require_available()
        if self._model is None or self._lens is None:
            raise JLENSUnavailableError("A model and compatible J-lens must be loaded")
        return self._model, self._lens

    def submit_read(self, prompt: str, **options: Any) -> Future:
        model, lens = self.require_bound()
        return self._executor.submit(lens.apply, model, prompt, **options)

    def submit_intervention(
        self, operation: str, prompt: str, concept: Any, **options: Any
    ) -> Future:
        module = self.require_available()
        model, lens = self.require_bound()

        def run():
            engine = module.InterventionEngine(model, lens)
            method = getattr(engine, operation, None)
            if operation not in {"inject", "replace"} or not callable(method):
                raise ValueError(f"unsupported intervention operation: {operation}")
            return method(prompt, concept, **options)

        return self._executor.submit(run)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
