"""Asynchronous adapter for the original :mod:`jlens.vis` slice renderer."""

from __future__ import annotations

import threading
from concurrent.futures import Future, ThreadPoolExecutor

from jstudio.services.protocols import SlicePage, SliceRequest


def context_token_ids(model, text: str, *, limit: int = 256) -> set[int]:
    """Return unique token IDs from the current context for rank pinning."""
    try:
        encoded = model.encode(text, max_length=512)
    except Exception:
        return set()
    if hasattr(encoded, "detach"):
        raw_values = encoded.detach().cpu().reshape(-1).tolist()
    elif encoded and isinstance(encoded[0], list | tuple):
        raw_values = [value for row in encoded for value in row]
    else:
        raw_values = list(encoded)
    seen: set[int] = set()
    for value in raw_values:
        token_id = int(value)
        if token_id in seen:
            continue
        seen.add(token_id)
        if len(seen) >= limit:
            break
    return seen


def render_slice(runtime, request: SliceRequest, generation: int) -> SlicePage:
    from jlens.vis import build_page, compute_slice

    pinned_token_ids = context_token_ids(runtime.lens_model, request.text)
    data = compute_slice(
        runtime.lens_model,
        runtime.lens,
        request.text,
        layer_stride=request.layer_stride,
        last_n_tokens=request.last_n_tokens,
        max_tracked=256,
        pinned_token_ids=pinned_token_ids,
        top_n=request.top_n,
        mask_display=request.mask_display,
    )
    html, _, _ = build_page(
        data,
        request.text,
        title=request.title,
        description=(
            "Interactive layer-by-position Jacobian-lens readout. "
            "context-token ranks are pinned so prompt/output tokens stay inspectable."
        ),
        mode="embed",
        pinned_token_ids=pinned_token_ids,
    )
    return SlicePage(request.run_id, generation, html)


class SliceRendererService:
    def __init__(self, runtime) -> None:
        self._runtime = runtime
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jstudio-slice"
        )
        self._lock = threading.Lock()
        self._generation = 0

    def request_slice(self, request: SliceRequest) -> Future[SlicePage]:
        with self._lock:
            self._generation += 1
            generation = self._generation
        return self._executor.submit(render_slice, self._runtime, request, generation)

    def close(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
