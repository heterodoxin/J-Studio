import threading
from types import SimpleNamespace

import torch

from jstudio.services.protocols import SliceRequest
from jstudio.services.slice_runtime import context_token_ids, render_slice


class RuntimeDouble:
    class LensModel:
        tokenizer = SimpleNamespace()

        def encode(self, text, *, max_length=512):
            assert text == "12+32^4! using PEMDAS"
            return torch.tensor([[12, 32, 4, 32]])

    lens_model = LensModel()
    lens = object()


def test_render_slice_uses_original_renderer(monkeypatch):
    import jlens.vis as vis

    calls = []
    monkeypatch.setattr(
        vis,
        "compute_slice",
        lambda *args, **kwargs: calls.append((args, kwargs)) or object(),
    )
    build_calls = []
    monkeypatch.setattr(
        vis,
        "build_page",
        lambda *args, **kwargs: build_calls.append(kwargs)
        or ("<html>original</html>", None, None),
    )

    page = render_slice(
        RuntimeDouble(), SliceRequest("r", "12+32^4! using PEMDAS", "Math"), 7
    )

    assert page.html == "<html>original</html>"
    assert page.generation == 7
    assert calls[0][1] == {
        "layer_stride": 1,
        "last_n_tokens": None,
        "max_tracked": 256,
        "pinned_token_ids": {4, 12, 32},
        "top_n": 10,
        "mask_display": True,
    }
    assert build_calls[0]["pinned_token_ids"] == {4, 12, 32}
    assert "context-token" in build_calls[0]["description"]


def test_context_token_ids_caps_unique_prompt_tokens():
    class LensModel:
        def encode(self, text, *, max_length=512):
            return [[1, 2, 1, 3, 4, 5]]

    assert context_token_ids(LensModel(), "any prompt", limit=3) == {1, 2, 3}


def test_render_slice_runs_on_slice_executor(monkeypatch):
    import jlens.vis as vis

    import jstudio.services.slice_runtime as module

    thread_names = []
    monkeypatch.setattr(
        vis,
        "compute_slice",
        lambda *args, **kwargs: thread_names.append(threading.current_thread().name)
        or object(),
    )
    monkeypatch.setattr(
        vis,
        "build_page",
        lambda *args, **kwargs: ("<html></html>", None, None),
    )
    service = module.SliceRendererService(RuntimeDouble())

    one = service.request_slice(SliceRequest("r1", "one", "One")).result(1)
    two = service.request_slice(SliceRequest("r2", "two", "Two")).result(1)
    service.close()

    assert (one.generation, two.generation) == (1, 2)
    assert all(name.startswith("jstudio-slice") for name in thread_names)
