from jstudio.services.hf_runtime import DEFAULT_MODEL_ID
from jstudio.ui import app as app_module


def test_normal_launch_selects_real_cached_qwen(monkeypatch):
    sentinel = object()
    calls = []
    monkeypatch.setattr(
        app_module,
        "create_hf_services",
        lambda model_id, local_files_only, lens_path: (
            calls.append((model_id, local_files_only, lens_path)) or sentinel
        ),
    )
    args = app_module._arguments([])

    result = app_module.select_services(args)

    assert result is sentinel
    assert calls == [(DEFAULT_MODEL_ID, True, None)]


def test_model_argument_defaults_to_startup_picker():
    args = app_module._arguments([])

    assert args.model is None


def test_startup_picker_choice_selects_model_and_lens(monkeypatch, tmp_path):
    sentinel = object()
    lens = tmp_path / "lens.pt"
    calls = []
    monkeypatch.setattr(
        app_module,
        "create_hf_services",
        lambda model_id, local_files_only, lens_path: (
            calls.append((model_id, local_files_only, lens_path)) or sentinel
        ),
    )
    args = app_module._arguments([])
    args.model = "Qwen/Qwen2.5-7B-Instruct"
    args.lens = lens

    result = app_module.select_services(args)

    assert result is sentinel
    assert calls == [("Qwen/Qwen2.5-7B-Instruct", True, lens)]


def test_demo_mode_is_explicit(monkeypatch):
    sentinel = object()
    monkeypatch.setattr(app_module, "create_fake_services", lambda: sentinel)
    args = app_module._arguments(["--demo"])

    assert app_module.select_services(args) is sentinel


def test_model_flag_allows_hub_downloads_only_when_requested(monkeypatch):
    calls = []
    monkeypatch.setattr(
        app_module,
        "create_hf_services",
        lambda model_id, local_files_only, lens_path: calls.append(
            (model_id, local_files_only, lens_path)
        ),
    )
    args = app_module._arguments(["--model", "Qwen/alternate", "--allow-download"])

    app_module.select_services(args)

    assert calls == [("Qwen/alternate", False, None)]
