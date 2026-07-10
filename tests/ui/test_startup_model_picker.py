from PySide6.QtCore import Qt

from jstudio.services.hf_cache import CachedModel, LensState
from jstudio.ui.startup import StartupModelDialog


def test_startup_model_dialog_selects_cached_model_and_detected_lens(qtbot, tmp_path):
    lens = tmp_path / "lens.pt"
    model = CachedModel(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        cache_path=tmp_path / "models--Qwen--Qwen2.5-7B-Instruct",
        lens_state=LensState.STABLE,
        lens_path=lens,
        lens_detail="Stable calibrated lens",
    )
    dialog = StartupModelDialog((model,))
    qtbot.addWidget(dialog)

    dialog.table.selectRow(0)

    assert dialog.selected_model_id() == "Qwen/Qwen2.5-7B-Instruct"
    assert dialog.selected_lens_path() == lens
    assert "Stable" in dialog.action_label.text()


def test_startup_model_dialog_browse_lens_overrides_detected_path(
    qtbot, tmp_path, monkeypatch
):
    model = CachedModel(
        model_id="Qwen/Qwen2.5-7B-Instruct",
        cache_path=tmp_path / "cache",
        lens_state=LensState.MISSING,
    )
    chosen = tmp_path / "manual.pt"
    dialog = StartupModelDialog((model,))
    qtbot.addWidget(dialog)
    monkeypatch.setattr(
        "jstudio.ui.startup.QFileDialog.getOpenFileName",
        lambda *args, **kwargs: (str(chosen), ""),
    )

    qtbot.mouseClick(dialog.browse_lens_button, Qt.MouseButton.LeftButton)

    assert dialog.selected_lens_path() == chosen
    assert dialog.lens_path.text() == str(chosen)
