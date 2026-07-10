from PySide6.QtCore import Qt

from jstudio.domain import BackendKind, ModelSessionSummary, SessionState
from jstudio.services.fake import create_fake_services
from jstudio.ui.app import create_application


def test_create_application_builds_ready_demo_window(qapp):
    services = create_fake_services(token_delay=0.001)
    app, window = create_application([], services=services, application=qapp)

    assert app is qapp
    assert window.windowTitle() == "J Studio"
    assert window.current_session is not None
    assert window.tabs.currentWidget() is window.main_workspace
    assert not app.windowIcon().isNull()
    assert not window.windowIcon().isNull()
    services.generation.close()


def test_first_launch_prompt_stream_pause_and_stop(qtbot, qapp):
    services = create_fake_services(token_delay=0.03)
    _, window = create_application([], services=services, application=qapp)
    qtbot.addWidget(window)
    window.show()
    workspace = window.main_workspace
    workspace.prompt.setPlainText("Summarize the results")

    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: workspace.activation_model.rowCount() == 8)
    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)
    assert workspace.first_read.text() == "Resume"
    qtbot.mouseClick(workspace.next_read, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(workspace.undo_read, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: workspace.first_read.text() == "First Read")
    services.generation.close()


def test_missing_lens_disables_analysis_with_recovery_action(qtbot, window):
    missing = ModelSessionSummary(
        session_id="local:no-lens",
        model_id="model/no-lens",
        revision="main",
        lens_id=None,
        layer_count=32,
        backend_kind=BackendKind.LOCAL,
        state=SessionState.READY,
    )

    window.set_session(missing)

    assert not window.commands["jlens_sweep"].isEnabled()
    assert "compatible J-lens" in window.commands["jlens_sweep"].statusTip()
    assert window.commands["load_lens"].isEnabled()


def test_main_run_opens_directly_in_jlens(qtbot, window):
    workspace = window.main_workspace
    workspace.prompt.setPlainText("inspect this prompt")
    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: workspace.last_run_id is not None)
    qtbot.waitUntil(lambda: len(window.services.lens.frames(workspace.last_run_id)) > 0)

    qtbot.mouseClick(workspace.open_jlens, Qt.MouseButton.LeftButton)

    assert window.tabs.currentWidget() is window.jlens_workspace
    assert window.jlens_workspace.run_id == workspace.last_run_id


def test_opening_jlens_tab_imports_current_main_run(qtbot, window):
    workspace = window.main_workspace
    workspace.prompt.setPlainText("current main prompt")
    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: workspace.last_run_id is not None)
    qtbot.waitUntil(lambda: bool(workspace.output_text))

    window.tabs.setCurrentWidget(window.jlens_workspace)

    qtbot.waitUntil(lambda: window.jlens_workspace.run_id == workspace.last_run_id)
    assert "current main prompt" in window.jlens_workspace._request.text
    assert workspace.output_text in window.jlens_workspace._request.text


def test_jlens_source_text_streams_with_main_generation(qtbot, window):
    workspace = window.main_workspace
    workspace.prompt.setPlainText("live j lens prompt")
    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: workspace.last_run_id is not None)
    qtbot.mouseClick(workspace.open_jlens, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: bool(workspace.output_text))

    assert workspace.output_text in window.jlens_workspace._request.text


def test_keyboard_shortcuts_focus_core_workflows(qtbot, window):
    qtbot.keyClick(window, Qt.Key.Key_2, Qt.KeyboardModifier.ControlModifier)
    assert window.tabs.currentWidget() is window.rules_workspace

    window.tabs.setCurrentWidget(window.main_workspace)
    window.main_workspace.prompt.setPlainText("keyboard prompt")
    qtbot.keyClick(window, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    qtbot.waitUntil(lambda: window.main_workspace.current_run_id is not None)
    qtbot.keyClick(window, Qt.Key.Key_F10)
    assert window.main_workspace.run_state.value == "paused"

    qtbot.keyClick(window, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert window.main_workspace.run_state.value == "running"

    qtbot.keyClick(window, Qt.Key.Key_Return, Qt.KeyboardModifier.ControlModifier)
    assert window.main_workspace.run_state.value == "paused"
