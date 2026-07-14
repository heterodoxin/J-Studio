from PySide6.QtCore import QSize
from PySide6.QtWidgets import QDockWidget

from jstudio.ui.shell.main_window import JStudioMainWindow


def test_main_window_geometry_tabs_and_native_structure(qtbot, services, project):
    window = JStudioMainWindow(services, project)
    qtbot.addWidget(window)

    assert window.size() == QSize(1101, 888)
    assert [window.tabs.tabText(index) for index in range(4)] == [
        "Main",
        "Chat",
        "J-Lens",
        "Rules",
    ]
    assert window.tabs.currentWidget() is window.main_workspace
    assert window.findChild(QDockWidget) is None
    assert window.session_bar.height() == 46


def test_shell_has_exact_top_level_menus(qtbot, services, project):
    window = JStudioMainWindow(services, project)
    qtbot.addWidget(window)

    assert [action.text() for action in window.menuBar().actions()] == [
        "File",
        "Edit",
        "Model",
        "Table",
        "Tools",
        "Help",
    ]
    assert window._wired_commands == set(window.commands)


def test_session_selection_updates_identity_and_command_state(qtbot, services, project):
    window = JStudioMainWindow(services, project)
    qtbot.addWidget(window)
    assert not window.commands["first_read"].isEnabled()

    window.set_session(services.sessions.list_sessions()[0])

    assert "Qwen3.6-27B" in window.session_bar.identity.text()
    assert window.commands["first_read"].isEnabled()
    assert window.commands["load_lens"].isEnabled()


def test_load_lens_command_starts_or_resumes_stable_fit(qtbot, services, project):
    window = JStudioMainWindow(services, project)
    qtbot.addWidget(window)
    window.set_session(services.sessions.list_sessions()[0])

    window.commands["load_lens"].trigger()

    assert services.lens.fit_requests == 1
    assert services.lens.fit_status().state.value == "stable"
