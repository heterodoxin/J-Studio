from PySide6.QtCore import QSize
from PySide6.QtGui import QPalette

from jstudio.ui.lensview.web_view import JLensWebView
from jstudio.ui.theme import apply_jstudio_theme


def test_modern_theme_installs_semantic_graphite_violet_styles(qapp, window):
    previous = qapp.styleSheet()
    try:
        apply_jstudio_theme(qapp)
        stylesheet = qapp.styleSheet()

        assert "#workspaceTabs" in stylesheet
        assert 'QWidget[role="panel"]' in stylesheet
        assert "#8b5cf6" in stylesheet
        assert window.tabs.objectName() == "workspaceTabs"
        assert window.session_bar.property("role") == "session"
    finally:
        qapp.setStyleSheet(previous)


def test_compact_native_shell_geometry_and_splitters(qtbot, window):
    qtbot.waitExposed(window)
    workspace = window.main_workspace
    upper = workspace.upper_splitter.sizes()
    vertical = workspace.vertical_splitter.sizes()

    assert window.size() == QSize(1101, 888)
    assert 0.44 <= upper[0] / sum(upper) <= 0.54
    assert workspace.upper_splitter.handleWidth() == 8
    assert workspace.vertical_splitter.handleWidth() == 8
    assert 179 <= vertical[1] <= 181
    assert not workspace.bottom_strip.isVisible()
    assert window.palette().color(QPalette.ColorRole.Text) != window.palette().color(
        QPalette.ColorRole.Base
    )


def test_session_identity_uses_compact_status_pills(window):
    assert window.session_bar.backend_badge.property("role") == "statusPill"
    assert window.session_bar.status.property("role") == "statusPill"
    assert window.session_bar.lens_status.property("role") == "statusPill"


def test_main_is_startup_tab_and_has_no_dashboard_navigation(window):
    assert window.tabs.currentWidget() is window.main_workspace
    assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
        "Main",
        "Chat",
        "J-Lens",
        "Rules",
    ]


def test_analysis_expansion_restores_compact_main(qtbot, window):
    window.tabs.setCurrentWidget(window.jlens_workspace)
    qtbot.waitUntil(lambda: window.width() >= 1000)

    window.tabs.setCurrentWidget(window.main_workspace)

    assert window.size() == QSize(1101, 888)


def test_window_manager_cannot_collapse_main_below_compact_size(qapp, window):
    window.resize(5112, 366)
    qapp.processEvents()

    assert window.width() >= 1101
    assert window.height() >= 888


def test_jlens_uses_original_interactive_research_surface(window):
    workspace = window.jlens_workspace

    assert isinstance(workspace.web, JLensWebView)
    assert workspace.sizeHint().width() >= 1200
    assert workspace.sizeHint().height() >= 780
    assert workspace.web.accessibleName() == "Interactive J-Lens slice"
    assert workspace.refresh_button.accessibleName()
    assert workspace.export_button.accessibleName()
