from PySide6.QtWidgets import QPushButton

from jstudio.ui.accessibility import apply_text_scale


def test_icon_only_controls_expose_names_and_tooltips(window):
    controls = (
        window.session_bar.select_button,
        window.session_bar.overflow,
        window.main_workspace.prompt_expand,
    )
    for control in controls:
        assert control.accessibleName()
        assert control.toolTip()


def test_key_workflow_controls_remain_visible_at_160_percent(qapp, qtbot, window):
    original_font = qapp.font()
    try:
        apply_text_scale(qapp, 160)
        window.resize(1200, 900)
        qtbot.wait(0)
        controls = (
            window.main_workspace.first_read,
            window.main_workspace.next_read,
            window.main_workspace.undo_read,
            window.main_workspace.add_manual,
        )
        for control in controls:
            assert control.isVisible()
            assert control.width() >= control.sizeHint().width()
            assert control.height() >= control.sizeHint().height()
        rules_index = window.tabs.indexOf(window.rules_workspace)
        assert window.tabs.tabBar().isVisible()
        assert window.tabs.tabBar().tabRect(rules_index).width() > 0
    finally:
        qapp.setFont(original_font)


def test_primary_buttons_are_keyboard_focusable(window):
    for button in window.main_workspace.findChildren(QPushButton):
        assert button.focusPolicy().value != 0
