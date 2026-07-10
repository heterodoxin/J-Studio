from PySide6.QtCore import Qt

from jstudio.domain import InterventionOperation


def test_replace_editor_validates_and_adds_draft(qtbot, window):
    editor = window.main_workspace.open_intervention_editor("replace", "injection")
    editor.target_term.setText("trusted")
    editor.strength.setValue(0.8)
    editor.layer_start.setValue(18)
    editor.layer_end.setValue(26)

    qtbot.mouseClick(editor.add_button, Qt.MouseButton.LeftButton)

    entry = window.project.interventions[-1]
    assert entry.draft.operation is InterventionOperation.REPLACE
    assert entry.draft.source_term == "injection"
    assert entry.draft.target_term == "trusted"
    assert window.main_workspace.intervention_model.rowCount() == 1


def test_intervention_editor_defaults_to_auto_max_budget(window):
    editor = window.main_workspace.open_intervention_editor("inject", "cat")
    maximum = window.current_session.capabilities.strength_max

    assert editor.strength.value() == maximum
    assert "maximum search budget" in editor.strength.toolTip().lower()
    assert "auto-search minimum effective scale" in editor.preview.text().lower()


def test_inject_requires_target_and_focuses_error(qtbot, window):
    editor = window.main_workspace.open_intervention_editor("inject", "")
    editor.target_term.clear()

    qtbot.mouseClick(editor.add_button, Qt.MouseButton.LeftButton)

    assert "target" in editor.error_label.text().lower()
    assert editor.focusWidget() is editor.target_term


def test_offline_trace_disables_arm_but_allows_draft(qtbot, window, services):
    window.set_session(services.sessions.list_sessions()[1])

    assert window.main_workspace.add_manual.isEnabled()
    assert not window.main_workspace.arm_button.isEnabled()
    assert "offline" in window.main_workspace.arm_button.toolTip().lower()
