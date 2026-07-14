from PySide6.QtCore import Qt

from jstudio.domain import InterventionDraft, InterventionEntry, InterventionOperation


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


def test_context_edit_prefills_and_replaces_entry_by_stable_id(qtbot, window):
    draft = InterventionDraft(
        InterventionOperation.REPLACE,
        "apple",
        "banana",
        0.75,
        4,
        9,
        duration="steps",
        step_count=3,
        match_mode="case-insensitive",
        trigger="before-token",
    )
    entry = InterventionEntry.from_draft(draft)
    window.project.interventions.append(entry)
    workspace = window.main_workspace
    workspace.intervention_model.replace_rows(window.project.interventions)

    workspace.stack_view.build_context_menu((0,)).actions()[1].trigger()
    editor = workspace._editors[-1]

    assert editor.source_term.text() == "apple"
    assert editor.target_term.text() == "banana"
    assert editor.strength.value() == 0.75
    assert editor.duration.currentText() == "N Steps"
    assert editor.step_count.value() == 3

    editor.target_term.setText("pear")
    qtbot.mouseClick(editor.add_button, Qt.MouseButton.LeftButton)

    assert window.project.interventions[0].intervention_id == entry.intervention_id
    assert window.project.interventions[0].draft.target_term == "pear"


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
