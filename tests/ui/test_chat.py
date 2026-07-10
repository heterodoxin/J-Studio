from dataclasses import replace

from PySide6.QtCore import Qt

from jstudio.domain import (
    InterventionDraft,
    InterventionEntry,
    InterventionOperation,
    RuleRecord,
    RuleTrigger,
)


def test_chat_send_stop_and_inspect(qtbot, window):
    window.tabs.setCurrentWidget(window.chat_workspace)
    window.chat_workspace.composer.setPlainText("Explain the diagram")
    qtbot.mouseClick(window.chat_workspace.send_button, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: window.chat_workspace.transcript_model.rowCount() >= 2)
    qtbot.waitUntil(lambda: window.chat_workspace.active_run_id is None)
    assistant_row = window.chat_workspace.transcript_model.rowCount() - 1
    window.chat_workspace.inspect_message(assistant_row)

    assert window.tabs.currentWidget() is window.jlens_workspace
    assert window.jlens_workspace.selection.run_id is not None


def test_chat_status_reports_active_controls(qtbot, window):
    assert window.chat_workspace.control_status.text() == "Baseline"
    editor = window.main_workspace.open_intervention_editor("inject", "caution")
    editor.target_term.setText("caution")
    qtbot.mouseClick(editor.add_button, Qt.MouseButton.LeftButton)
    window.main_workspace._arm_stack()

    window.chat_workspace.refresh_control_status()

    assert "1 intervention" in window.chat_workspace.control_status.text()


def test_chat_send_syncs_enabled_intervention_drafts(qtbot, window):
    draft = InterventionDraft(
        InterventionOperation.INJECT,
        None,
        "cat",
        2.0,
        0,
        4,
    )
    entry = replace(InterventionEntry.from_draft(draft), enabled=True)
    window.project.interventions.append(entry)
    window.tabs.setCurrentWidget(window.chat_workspace)
    window.chat_workspace.composer.setPlainText("continue")

    qtbot.mouseClick(window.chat_workspace.send_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.chat_workspace.active_run_id is None)

    assert window.services.generation.last_request.intervention_ids == (
        entry.intervention_id,
    )
    assert window.services.generation.last_request.intervention_drafts == (draft,)


def test_chat_send_syncs_enabled_rules(qtbot, window):
    rule = RuleRecord(
        "rule-1",
        "Guard",
        "function run(ctx) { return []; }",
        RuleTrigger.BEFORE_TOKEN,
        enabled=True,
        trusted=True,
        config={"term": "cat"},
    )
    window.project.rules.append(rule)
    window.tabs.setCurrentWidget(window.chat_workspace)
    window.chat_workspace.composer.setPlainText("continue")

    qtbot.mouseClick(window.chat_workspace.send_button, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: window.chat_workspace.active_run_id is None)

    assert window.services.generation.last_request.rule_ids == ("rule-1",)
    assert window.services.generation.last_request.rule_records == (rule,)


def test_chat_intervention_event_updates_status_not_transcript(window):
    before = window.chat_workspace.transcript_model.rowCount()

    window.chat_workspace._on_intervention("inject-cat", "applied", "scale 0.25")

    assert window.chat_workspace.transcript_model.rowCount() == before
    assert "inject-cat: applied" in window.chat_workspace.control_status.text()
