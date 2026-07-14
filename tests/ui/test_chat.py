from dataclasses import replace

from PySide6.QtCore import Qt

from jstudio.domain import (
    InterventionDraft,
    InterventionEntry,
    InterventionOperation,
    RuleRecord,
    RuleTrigger,
)
from jstudio.ui.chat import ChatMessage


def test_chat_workspace_has_modern_semantic_surfaces(window):
    chat = window.chat_workspace

    assert chat.transcript.objectName() == "chatTranscript"
    assert chat.composer.objectName() == "chatComposer"
    assert chat.composer.property("role") == "data"
    assert chat.send_button.property("role") == "primary"
    assert chat.control_status.property("role") == "statusPill"


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


def test_chat_error_replaces_generating_row_with_actionable_detail(window):
    chat = window.chat_workspace
    chat._assistant_row = chat.transcript_model.append(
        ChatMessage("Model", "Generating…")
    )

    chat._on_error("run-1", "Local Qwen generation failed", "TypeError('bad cache')")

    message = chat.transcript_model.message(chat._assistant_row)
    assert message.role == "Error"
    assert "TypeError('bad cache')" in message.content


def test_chat_edit_and_continue_actions_create_real_conversation_branches(window):
    chat = window.chat_workspace
    chat.transcript_model.append(ChatMessage("You", "first question"))
    chat.transcript_model.append(ChatMessage("Model", "first answer", run_id="r1"))
    chat.transcript_model.append(ChatMessage("You", "later question"))

    chat.edit_and_resend(0)

    assert chat.composer.toPlainText() == "first question"
    assert chat.transcript_model.rowCount() == 0

    chat.transcript_model.append(ChatMessage("You", "new question"))
    chat.transcript_model.append(ChatMessage("Model", "new answer", run_id="r2"))
    chat.continue_response(1)
    assert chat.composer.toPlainText() == "Continue from your previous response."

    chat.add_output_to_prompt(1)
    assert "new answer" in chat.composer.toPlainText()
