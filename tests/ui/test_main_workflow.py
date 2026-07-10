from PySide6.QtCore import Qt

from jstudio.domain import (
    InterventionDraft,
    InterventionEntry,
    InterventionOperation,
    RunState,
)
from jstudio.project import ProjectDocument
from jstudio.services.fake import create_fake_services
from jstudio.ui.shell.main_window import JStudioMainWindow


def test_main_workspace_matches_scanner_structure(window):
    workspace = window.main_workspace

    assert workspace.found_table.model().columnCount() == 3
    assert workspace.upper_splitter.handleWidth() == 7
    assert workspace.vertical_splitter.handleWidth() == 7
    assert workspace.bottom_strip.height() == 33
    assert workspace.first_read.text() == "First Read"
    assert workspace.next_read.text() == "Next Read"
    assert workspace.undo_read.text() == "Undo Read"
    assert not workspace.output.isVisible()


def test_first_read_pause_next_resume_stop(qtbot):
    services = create_fake_services(token_delay=0.04)
    project = ProjectDocument.new()
    window = JStudioMainWindow(services, project)
    qtbot.addWidget(window)
    window.set_session(services.sessions.list_sessions()[0])
    workspace = window.main_workspace
    workspace.prompt.setPlainText("Inspect this prompt")

    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: workspace.run_state is RunState.RUNNING)
    assert workspace.button_labels() == ("Pause", "Next Token", "Stop")

    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)
    assert workspace.run_state is RunState.PAUSED
    qtbot.mouseClick(workspace.next_read, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)
    qtbot.mouseClick(workspace.undo_read, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: workspace.run_state is RunState.READY)
    assert workspace.button_labels() == ("First Read", "Next Read", "Undo Read")
    services.generation.close()


def test_streamed_frames_update_found_concepts(qtbot, window):
    workspace = window.main_workspace
    workspace.prompt.setPlainText("Inspect this prompt")
    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)

    qtbot.waitUntil(lambda: workspace.activation_model.rowCount() >= 8)

    assert workspace.found_count.text().startswith("Found: 8")
    assert workspace.activation_model.data(workspace.activation_model.index(0, 0))


def test_clicking_intervention_checkbox_updates_project(qtbot, window):
    draft = InterventionDraft(
        InterventionOperation.INJECT,
        None,
        "cat",
        0.5,
        1,
        2,
    )
    window.project.interventions.append(InterventionEntry.from_draft(draft))
    workspace = window.main_workspace
    workspace.intervention_model.replace_rows(window.project.interventions)
    index = workspace.intervention_model.index(0, 0)
    rect = workspace.stack_view.table.visualRect(index)

    qtbot.mouseClick(
        workspace.stack_view.table.viewport(),
        Qt.MouseButton.LeftButton,
        pos=rect.center(),
    )

    assert window.project.interventions[0].enabled


def test_main_chat_button_switches_to_synced_chat_without_pasting_run(qtbot, window):
    workspace = window.main_workspace
    workspace.prompt.setPlainText("Inspect this prompt")
    qtbot.mouseClick(workspace.first_read, Qt.MouseButton.LeftButton)
    qtbot.waitUntil(lambda: bool(workspace.output_text))

    qtbot.mouseClick(workspace.send_to_chat, Qt.MouseButton.LeftButton)

    assert window.tabs.currentWidget() is window.chat_workspace
    assert window.chat_workspace.composer.toPlainText() == ""
    assert "synced" in window.chat_workspace.control_status.text().lower()
    transcript = "\n".join(
        window.chat_workspace.transcript_model.message(row).content
        for row in range(window.chat_workspace.transcript_model.rowCount())
    )
    assert workspace.output_text in transcript


def test_editing_completed_prompt_makes_rerun_action_explicit(window):
    workspace = window.main_workspace
    workspace._completed_prompt = "original prompt"

    workspace.prompt.setPlainText("edited prompt")

    assert workspace.first_read.text() == "Rerun Edited Prompt"
    assert "rerun" in workspace.status.text().lower()
