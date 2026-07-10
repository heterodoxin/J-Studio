from PySide6.QtCore import Qt

from jstudio.ui.sessions.picker import SessionPickerDialog


def test_picker_has_three_sources_and_capability_preview(qtbot, services):
    picker = SessionPickerDialog(services.sessions)
    qtbot.addWidget(picker)
    picker.show()

    assert [picker.tabs.tabText(index) for index in range(3)] == [
        "Local Models",
        "Remote Workers",
        "Offline Traces",
    ]
    picker.local_table.selectRow(0)
    qtbot.waitUntil(lambda: "Qwen" in picker.preview.text())
    assert picker.open_button.isEnabled()


def test_picker_enter_opens_selected_session(qtbot, services):
    picker = SessionPickerDialog(services.sessions)
    qtbot.addWidget(picker)
    picker.show()
    picker.local_table.selectRow(0)

    with qtbot.waitSignal(picker.session_opened) as signal:
        qtbot.keyClick(picker.local_table, Qt.Key.Key_Return)

    assert signal.args[0].session_id == "local:qwen-27b"
