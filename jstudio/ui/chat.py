"""Conventional multi-turn chat backed by replaceable generation services."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import QAbstractListModel, QModelIndex, Qt, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QListView,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jstudio.domain import ModelSessionSummary, RunMode
from jstudio.project import ProjectDocument
from jstudio.services.protocols import GenerationRequest, JStudioServices
from jstudio.ui.main_workspace import _GenerationBridge


@dataclass(frozen=True, slots=True)
class ChatMessage:
    role: str
    content: str
    run_id: str | None = None
    event: bool = False


class ChatTranscriptModel(QAbstractListModel):
    MessageRole = Qt.ItemDataRole.UserRole + 1

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._messages: list[ChatMessage] = []

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._messages)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        message = self._messages[index.row()]
        if role == self.MessageRole:
            return message
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return f"{message.role}: {message.content}"
        if role == Qt.ItemDataRole.DisplayRole:
            marker = "▸ " if message.event else ""
            return f"{marker}{message.role}\n{message.content}"
        return None

    def append(self, message: ChatMessage) -> int:
        row = len(self._messages)
        self.beginInsertRows(QModelIndex(), row, row)
        self._messages.append(message)
        self.endInsertRows()
        return row

    def replace(self, row: int, message: ChatMessage) -> None:
        self._messages[row] = message
        self.dataChanged.emit(self.index(row), self.index(row))

    def message(self, row: int) -> ChatMessage:
        return self._messages[row]


class ChatWorkspace(QWidget):
    inspect_requested = Signal(str, int)
    controls_requested = Signal()
    rules_requested = Signal()

    def __init__(
        self,
        services: JStudioServices,
        project: ProjectDocument,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.services = services
        self.project = project
        self.session: ModelSessionSummary | None = None
        self.active_run_id: str | None = None
        self._assistant_row: int | None = None
        self._output = ""
        self.bridge = _GenerationBridge(self)
        self.bridge.started.connect(self._on_started)
        self.bridge.token.connect(self._on_token)
        self.bridge.frame.connect(lambda _frame: None)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.error.connect(self._on_error)
        self.bridge.intervention.connect(self._on_intervention)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        self.transcript_model = ChatTranscriptModel(self)
        self.transcript = QListView(self)
        self.transcript.setModel(self.transcript_model)
        self.transcript.setWordWrap(True)
        self.transcript.setSpacing(6)
        self.transcript.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.transcript.customContextMenuRequested.connect(self._context_menu)
        layout.addWidget(self.transcript, 1)
        controls_row = QHBoxLayout()
        self.control_status = QLabel("Baseline", self)
        self.control_status.setToolTip("Click to review active interventions and rules")
        self.control_status.setCursor(Qt.CursorShape.PointingHandCursor)
        self.interventions_button = QPushButton("Interventions", self)
        self.rules_button = QPushButton("Rules", self)
        self.interventions_button.clicked.connect(self.controls_requested)
        self.rules_button.clicked.connect(self.rules_requested)
        controls_row.addWidget(self.control_status, 1)
        controls_row.addWidget(self.interventions_button)
        controls_row.addWidget(self.rules_button)
        layout.addLayout(controls_row)
        compose = QHBoxLayout()
        self.composer = QPlainTextEdit(self)
        self.composer.setPlaceholderText("Send a prompt to the model…")
        self.composer.setMaximumHeight(90)
        buttons = QVBoxLayout()
        self.send_button = QPushButton("Send", self)
        self.stop_button = QPushButton("Stop", self)
        self.stop_button.hide()
        self.send_button.clicked.connect(self.send)
        self.stop_button.clicked.connect(self.stop)
        buttons.addWidget(self.send_button)
        buttons.addWidget(self.stop_button)
        buttons.addStretch(1)
        compose.addWidget(self.composer, 1)
        compose.addLayout(buttons)
        layout.addLayout(compose)

    def set_session(self, session: ModelSessionSummary | None) -> None:
        self.session = session
        enabled = bool(session and session.capabilities.generate)
        self.send_button.setEnabled(enabled)
        self.composer.setEnabled(enabled)
        if session and not enabled:
            self.composer.setPlaceholderText("Offline traces cannot generate")

    def refresh_control_status(self) -> None:
        interventions = sum(entry.enabled for entry in self.project.interventions)
        rules = sum(rule.enabled for rule in self.project.rules)
        if not interventions and not rules:
            text = "Baseline"
        else:
            pieces = []
            if rules:
                pieces.append(f"{rules} rule{'s' if rules != 1 else ''}")
            if interventions:
                pieces.append(
                    f"{interventions} intervention{'s' if interventions != 1 else ''}"
                )
            text = " + ".join(pieces) + " active"
        self.control_status.setText(text)

    def sync_from_main(self, handoff: str = "") -> None:
        self.composer.clear()
        self.refresh_control_status()
        self.control_status.setText(f"Synced · {self.control_status.text()}")
        if handoff.strip():
            self.transcript_model.append(
                ChatMessage("Synced Main", handoff.strip(), event=True)
            )

    def send(self) -> None:
        if self.session is None or self.active_run_id is not None:
            return
        text = self.composer.toPlainText().strip()
        if not text:
            self.composer.setFocus()
            return
        self.transcript_model.append(ChatMessage("You", text))
        self._assistant_row = self.transcript_model.append(
            ChatMessage(self.session.display_name or self.session.model_id, "Generating…")
        )
        self.composer.clear()
        self._output = ""
        enabled_entries = tuple(
            entry for entry in self.project.interventions if entry.enabled
        )
        intervention_ids = tuple(entry.intervention_id for entry in enabled_entries)
        rule_ids = tuple(rule.rule_id for rule in self.project.rules if rule.enabled)
        rule_records = tuple(rule for rule in self.project.rules if rule.enabled)
        mode = RunMode.WITH_STACK if intervention_ids or rule_ids else RunMode.BASELINE
        self.send_button.hide()
        self.stop_button.show()
        self.active_run_id = self.services.generation.start(
            GenerationRequest(
                self.session.session_id,
                text,
                mode,
                intervention_ids,
                tuple(entry.draft for entry in enabled_entries),
                rule_ids,
                rule_records,
            ),
            self.bridge,
        )

    def stop(self) -> None:
        if self.active_run_id:
            try:
                self.services.generation.stop(self.active_run_id)
            except KeyError:
                pass

    def _on_started(self, run) -> None:
        self.active_run_id = run.run_id

    def _on_token(self, run_id: str, token: str, output_text: str) -> None:
        self._output = output_text
        if self._assistant_row is not None:
            current = self.transcript_model.message(self._assistant_row)
            self.transcript_model.replace(
                self._assistant_row,
                ChatMessage(current.role, output_text, run_id=run_id),
            )
            self.transcript.scrollToBottom()

    def _on_finished(self, run) -> None:
        if self._assistant_row is not None:
            current = self.transcript_model.message(self._assistant_row)
            self.transcript_model.replace(
                self._assistant_row,
                ChatMessage(current.role, run.output_text, run_id=run.run_id),
            )
        self.project.runs.append(run)
        self.active_run_id = None
        self.send_button.show()
        self.stop_button.hide()

    def _on_error(self, run_id: str, message: str, detail: str) -> None:
        self.transcript_model.append(ChatMessage("Error", message, event=True))
        self.active_run_id = None
        self.send_button.show()
        self.stop_button.hide()

    def _on_intervention(self, intervention_id: str, state: str, detail: str) -> None:
        self.refresh_control_status()
        self.control_status.setText(
            f"{intervention_id}: {state} — {detail} · {self.control_status.text()}"
        )

    def inspect_message(self, row: int) -> None:
        message = self.transcript_model.message(row)
        if message.run_id:
            frames = self.services.lens.frames(message.run_id)
            position = frames[-1].token_index if frames else 0
            self.inspect_requested.emit(message.run_id, position)

    def _context_menu(self, position) -> None:
        index = self.transcript.indexAt(position)
        if not index.isValid():
            return
        message = self.transcript_model.message(index.row())
        menu = QMenu(self)
        menu.addAction("Copy")
        if message.role == "You":
            menu.addAction("Edit and Resend")
        elif message.run_id:
            menu.addAction("Regenerate")
            menu.addAction("Continue")
            menu.addAction("Compare")
            menu.addAction("Add Output to Prompt")
            menu.addAction("Inspect with J-Lens", lambda: self.inspect_message(index.row()))
        menu.popup(self.transcript.viewport().mapToGlobal(position))
