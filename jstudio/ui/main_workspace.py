"""Main read workspace with a Cheat Engine-inspired layout."""

from __future__ import annotations

from PySide6.QtCore import QEvent, QObject, Qt, Signal
from PySide6.QtGui import QAction, QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QStyledItemDelegate,
    QStyleOptionViewItem,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from jstudio.domain import (
    InterventionEntry,
    InterventionState,
    ModelSessionSummary,
    RunMode,
    RunState,
)
from jstudio.project import ProjectDocument
from jstudio.services.protocols import (
    GenerationRequest,
    JStudioServices,
    ReadConfiguration,
)
from jstudio.ui.interventions.editor import InterventionEditor
from jstudio.ui.interventions.stack import InterventionStackView
from jstudio.ui.models import ActivationTableModel, InterventionTableModel


class _GenerationBridge(QObject):
    started = Signal(object)
    token = Signal(str, str, str)
    frame = Signal(object)
    intervention = Signal(str, str, str)
    finished = Signal(object)
    error = Signal(str, str, str)

    def on_started(self, run):
        self.started.emit(run)

    def on_token(self, run_id, token, output_text):
        self.token.emit(run_id, token, output_text)

    def on_frame(self, frame):
        self.frame.emit(frame)

    def on_intervention(self, intervention_id, state, detail):
        self.intervention.emit(intervention_id, state, detail)

    def on_finished(self, run):
        self.finished.emit(run)

    def on_error(self, run_id, message, detail=""):
        self.error.emit(run_id, message, detail)


class SignedScoreDelegate(QStyledItemDelegate):
    def paint(self, painter: QPainter, option: QStyleOptionViewItem, index) -> None:
        if index.column() != 1:
            super().paint(painter, option, index)
            return
        activation = index.data(Qt.ItemDataRole.UserRole)
        if activation is None:
            super().paint(painter, option, index)
            return
        painter.save()
        painter.fillRect(option.rect, option.palette.base())
        center = option.rect.center().x()
        painter.setPen(option.palette.mid().color())
        painter.drawLine(center, option.rect.top() + 2, center, option.rect.bottom() - 2)
        width = int(option.rect.width() * min(abs(activation.score), 1.0) / 2)
        color = QColor("#2563eb") if activation.score >= 0 else QColor("#b91c1c")
        rect = option.rect.adjusted(2, 4, -2, -4)
        if activation.score >= 0:
            rect.setLeft(center)
            rect.setRight(center + width)
        else:
            rect.setLeft(center - width)
            rect.setRight(center)
        painter.fillRect(rect, color)
        painter.setPen(option.palette.text().color())
        painter.drawText(
            option.rect.adjusted(4, 0, -4, 0),
            Qt.AlignmentFlag.AlignRight,
            f"{activation.score:+.2f}",
        )
        painter.restore()


class MainReadWorkspace(QWidget):
    session_requested = Signal()
    model_view_requested = Signal()
    jlens_requested = Signal(str, int)
    rules_requested = Signal()
    run_state_changed = Signal(object)
    chat_requested = Signal(str)

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
        self.run_state = RunState.READY
        self.current_run_id: str | None = None
        self.last_run_id: str | None = None
        self.output_text = ""
        self._completed_prompt = ""
        self._run_activations: dict[str, object] = {}
        self._editors: list[InterventionEditor] = []
        self.bridge = _GenerationBridge(self)
        self.bridge.started.connect(self._on_started)
        self.bridge.token.connect(self._on_token)
        self.bridge.frame.connect(self._on_frame)
        self.bridge.intervention.connect(self._on_intervention)
        self.bridge.finished.connect(self._on_finished)
        self.bridge.error.connect(self._on_error)

        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 8)
        root.setSpacing(8)
        self.vertical_splitter = QSplitter(Qt.Orientation.Vertical, self)
        self.vertical_splitter.setHandleWidth(8)
        self.upper_splitter = QSplitter(Qt.Orientation.Horizontal, self.vertical_splitter)
        self.upper_splitter.setHandleWidth(8)
        self.found_panel = self._build_found_panel()
        self.controls_panel = self._build_controls_panel()
        self.upper_splitter.addWidget(self.found_panel)
        self.upper_splitter.addWidget(self.controls_panel)
        self.upper_splitter.setSizes([330, 386])
        self.intervention_model = InterventionTableModel(self.project.interventions, self)
        self.intervention_model.enabled_changed.connect(
            self._intervention_enabled_changed
        )
        self.stack_view = InterventionStackView(
            self.intervention_model, self.vertical_splitter
        )
        self.arm_button = self.stack_view.arm
        self.stack_view.inject.clicked.connect(
            lambda: self.open_intervention_editor("inject", "")
        )
        self.stack_view.replace.clicked.connect(
            lambda: self.open_intervention_editor("replace", self._selected_term())
        )
        self.stack_view.suppress.clicked.connect(
            lambda: self.open_intervention_editor("suppress", self._selected_term())
        )
        self.stack_view.clear.clicked.connect(self._clear_stack)
        self.stack_view.preview.clicked.connect(self.preview_selected_intervention)
        self.stack_view.bake.clicked.connect(self._bake_stack)
        self.stack_view.group.clicked.connect(
            lambda: self.status.setText(
                "Intervention execution follows the current table order"
            )
        )
        self.arm_button.clicked.connect(self._arm_stack)
        self.vertical_splitter.addWidget(self.upper_splitter)
        self.vertical_splitter.addWidget(self.stack_view)
        for pane in (self.upper_splitter, self.stack_view):
            pane.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Ignored)
        self.vertical_splitter.setSizes([328, 108])
        root.addWidget(self.vertical_splitter, 1)
        self.bottom_strip = QWidget(self)
        self.bottom_strip.setMinimumHeight(32)
        bottom = QHBoxLayout(self.bottom_strip)
        bottom.setContentsMargins(0, 2, 0, 2)
        self.advanced_button = QPushButton("Advanced Options", self.bottom_strip)
        self.rules_button = QPushButton("Rules", self.bottom_strip)
        bottom.addWidget(self.advanced_button)
        bottom.addStretch(1)
        bottom.addWidget(self.rules_button)
        self.rules_button.clicked.connect(self.rules_requested)
        root.addWidget(self.bottom_strip)
        self.bottom_strip.hide()
        self._sync_button_minimums()
        self._sync_session_state()

    def changeEvent(self, event) -> None:
        super().changeEvent(event)
        if event.type() in {QEvent.Type.FontChange, QEvent.Type.StyleChange}:
            self._sync_button_minimums()

    def _sync_button_minimums(self) -> None:
        for button in self.findChildren(QPushButton):
            button.setMinimumHeight(button.sizeHint().height())
        if hasattr(self, "bottom_strip"):
            strip_height = max(32, self.bottom_strip.sizeHint().height())
            self.bottom_strip.setMinimumHeight(strip_height)

    def _build_found_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setProperty("role", "panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        header = QHBoxLayout()
        title = QLabel("Found Concepts")
        title.setProperty("role", "heading")
        header.addWidget(title)
        self.found_count = QLabel("Found: 0 Concepts")
        self.found_count.setProperty("role", "muted")
        header.addStretch(1)
        header.addWidget(self.found_count)
        layout.addLayout(header)
        self.activation_model = ActivationTableModel(self)
        self.found_table = QTableView(panel)
        self.found_table.setObjectName("foundConcepts")
        self.found_table.setModel(self.activation_model)
        self.found_table.setSelectionBehavior(
            QAbstractItemView.SelectionBehavior.SelectRows
        )
        self.found_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.found_table.setAlternatingRowColors(True)
        self.found_table.setItemDelegateForColumn(1, SignedScoreDelegate(self.found_table))
        self.found_table.doubleClicked.connect(self._double_click_activation)
        self.found_table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.found_table.customContextMenuRequested.connect(self._show_context_menu)
        self.found_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.found_table, 1)
        actions = QHBoxLayout()
        self.add_found = QPushButton("Add →")
        self.add_found.setProperty("role", "primary")
        self.add_found.setToolTip("Add selected concept to the intervention list")
        self.clear_found = QPushButton("Clear")
        self.clear_found.setProperty("role", "ghost")
        self.model_view = QPushButton("Model View")
        self.open_jlens = QPushButton("J-Lens")
        self.add_found.clicked.connect(self._add_selected)
        self.clear_found.clicked.connect(lambda: self._replace_activations(()))
        self.model_view.clicked.connect(self.model_view_requested)
        self.open_jlens.clicked.connect(self._request_jlens)
        actions.addWidget(self.add_found)
        actions.addWidget(self.clear_found)
        actions.addStretch(1)
        actions.addWidget(self.open_jlens)
        actions.addWidget(self.model_view)
        layout.addLayout(actions)
        return panel

    def _build_controls_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setProperty("role", "panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        heading = QHBoxLayout()
        title = QLabel("Read J-Space", panel)
        title.setProperty("role", "heading")
        self.status = QLabel("Ready", panel)
        self.status.setProperty("role", "muted")
        heading.addWidget(title)
        heading.addStretch(1)
        heading.addWidget(self.status)
        layout.addLayout(heading)
        buttons = QHBoxLayout()
        self.first_read = QPushButton("First Read")
        self.first_read.setProperty("role", "primary")
        self.next_read = QPushButton("Next Read")
        self.undo_read = QPushButton("Undo Read")
        for button in (self.first_read, self.next_read, self.undo_read):
            buttons.addWidget(button)
        self.first_read.clicked.connect(self._first_action)
        self.next_read.clicked.connect(self._next_action)
        self.undo_read.clicked.connect(self._undo_action)
        layout.addLayout(buttons)
        form = QFormLayout()
        prompt_row = QWidget(panel)
        prompt_layout = QHBoxLayout(prompt_row)
        prompt_layout.setContentsMargins(0, 0, 0, 0)
        self.prompt = QPlainTextEdit(prompt_row)
        self.prompt.setObjectName("promptEditor")
        self.prompt.setMinimumHeight(86)
        self.prompt.setMaximumHeight(118)
        self.prompt.setPlaceholderText("Enter a prompt to inspect")
        self.prompt.textChanged.connect(self._prompt_edited)
        self.prompt_expand = QPushButton("Expand", prompt_row)
        self.prompt_expand.setProperty("role", "ghost")
        self.prompt_expand.setToolTip("Open larger prompt editor")
        self.prompt_expand.setAccessibleName("Open larger prompt editor")
        self.prompt_expand.clicked.connect(self._expand_prompt)
        prompt_layout.addWidget(self.prompt, 1)
        prompt_layout.addWidget(
            self.prompt_expand, 0, Qt.AlignmentFlag.AlignTop
        )
        self.read_type = QComboBox(panel)
        self.read_type.addItems(
            [
                "Exact concepts",
                "Changed concepts",
                "Increased score",
                "Decreased score",
                "Unknown initial state",
            ]
        )
        self.concept_type = QComboBox(panel)
        self.concept_type.addItems(["Tokens", "Words", "Phrases", "All verbalizable"])
        form.addRow("Prompt", prompt_row)
        self.output = QPlainTextEdit(panel)
        self.output.setReadOnly(True)
        self.output.setMaximumHeight(96)
        self.output.setPlaceholderText("Generated output appears here")
        self.output.hide()
        output_row = QWidget(panel)
        output_layout = QVBoxLayout(output_row)
        output_layout.setContentsMargins(0, 0, 0, 0)
        self.send_to_chat = QPushButton("Continue / Edit in Chat", output_row)
        self.send_to_chat.setEnabled(False)
        self.send_to_chat.clicked.connect(
            lambda: self.chat_requested.emit(self._chat_handoff_text())
        )
        output_layout.addWidget(self.send_to_chat)
        form.addRow("Chat", output_row)
        form.addRow("Read Type", self.read_type)
        form.addRow("Concept Type", self.concept_type)
        layout.addLayout(form)
        self.advanced_scan = QPushButton("▸  Advanced scan", panel)
        self.advanced_scan.setCheckable(True)
        self.advanced_scan.setProperty("role", "ghost")
        self.advanced_scan.setAccessibleName("Show advanced J-Space scan options")
        layout.addWidget(self.advanced_scan)
        self.scan_options = QGroupBox("Scan range and filters", panel)
        option_layout = QFormLayout(self.scan_options)
        layer_row = QWidget(self.scan_options)
        layer_layout = QHBoxLayout(layer_row)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_start = QSpinBox(layer_row)
        self.layer_stop = QSpinBox(layer_row)
        self.layer_step = QSpinBox(layer_row)
        self.layer_step.setRange(1, 1024)
        self.layer_step.setValue(4)
        layer_layout.addWidget(QLabel("Start"))
        layer_layout.addWidget(self.layer_start)
        layer_layout.addWidget(QLabel("Stop"))
        layer_layout.addWidget(self.layer_stop)
        layer_layout.addWidget(QLabel("Step"))
        layer_layout.addWidget(self.layer_step)
        flags = QWidget(self.scan_options)
        flags_layout = QHBoxLayout(flags)
        flags_layout.setContentsMargins(0, 0, 0, 0)
        self.positive = QCheckBox("Positive")
        self.negative = QCheckBox("Negative")
        self.injected = QCheckBox("Injected")
        self.fast_read = QCheckBox("Fast Read")
        for checkbox in (self.positive, self.negative, self.injected, self.fast_read):
            checkbox.setChecked(True)
            flags_layout.addWidget(checkbox)
        self.pause_while_reading = QCheckBox("Pause generation while reading")
        self.live_update = QCheckBox("Live update found concepts")
        limits = QWidget(self.scan_options)
        limits_layout = QHBoxLayout(limits)
        limits_layout.setContentsMargins(0, 0, 0, 0)
        self.max_concepts = QSpinBox(limits)
        self.max_concepts.setRange(8, 2000)
        self.max_concepts.setValue(200)
        self.max_tokens = QSpinBox(limits)
        self.max_tokens.setRange(1, 32768)
        self.max_tokens.setValue(2048)
        limits_layout.addWidget(QLabel("Concepts"))
        limits_layout.addWidget(self.max_concepts)
        limits_layout.addWidget(QLabel("Max Tokens"))
        limits_layout.addWidget(self.max_tokens)
        option_layout.addRow(layer_row)
        option_layout.addRow(flags)
        option_layout.addRow(limits)
        option_layout.addRow(self.pause_while_reading)
        option_layout.addRow(self.live_update)
        layout.addWidget(self.scan_options)
        self.scan_options.hide()
        self.advanced_scan.toggled.connect(self._toggle_scan_options)
        self.add_manual = QPushButton("Add Intervention Manually")
        self.add_manual.setProperty("role", "ghost")
        self.add_manual.clicked.connect(lambda: self.open_intervention_editor("inject", ""))
        layout.addWidget(self.add_manual)
        layout.addStretch(1)
        return panel

    def _toggle_scan_options(self, expanded: bool) -> None:
        self.scan_options.setVisible(expanded)
        self.advanced_scan.setText(
            "▾  Advanced scan" if expanded else "▸  Advanced scan"
        )

    def _expand_prompt(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Edit Prompt")
        dialog.resize(720, 420)
        layout = QVBoxLayout(dialog)
        editor = QPlainTextEdit(dialog)
        editor.setPlainText(self.prompt.toPlainText())
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Cancel | QDialogButtonBox.StandardButton.Ok,
            dialog,
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(editor, 1)
        layout.addWidget(buttons)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.prompt.setPlainText(editor.toPlainText())

    def set_session(self, session: ModelSessionSummary | None) -> None:
        self.session = session
        if session is not None:
            self.layer_start.setRange(0, session.layer_count - 1)
            self.layer_stop.setRange(0, session.layer_count - 1)
            self.layer_stop.setValue(session.layer_count - 1)
        self._sync_session_state()

    def _sync_session_state(self) -> None:
        can_generate = bool(self.session and self.session.capabilities.generate)
        can_intervene = bool(self.session and self.session.capabilities.intervene)
        self.first_read.setEnabled(can_generate)
        self.next_read.setEnabled(can_generate)
        self.undo_read.setEnabled(can_generate)
        self.model_view.setEnabled(bool(self.session and self.session.capabilities.inspect))
        self.open_jlens.setEnabled(bool(self.session and self.session.capabilities.inspect))
        self.add_manual.setEnabled(self.session is not None)
        self.arm_button.setEnabled(can_intervene and bool(self.project.interventions))
        self.arm_button.setToolTip(
            "" if can_intervene else "Offline traces cannot apply interventions"
        )
        self.stack_view.bake.setEnabled(can_intervene)

    def _bake_stack(self) -> None:
        if self.session is None:
            self.status.setText("Select a model session before baking")
            return
        drafts = tuple(
            entry.draft for entry in self.project.interventions if entry.enabled
        )
        if not drafts:
            self.status.setText("Enable suppress/replace interventions before baking")
            return
        path, _ = QFileDialog.getSaveFileName(
            self,
            "Bake J-Space Projection",
            "jspace-projection.safetensors",
            "Safetensors (*.safetensors)",
        )
        if not path:
            return
        try:
            weights, manifest = self.services.interventions.bake(
                self.session.session_id,
                drafts,
                path,
            )
        except Exception as exc:
            self.status.setText(f"Bake failed · {exc}")
            return
        self.status.setText(f"Baked {weights.name} + {manifest.name}")

    def button_labels(self) -> tuple[str, str, str]:
        return self.first_read.text(), self.next_read.text(), self.undo_read.text()

    def _set_run_state(self, state: RunState) -> None:
        self.run_state = state
        running = state in (RunState.RUNNING, RunState.PAUSED)
        self.first_read.setText(
            "Resume" if state is RunState.PAUSED else ("Pause" if running else "First Read")
        )
        self.next_read.setText("Next Token" if running else "Next Read")
        self.undo_read.setText("Stop" if running else "Undo Read")
        self.status.setText(
            "Generating" if state is RunState.RUNNING else state.value.title()
        )
        self.run_state_changed.emit(state)

    def _first_action(self) -> None:
        if self.run_state is RunState.READY:
            if self.session is None:
                self.session_requested.emit()
                return
            text = self.prompt.toPlainText().strip()
            if not text:
                self.prompt.setFocus()
                self.status.setText("Enter a prompt")
                return
            enabled_entries = tuple(
                entry for entry in self.project.interventions if entry.enabled
            )
            enabled_rules = tuple(rule for rule in self.project.rules if rule.enabled)
            enabled = tuple(entry.intervention_id for entry in enabled_entries)
            mode = (
                RunMode.WITH_STACK
                if enabled_entries or enabled_rules
                else RunMode.BASELINE
            )
            self._set_run_state(RunState.RUNNING)
            self.current_run_id = self.services.generation.start(
                GenerationRequest(
                    session_id=self.session.session_id,
                    prompt=text,
                    mode=mode,
                    intervention_ids=enabled,
                    intervention_drafts=tuple(
                        entry.draft for entry in enabled_entries
                    ),
                    rule_ids=tuple(
                        rule.rule_id for rule in enabled_rules
                    ),
                    rule_records=enabled_rules,
                    read=ReadConfiguration(
                        layers=tuple(
                            range(
                                self.layer_start.value(),
                                self.layer_stop.value() + 1,
                                self.layer_step.value(),
                            )
                        ),
                        max_concepts=self.max_concepts.value(),
                        max_new_tokens=self.max_tokens.value(),
                    ),
                ),
                self.bridge,
            )
            return
        if self.current_run_id is None:
            return
        if self.run_state is RunState.RUNNING:
            self.services.generation.pause(self.current_run_id)
            self._set_run_state(RunState.PAUSED)
        elif self.run_state is RunState.PAUSED:
            self.services.generation.resume(self.current_run_id)
            self._set_run_state(RunState.RUNNING)

    def _next_action(self) -> None:
        if self.current_run_id and self.run_state in (RunState.RUNNING, RunState.PAUSED):
            self.services.generation.next_token(self.current_run_id)
            self._set_run_state(RunState.PAUSED)
        elif self.run_state is RunState.READY:
            self.status.setText("Next Read refines the current result set")

    def _undo_action(self) -> None:
        if self.current_run_id and self.run_state in (RunState.RUNNING, RunState.PAUSED):
            try:
                self.services.generation.stop(self.current_run_id)
            except KeyError:
                self._set_run_state(RunState.READY)
        else:
            self.status.setText("Restored previous read")

    def _on_started(self, run) -> None:
        self.current_run_id = run.run_id
        self.last_run_id = run.run_id
        self._run_activations.clear()
        self._set_run_state(RunState.RUNNING)

    def _on_token(self, run_id: str, token: str, output_text: str) -> None:
        self.output_text = output_text
        self.output.setPlainText(output_text)
        self.send_to_chat.setEnabled(bool(output_text))
        self.status.setText(f"Generating · {len(output_text.split())} tokens")

    def _on_frame(self, frame) -> None:
        self.last_run_id = frame.run_id
        for activation in frame.activations:
            previous = self._run_activations.get(activation.term)
            if previous is None or activation.score > previous.score:
                self._run_activations[activation.term] = activation
        activations = sorted(
            self._run_activations.values(), key=lambda item: item.score, reverse=True
        )
        self._replace_activations(activations)

    def _on_intervention(self, intervention_id: str, state: str, detail: str) -> None:
        updated = []
        for entry in self.project.interventions:
            if entry.intervention_id == intervention_id:
                from dataclasses import replace

                entry = replace(entry, state=InterventionState(state), status_detail=detail)
            updated.append(entry)
        self.project.interventions[:] = updated
        self.intervention_model.replace_rows(updated)
        # A dropped intervention must not look like a normal run: surface it.
        if state == InterventionState.FAILED.value:
            self.status.setText(f"Intervention not applied · {detail}")

    def _on_finished(self, run) -> None:
        self.project.runs.append(run)
        self.output_text = run.output_text
        self._completed_prompt = run.prompt
        self.output.setPlainText(run.output_text)
        self.send_to_chat.setEnabled(bool(run.output_text))
        self.current_run_id = None
        self._set_run_state(RunState.READY)
        if run.state is RunState.COMPLETE:
            speed = (
                f" · {run.decode_tokens_per_second:.1f} tok/s"
                if run.decode_tokens_per_second is not None
                else ""
            )
            self.status.setText(f"Read complete{speed}")
        else:
            self.status.setText("Partial read retained")

    def _prompt_edited(self) -> None:
        if self.run_state is not RunState.READY or not self._completed_prompt:
            return
        changed = self.prompt.toPlainText().strip() != self._completed_prompt.strip()
        self.first_read.setText("Rerun Edited Prompt" if changed else "First Read")
        if changed:
            self.status.setText("Prompt edited — rerun to refresh output and J-space")

    def _chat_handoff_text(self) -> str:
        concepts = sorted(
            self._run_activations.values(), key=lambda value: value.score, reverse=True
        )[:20]
        concept_lines = [
            f"- {value.term}: {value.score:+.3f} (L{value.layer}, P{value.token_index})"
            for value in concepts
        ] or ["- No concepts captured"]
        intervention_lines = []
        for entry in self.project.interventions:
            draft = entry.draft
            intervention_lines.append(
                f"- [{'enabled' if entry.enabled else 'disabled'}] "
                f"{draft.operation.value}: {draft.source_term or '—'} → "
                f"{draft.target_term or '—'}; max budget {draft.strength:.3g}; "
                f"layers {draft.layer_start}-{draft.layer_end}"
            )
        if not intervention_lines:
            intervention_lines.append("- None")
        return "\n".join(
            (
                "Prompt:",
                self.prompt.toPlainText().strip(),
                "",
                "Model output:",
                self.output_text,
                "",
                "J-Lens concepts:",
                *concept_lines,
                "",
                "Intervention edits:",
                *intervention_lines,
            )
        )

    def _on_error(self, run_id: str, message: str, detail: str) -> None:
        self.current_run_id = None
        self._set_run_state(RunState.READY)
        visible_detail = detail if len(detail) <= 180 else f"{detail[:177]}…"
        self.status.setText(
            f"{message} — {visible_detail}" if visible_detail else message
        )
        self.status.setToolTip(detail)

    def _replace_activations(self, activations) -> None:
        self.activation_model.replace_rows(activations)
        count = self.activation_model.rowCount()
        self.found_count.setText(f"Found: {count} Concept{'s' if count != 1 else ''}")

    def _selected_term(self) -> str:
        rows = self.found_table.selectionModel().selectedRows()
        if not rows:
            return ""
        return self.activation_model.activation(rows[0].row()).term

    def _request_jlens(self) -> None:
        if self.last_run_id is None:
            self.status.setText("Run First Read before opening J-Lens")
            return
        selected = self.found_table.selectionModel().selectedRows()
        position = 0
        if selected:
            position = self.activation_model.activation(selected[0].row()).token_index
        self.jlens_requested.emit(self.last_run_id, position)

    def _double_click_activation(self, index) -> None:
        self.open_intervention_editor(
            "inject", self.activation_model.activation(index.row()).term
        )

    def _add_selected(self) -> None:
        term = self._selected_term()
        if term:
            self.open_intervention_editor("inject", term)

    def _show_context_menu(self, position) -> None:
        term = self._selected_term()
        if not term:
            return
        menu = QMenu(self)
        for label, operation in (
            ("Inject selected term…", "inject"),
            ("Replace selected term…", "replace"),
            ("Suppress selected term…", "suppress"),
        ):
            action = QAction(label, menu)
            action.triggered.connect(
                lambda _checked=False, op=operation: self.open_intervention_editor(op, term)
            )
            menu.addAction(action)
        menu.addSeparator()
        menu.addAction("Model View", self.model_view_requested)
        menu.addAction("Open in J-Lens", self._request_jlens)
        menu.addAction("Trace Influence")
        menu.addAction("Copy Details")
        menu.popup(self.found_table.viewport().mapToGlobal(position))

    def open_intervention_editor(self, operation: str, term: str) -> InterventionEditor:
        if self.session is None:
            self.session_requested.emit()
            raise RuntimeError("select a session before creating an intervention")
        editor = InterventionEditor(
            self.session, operation=operation, term=term, parent=self.window()
        )
        editor.draft_added.connect(self._add_draft)
        editor.preview_requested.connect(self._preview_draft)
        editor.finished.connect(
            lambda: self._editors.remove(editor) if editor in self._editors else None
        )
        self._editors.append(editor)
        editor.show()
        editor.raise_()
        editor.activateWindow()
        return editor

    def _preview_draft(self, draft) -> None:
        valid, detail = self.services.interventions.preview(self.session.session_id, draft)
        editor = self._editors[-1] if self._editors else None
        if editor is not None:
            editor.preview.setText(("✓ " if valid else "⚠ ") + detail)

    def preview_selected_intervention(self) -> None:
        rows = self.stack_view.table.selectionModel().selectedRows()
        if not rows:
            self.status.setText("Select an intervention to preview")
            return
        entry = self.intervention_model.record(rows[0].row())
        valid, detail = self.services.interventions.preview(
            self.session.session_id, entry.draft
        )
        self.status.setText(("Compatible · " if valid else "Unavailable · ") + detail)

    def _add_draft(self, draft) -> None:
        entry = InterventionEntry.from_draft(draft)
        self.project.interventions.append(entry)
        self.project.dirty = True
        self.intervention_model.replace_rows(self.project.interventions)
        self._sync_session_state()

    def _clear_stack(self) -> None:
        self.project.interventions.clear()
        self.intervention_model.replace_rows(())
        self._sync_session_state()

    def set_selected_enabled(self, enabled: bool) -> None:
        from dataclasses import replace

        selected = {
            index.row() for index in self.stack_view.table.selectionModel().selectedRows()
        }
        if not selected:
            self.status.setText("Select intervention rows first")
            return
        self.project.interventions[:] = [
            replace(entry, enabled=enabled) if row in selected else entry
            for row, entry in enumerate(self.project.interventions)
        ]
        self.intervention_model.replace_rows(self.project.interventions)
        self.project.dirty = True

    def _intervention_enabled_changed(self, row: int, enabled: bool) -> None:
        from dataclasses import replace

        self.project.interventions[row] = replace(
            self.project.interventions[row], enabled=enabled
        )
        self.project.dirty = True

    def _arm_stack(self) -> None:
        from dataclasses import replace

        self.project.interventions[:] = [
            replace(
                entry,
                enabled=True,
                state=InterventionState.ARMED,
                status_detail="Armed",
            )
            for entry in self.project.interventions
        ]
        self.intervention_model.replace_rows(self.project.interventions)
        self.arm_button.setText("Disarm Stack")
