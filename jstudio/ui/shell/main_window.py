"""Native J Studio application window."""

from __future__ import annotations

import json
from dataclasses import replace

from PySide6.QtCore import QSize, Qt, Signal
from PySide6.QtWidgets import (
    QApplication,
    QFileDialog,
    QMainWindow,
    QMenu,
    QMessageBox,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from jstudio.domain import LensFitState, LensFitStatus, ModelSessionSummary, RunState
from jstudio.project import ProjectDocument
from jstudio.services.protocols import JStudioServices
from jstudio.ui.chat import ChatWorkspace
from jstudio.ui.lensview.workspace import JLensWorkspace
from jstudio.ui.main_workspace import MainReadWorkspace
from jstudio.ui.rules.workspace import RulesWorkspace
from jstudio.ui.secondary import TOOL_CLASSES
from jstudio.ui.sessions.picker import SessionPickerDialog
from jstudio.ui.settings import SettingsWindow
from jstudio.ui.shell.commands import CommandRegistry, populate_menus
from jstudio.ui.shell.session_bar import SessionBar


class JStudioMainWindow(QMainWindow):
    COMPACT_SIZE = QSize(1101, 888)
    _fit_changed = Signal(object)

    def __init__(
        self,
        services: JStudioServices,
        project: ProjectDocument,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.services = services
        self.project = project
        self.current_session: ModelSessionSummary | None = None
        self.run_state = RunState.READY
        self._analysis_auto_expanded = False
        self._tool_windows = {}
        self._settings_window = None
        self.setWindowTitle("J Studio")
        self.setMinimumSize(self.COMPACT_SIZE)
        self.resize(self.COMPACT_SIZE)

        self.commands = CommandRegistry(self)
        populate_menus(self, self.commands)

        central = QWidget(self)
        central.setObjectName("applicationRoot")
        layout = QVBoxLayout(central)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.session_bar = SessionBar(central)
        self.tabs = QTabWidget(central)
        self.tabs.setObjectName("workspaceTabs")
        self.tabs.setDocumentMode(False)
        self.tabs.setMovable(False)
        self.tabs.setTabsClosable(False)

        self.main_workspace = MainReadWorkspace(services, project, self)
        self.chat_workspace = ChatWorkspace(services, project, self)
        self.jlens_workspace = JLensWorkspace(services, self)
        self.rules_workspace = RulesWorkspace(services.rules, project, self)
        for title, widget in (
            ("Main", self.main_workspace),
            ("Chat", self.chat_workspace),
            ("J-Lens", self.jlens_workspace),
            ("Rules", self.rules_workspace),
        ):
            # A hidden analysis page must not force the compact Main tab to use
            # its much larger minimum size. The selected page remains resizable.
            widget.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
            self.tabs.addTab(widget, title)
        layout.addWidget(self.session_bar)
        layout.addWidget(self.tabs, 1)
        self.setCentralWidget(central)
        self.tabs.currentChanged.connect(self._tab_changed)
        self.statusBar().showMessage("Ready")
        self._wired_commands: set[str] = set()
        self._wire_commands()

        self.session_bar.select_button.clicked.connect(
            lambda: self.commands["select_session"].trigger()
        )
        self.session_bar.overflow.clicked.connect(self._session_menu)
        self.main_workspace.session_requested.connect(self.open_session_picker)
        self.main_workspace.run_state_changed.connect(self._run_state_changed)
        self.chat_workspace.inspect_requested.connect(self._inspect_run)
        self.main_workspace.bridge.token.connect(self._stream_main_run_to_jlens)
        self.main_workspace.rules_requested.connect(
            lambda: self.commands["rules"].trigger()
        )
        self.main_workspace.model_view_requested.connect(
            lambda: self.open_tool("model_view")
        )
        self.main_workspace.jlens_requested.connect(self._inspect_run)
        self.main_workspace.chat_requested.connect(self._continue_main_in_chat)
        self.chat_workspace.controls_requested.connect(
            lambda: self.tabs.setCurrentWidget(self.main_workspace)
        )
        self.chat_workspace.rules_requested.connect(
            lambda: self.tabs.setCurrentWidget(self.rules_workspace)
        )
        self.main_workspace.advanced_button.clicked.connect(
            lambda: self.open_tool("generation_trace")
        )
        self._fit_changed.connect(self._apply_fit_status)
        self._unsubscribe_fit = self.services.lens.subscribe_fit(self._fit_changed.emit)

    def _connect_command(self, name: str, callback) -> None:
        self.commands[name].triggered.connect(callback)
        self._wired_commands.add(name)

    def _wire_commands(self) -> None:
        direct = {
            "new_project": self.new_project,
            "open_project": self.choose_open_project,
            "save": self.choose_save_project,
            "save_as": lambda: self.choose_save_project(force_dialog=True),
            "close_project": self.new_project,
            "quit": self.close,
            "undo_read": self.main_workspace._undo_action,
            "settings": self.open_settings,
            "select_session": self.open_session_picker,
            "detach": lambda: self.set_session(None),
            "add_intervention": lambda: self.main_workspace.open_intervention_editor(
                "inject", ""
            ),
            "add_rule": self._new_rule,
            "load_lens": self.request_stable_lens_fit,
            "activate_selected": lambda: self.main_workspace.set_selected_enabled(True),
            "deactivate_selected": lambda: self.main_workspace.set_selected_enabled(False),
            "export_activations": self.export_activations,
            "export_report": self.export_report,
            "rules": lambda: self.tabs.setCurrentWidget(self.rules_workspace),
            "first_read": self.main_workspace._first_action,
            "next_token": self.main_workspace._next_action,
        }
        for name, callback in direct.items():
            self._connect_command(name, callback)
        for name in (
            "recent_models",
            "load_local",
            "connect_worker",
            "open_offline",
            "import_trace",
        ):
            self._connect_command(name, self.open_session_picker)
        for name in TOOL_CLASSES:
            self._connect_command(
                name, lambda _checked=False, tool=name: self.open_tool(tool)
            )
        messages = {
            "model_information": (
                "Model and lens provenance are shown in the session bar and J-Lens Details."
            ),
            "add_group": (
                "Select intervention rows, then drag them together to define "
                "execution order."
            ),
            "table_files": (
                "Portable project data is stored in the current .jstudio.json file."
            ),
            "concept_help": (
                "Found Concepts are fitted Jacobian-lens readouts across prompt "
                "positions and layers."
            ),
            "rule_api": (
                "Rules return declarative inject, replace, suppress, tag, log, "
                "or stop actions."
            ),
            "keyboard_reference": (
                "Ctrl+Enter Run/Pause · F10 Next Token · Ctrl+2 Rules · Ctrl+K Session"
            ),
            "research_references": "See Help references in the J Studio design document.",
            "report_issue": (
                "Save the project and generation trace, then attach both to the issue."
            ),
            "about": (
                "J Studio · decoder-model Jacobian inspection and calibrated interventions"
            ),
        }
        for name, message in messages.items():
            self._connect_command(
                name,
                lambda _checked=False, text=message: QMessageBox.information(
                    self, "J Studio", text
                ),
            )

    def _session_menu(self) -> None:
        menu = QMenu(self)
        for name in ("select_session", "model_information", "load_lens", "detach"):
            menu.addAction(self.commands[name])
        menu.popup(
            self.session_bar.overflow.mapToGlobal(
                self.session_bar.overflow.rect().bottomLeft()
            )
        )

    def _new_rule(self) -> None:
        self.tabs.setCurrentWidget(self.rules_workspace)
        self.rules_workspace.new_rule("New Rule")

    def request_stable_lens_fit(self) -> None:
        self.services.lens.start_fit()
        self.statusBar().showMessage("Stable lens fit requested")

    def new_project(self) -> None:
        self._bind_project(ProjectDocument.new())
        self.setWindowTitle("J Studio")

    def choose_open_project(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Open J Studio Project", "", "J Studio Project (*.jstudio.json *.json)"
        )
        if path:
            self.open_project(path)

    def choose_save_project(self, *, force_dialog: bool = False) -> None:
        path = None if force_dialog else self.project.path
        if path is None:
            selected, _ = QFileDialog.getSaveFileName(
                self, "Save J Studio Project", "", "J Studio Project (*.jstudio.json)"
            )
            if not selected:
                return
            path = selected
        self.save_project(path)

    def export_activations(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Activations", "activations.json", "JSON (*.json)"
        )
        if not path:
            return
        values = [
            self.main_workspace.activation_model.activation(row)
            for row in range(self.main_workspace.activation_model.rowCount())
        ]
        payload = [
            {
                "term": value.term,
                "score": value.score,
                "confidence": value.confidence,
                "layer": value.layer,
                "token_index": value.token_index,
            }
            for value in values
        ]
        with open(path, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)

    def export_report(self) -> None:
        path, _ = QFileDialog.getSaveFileName(
            self, "Export Project Report", "j-studio-report.json", "JSON (*.json)"
        )
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                json.dump(self.project.to_dict(), handle, indent=2)

    def set_session(self, session: ModelSessionSummary | None) -> None:
        self.current_session = session
        self.session_bar.set_session(session)
        self.jlens_workspace.set_lens_identity(session.lens_id if session else None)
        self.main_workspace.set_session(session)
        self.chat_workspace.set_session(session)
        self.commands.refresh(session, self.run_state)
        if session is not None:
            known = any(
                value.session_id == session.session_id
                for value in self.services.sessions.list_sessions()
            )
            if known:
                self._apply_fit_status(self.services.lens.fit_status())

    def _apply_fit_status(self, status: LensFitStatus) -> None:
        self.session_bar.set_fit_status(status)
        self.jlens_workspace.set_fit_status(status)
        if self.current_session is None:
            return
        session = self.current_session
        try:
            refreshed = next(
                value
                for value in self.services.sessions.refresh()
                if value.session_id == session.session_id
            )
        except (StopIteration, KeyError):
            refreshed = session
        inspect = bool(refreshed.capabilities.inspect)
        intervene = (
            refreshed.capabilities.intervene
            if status.state is LensFitState.STABLE
            else False
        )
        lens_id = session.lens_id
        if status.state is LensFitState.STABLE:
            lens_id = refreshed.lens_id or session.lens_id
        elif not inspect:
            lens_id = None
        updated = replace(
            refreshed,
            lens_id=lens_id,
            capabilities=replace(
                refreshed.capabilities,
                inspect=inspect,
                intervene=intervene,
            ),
        )
        self.current_session = updated
        self.session_bar.set_session(updated)
        self.session_bar.set_fit_status(status)
        self.jlens_workspace.set_lens_identity(updated.lens_id)
        self.main_workspace.set_session(updated)
        self.chat_workspace.set_session(updated)
        self.commands.refresh(updated, self.run_state)

    def _run_state_changed(self, state: RunState) -> None:
        self.run_state = state
        self.commands.refresh(self.current_session, state)

    def _tab_changed(self, index: int) -> None:
        selected = self.tabs.widget(index)
        if (
            selected is self.jlens_workspace
            and self.main_workspace.last_run_id
            and self.jlens_workspace.run_id != self.main_workspace.last_run_id
        ):
            self._inspect_run(self.main_workspace.last_run_id, 0)
        if selected is self.jlens_workspace and (
            self.width() < 1000 or self.height() < 720
        ):
            self._analysis_auto_expanded = True
            self.resize(1100, 760)
        elif selected is self.main_workspace and self._analysis_auto_expanded:
            self._analysis_auto_expanded = False
            self.resize(self.COMPACT_SIZE)

    def open_session_picker(self) -> SessionPickerDialog:
        picker = SessionPickerDialog(self.services.sessions, self)
        picker.session_opened.connect(self.set_session)
        picker.show()
        return picker

    def _inspect_run(self, run_id: str, position: int) -> None:
        frames = self.services.lens.frames(run_id)
        run = next(
            (candidate for candidate in self.project.runs if candidate.run_id == run_id),
            None,
        )
        if run_id == self.main_workspace.last_run_id:
            if run is not None and run.inspection_text:
                text = run.inspection_text
            else:
                prompt = self.main_workspace.prompt.toPlainText().strip()
                output = self.main_workspace.output_text.strip()
                if run is not None and run.output_text:
                    output = run.output_text.strip()
                text = prompt + ("\n\n" + output if output else "")
        elif run is None:
            text = "".join(frame.token_text for frame in frames) or "No captured text"
        else:
            text = run.inspection_text or (
                run.prompt + ("\n\n" + run.output_text if run.output_text else "")
            )
        self.jlens_workspace.inspect(
            run_id, text, "Chat inspection", position=position
        )
        self.tabs.setCurrentWidget(self.jlens_workspace)

    def _continue_main_in_chat(self, handoff: str) -> None:
        self.chat_workspace.sync_from_main(handoff)
        self.tabs.setCurrentWidget(self.chat_workspace)

    def _stream_main_run_to_jlens(
        self, run_id: str, _token: str, output_text: str
    ) -> None:
        if self.jlens_workspace.run_id != run_id:
            return
        prompt = self.main_workspace.prompt.toPlainText().strip()
        text = prompt + ("\n\n" + output_text if output_text else "")
        self.jlens_workspace.stream_text(run_id, text)
        self.chat_workspace.composer.setFocus()

    def open_tool(self, name: str):
        window = self._tool_windows.get(name)
        if window is None:
            window = TOOL_CLASSES[name](self)
            self._tool_windows[name] = window
            self._configure_tool(name, window)
        self._refresh_tool(name, window)
        window.show()
        window.raise_()
        window.activateWindow()
        return window

    def _configure_tool(self, name: str, window) -> None:
        if name == "generation_trace":
            bindings = {
                "Generate": self.main_workspace._first_action,
                "One Token": self.main_workspace._next_action,
                "Pause": self.main_workspace._first_action,
                "Resume": self.main_workspace._first_action,
                "Stop": self.main_workspace._undo_action,
                "Baseline": self.main_workspace._first_action,
                "With Stack": self.main_workspace._first_action,
                "Compare": lambda: self.open_tool("experiments"),
            }
            for label, callback in bindings.items():
                window.actions[label].triggered.connect(callback)
        elif name == "jlens_sweep":
            window.run_button.clicked.connect(self._run_sweep)
        elif name == "influence_trace":
            window.run_button.clicked.connect(
                lambda: window.graph.set_terms(self._active_terms())
            )
        elif name == "snapshot_manager":
            window.capture_button.clicked.connect(
                lambda: window.snapshots.addItem(
                    f"{self.main_workspace.last_run_id or 'No run'} · "
                    f"{len(self._active_terms())} concepts"
                )
            )
        elif name == "experiments":
            window.run_button.clicked.connect(self.main_workspace._first_action)
        elif name == "layer_explorer":
            window.actions["Trace Influence"].clicked.connect(
                lambda: self.open_tool("influence_trace")
            )
            window.actions["Inject at Selection"].clicked.connect(
                lambda: self.main_workspace.open_intervention_editor(
                    "inject", self._active_terms()[0] if self._active_terms() else ""
                )
            )
            window.actions["Replace at Selection"].clicked.connect(
                lambda: self.main_workspace.open_intervention_editor(
                    "replace", self._active_terms()[0] if self._active_terms() else ""
                )
            )
            window.actions["Suppress at Selection"].clicked.connect(
                lambda: self.main_workspace.open_intervention_editor(
                    "suppress", self._active_terms()[0] if self._active_terms() else ""
                )
            )
            window.actions["Copy Coordinates"].clicked.connect(
                lambda: QApplication.clipboard().setText(
                    f"run={self.jlens_workspace.run_id} "
                    f"position={self.jlens_workspace.selection.position} "
                    f"layer={self.jlens_workspace.selection.layer}"
                )
            )

    def _active_terms(self) -> list[str]:
        return [
            self.main_workspace.activation_model.activation(row).term
            for row in range(self.main_workspace.activation_model.rowCount())
        ]

    def _run_sweep(self) -> None:
        window = self._tool_windows["jlens_sweep"]
        prompt = window.prompt.toPlainText().strip()
        if prompt:
            self.main_workspace.prompt.setPlainText(prompt)
        self.tabs.setCurrentWidget(self.main_workspace)
        self.main_workspace._first_action()

    def _refresh_tool(self, name: str, window) -> None:
        if name == "model_view":
            window.response.setPlainText(
                self.main_workspace.output_text
                or "No generated response yet. Run First Read from Main."
            )
        elif name == "layer_explorer":
            selection = self.jlens_workspace.selection
            window.provenance.setText(
                f"Run {selection.run_id or '—'} · Layer {selection.layer} · "
                f"Position {selection.position}"
            )
            terms = self._active_terms()
            window.details.setPlainText("\n".join(terms[:100]) or "No readout selected")
        elif name == "jlens_sweep" and not window.prompt.toPlainText():
            window.prompt.setPlainText(self.main_workspace.prompt.toPlainText())
        elif name == "influence_trace":
            window.graph.set_terms(self._active_terms())

    def tool_window(self, name: str):
        return self._tool_windows.get(name)

    def open_settings(self):
        if self._settings_window is None:
            self._settings_window = SettingsWindow(self)
        self._settings_window.show()
        self._settings_window.raise_()
        return self._settings_window

    def _bind_project(self, project: ProjectDocument) -> None:
        self.project = project
        self.main_workspace.project = project
        self.main_workspace.intervention_model.replace_rows(project.interventions)
        self.chat_workspace.project = project
        self.chat_workspace.refresh_control_status()
        self.rules_workspace.project = project
        self.rules_workspace.rule_model.replace_rows(project.rules)

    def open_project(self, path, *, imported: bool = False) -> None:
        project = ProjectDocument.load(path, imported=imported)
        self._bind_project(project)
        self.setWindowTitle(f"J Studio — {project.name}")

    def save_project(self, path=None) -> None:
        self.project.save(path)
        self.setWindowTitle(f"J Studio — {self.project.path.name}")

    def keyPressEvent(self, event) -> None:
        if (
            event.modifiers() & Qt.KeyboardModifier.ControlModifier
            and event.key() == Qt.Key.Key_2
        ):
            self.commands["rules"].trigger()
            event.accept()
            return
        if event.key() == Qt.Key.Key_F10:
            if self.commands["next_token"].isEnabled():
                self.commands["next_token"].trigger()
            event.accept()
            return
        if event.modifiers() & Qt.KeyboardModifier.ControlModifier and event.key() in (
            Qt.Key.Key_Return,
            Qt.Key.Key_Enter,
        ):
            if self.commands["first_read"].isEnabled():
                self.commands["first_read"].trigger()
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:
        self._unsubscribe_fit()
        self.rules_workspace.shutdown()
        self.services.generation.close()
        super().closeEvent(event)
