"""Global command construction and state gating."""

from __future__ import annotations

from PySide6.QtGui import QAction, QKeySequence

from jstudio.domain import ModelSessionSummary, RunState


class CommandRegistry(dict[str, QAction]):
    def __init__(self, parent) -> None:
        super().__init__()
        definitions = {
            "new_project": ("New Project", "Ctrl+N"),
            "open_project": ("Open Project", "Ctrl+O"),
            "save": ("Save", "Ctrl+S"),
            "save_as": ("Save As", "Ctrl+Shift+S"),
            "import_trace": ("Import Trace", ""),
            "export_report": ("Export Report", ""),
            "close_project": ("Close Project", ""),
            "quit": ("Quit", "Ctrl+Q"),
            "undo_read": ("Undo Read", ""),
            "settings": ("Settings", "Ctrl+,"),
            "select_session": ("Select Model Session", "Ctrl+K"),
            "recent_models": ("Recent Models", ""),
            "load_local": ("Load Local Model", ""),
            "connect_worker": ("Connect Worker", ""),
            "open_offline": ("Open Offline Trace", ""),
            "load_lens": ("Load Lens", ""),
            "model_information": ("Model Information", ""),
            "detach": ("Detach", ""),
            "add_intervention": ("Add Intervention", ""),
            "add_group": ("Add Group", ""),
            "add_rule": ("Add Rule", ""),
            "activate_selected": ("Activate Selected", ""),
            "deactivate_selected": ("Deactivate Selected", ""),
            "table_files": ("Table Files", ""),
            "export_activations": ("Export Activations", ""),
            "model_view": ("Model View", ""),
            "layer_explorer": ("Layer Explorer", ""),
            "jlens_sweep": ("J-Lens Sweep", ""),
            "influence_trace": ("Influence Trace", ""),
            "generation_trace": ("Generation Trace", ""),
            "experiments": ("Experiments", ""),
            "snapshot_manager": ("Snapshot Manager", ""),
            "rules": ("Rules", "Ctrl+2"),
            "concept_help": ("J-space Concepts", ""),
            "rule_api": ("Rule API", ""),
            "keyboard_reference": ("Keyboard Reference", ""),
            "research_references": ("Research References", ""),
            "report_issue": ("Report Issue", ""),
            "about": ("About", ""),
            "first_read": ("First Read", "Ctrl+Enter"),
            "next_token": ("Generate Next Token", "F10"),
        }
        for name, (text, shortcut) in definitions.items():
            action = QAction(text, parent)
            if shortcut:
                native_shortcut = "Ctrl+Return" if shortcut == "Ctrl+Enter" else shortcut
                action.setShortcut(QKeySequence(native_shortcut))
            self[name] = action
        self.refresh(None, RunState.READY)

    def refresh(self, session: ModelSessionSummary | None, run_state: RunState) -> None:
        has_session = session is not None
        inspect = has_session and session.capabilities.inspect
        generate = has_session and session.capabilities.generate
        intervene = has_session and session.capabilities.intervene
        running = run_state in (RunState.RUNNING, RunState.PAUSED)
        requirements = {
            "first_read": (generate, "Select a generative model session"),
            "next_token": (generate and running, "Start generation first"),
            "load_lens": (has_session, "Select a model session first"),
            "model_information": (has_session, "Select a model session first"),
            "detach": (has_session, "No model session is attached"),
            "add_intervention": (
                intervene,
                "Select a live session with intervention support",
            ),
            "activate_selected": (intervene, "Live intervention support is required"),
            "deactivate_selected": (intervene, "Live intervention support is required"),
            "model_view": (inspect, "Load a session or offline trace first"),
            "layer_explorer": (
                inspect and bool(session and session.lens_id),
                "Load a compatible J-lens",
            ),
            "jlens_sweep": (
                inspect and bool(session and session.lens_id),
                "Load a compatible J-lens",
            ),
            "influence_trace": (
                inspect and bool(session and session.lens_id),
                "Load a compatible J-lens",
            ),
            "generation_trace": (has_session, "Select a model session first"),
            "experiments": (has_session, "Select a model session first"),
        }
        for name, (enabled, reason) in requirements.items():
            self[name].setEnabled(enabled)
            self[name].setStatusTip("" if enabled else reason)


def populate_menus(window, commands: CommandRegistry) -> None:
    menu_specs = (
        (
            "File",
            (
                "new_project",
                "open_project",
                "save",
                "save_as",
                "import_trace",
                "export_report",
                "close_project",
                "quit",
            ),
        ),
        ("Edit", ("undo_read", "settings")),
        (
            "Model",
            (
                "select_session",
                "recent_models",
                "load_local",
                "connect_worker",
                "open_offline",
                "load_lens",
                "model_information",
                "detach",
            ),
        ),
        (
            "Table",
            (
                "add_intervention",
                "add_group",
                "add_rule",
                "activate_selected",
                "deactivate_selected",
                "table_files",
                "export_activations",
            ),
        ),
        (
            "Tools",
            (
                "model_view",
                "layer_explorer",
                "jlens_sweep",
                "influence_trace",
                "generation_trace",
                "experiments",
                "snapshot_manager",
                "rules",
            ),
        ),
        (
            "Help",
            (
                "concept_help",
                "rule_api",
                "keyboard_reference",
                "research_references",
                "report_issue",
                "about",
            ),
        ),
    )
    for title, names in menu_specs:
        menu = window.menuBar().addMenu(title)
        for name in names:
            menu.addAction(commands[name])
