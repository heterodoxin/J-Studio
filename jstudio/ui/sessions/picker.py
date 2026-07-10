"""Model Session Picker dialog."""

from PySide6.QtCore import QSortFilterProxyModel, Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QSplitter,
    QTableView,
    QTabWidget,
    QVBoxLayout,
)

from jstudio.domain import BackendKind
from jstudio.ui.models import SessionTableModel


class SessionPickerDialog(QDialog):
    session_opened = Signal(object)

    def __init__(self, service, parent=None) -> None:
        super().__init__(parent)
        self.service = service
        self.setWindowTitle("Select a Model Session")
        self.resize(820, 580)
        self.setMinimumSize(660, 440)
        root = QVBoxLayout(self)
        self.search = QLineEdit(self)
        self.search.setPlaceholderText("Search models, workers, or traces")
        self.search.setClearButtonEnabled(True)
        root.addWidget(self.search)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.tabs = QTabWidget(splitter)
        self.preview = QLabel("Select a session to inspect compatibility.", splitter)
        self.preview.setWordWrap(True)
        self.preview.setMinimumWidth(250)
        sessions = service.list_sessions()
        self._tables = []
        for title, kind in (
            ("Local Models", BackendKind.LOCAL),
            ("Remote Workers", BackendKind.REMOTE_WORKER),
            ("Offline Traces", BackendKind.OFFLINE_TRACE),
        ):
            table = QTableView(self.tabs)
            source = SessionTableModel(
                [s for s in sessions if s.backend_kind is kind], table
            )
            proxy = QSortFilterProxyModel(table)
            proxy.setSourceModel(source)
            proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
            proxy.setFilterKeyColumn(-1)
            table.setModel(proxy)
            table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
            table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
            table.setSortingEnabled(True)
            table.horizontalHeader().setStretchLastSection(True)
            table.selectionModel().selectionChanged.connect(self._selection_changed)
            table.activated.connect(lambda _index, t=table: self._open_from_table(t))
            self.tabs.addTab(table, title)
            self._tables.append(table)
        self.local_table, self.remote_table, self.offline_table = self._tables
        splitter.addWidget(self.tabs)
        splitter.addWidget(self.preview)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 2)
        root.addWidget(splitter, 1)
        footer = QHBoxLayout()
        footer.addWidget(QPushButton("Manage Models…"))
        footer.addWidget(QPushButton("Connect Worker…"))
        footer.addWidget(QPushButton("Open Trace…"))
        footer.addStretch(1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        self.open_button = buttons.addButton("Open", QDialogButtonBox.ButtonRole.AcceptRole)
        self.open_button.setEnabled(False)
        footer.addWidget(buttons)
        root.addLayout(footer)
        buttons.rejected.connect(self.reject)
        self.open_button.clicked.connect(self._open_current)
        self.search.textChanged.connect(self._filter)
        self.search.setFocus()

    def _filter(self, text: str) -> None:
        for table in self._tables:
            table.model().setFilterFixedString(text)

    def _current_session(self, table=None):
        table = table or self._tables[self.tabs.currentIndex()]
        index = table.currentIndex()
        if not index.isValid():
            return None
        source_index = table.model().mapToSource(index)
        return table.model().sourceModel().record(source_index.row())

    def _selection_changed(self) -> None:
        session = self._current_session()
        self.open_button.setEnabled(session is not None)
        if session is None:
            self.preview.setText("Select a session to inspect compatibility.")
            return
        capabilities = []
        if session.capabilities.inspect:
            capabilities.append("inspection")
        if session.capabilities.generate:
            capabilities.append("generation")
        if session.capabilities.intervene:
            capabilities.append("interventions")
        lens = session.lens_id or "Missing — load a compatible J-lens"
        self.preview.setText(
            f"<b>{session.display_name or session.model_id}</b><br>"
            f"Revision: {session.revision}<br>Lens: {lens}<br>"
            f"Layers: {session.layer_count}<br>Capabilities: {', '.join(capabilities)}"
        )

    def _open_current(self) -> None:
        self._open_from_table(self._tables[self.tabs.currentIndex()])

    def _open_from_table(self, table) -> None:
        session = self._current_session(table)
        if session is None:
            return
        try:
            opened = self.service.open_session(session.session_id)
        except Exception as exc:
            self.preview.setText(
                f"<b>Opening failed.</b><br>{exc}<br>Selection has been preserved."
            )
            return
        self.session_opened.emit(opened)
        self.accept()
