"""Intervention stack table and controls."""

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QHBoxLayout,
    QMenu,
    QPushButton,
    QTableView,
    QVBoxLayout,
    QWidget,
)

from jstudio.ui.models import InterventionTableModel


class InterventionStackView(QWidget):
    action_requested = Signal(str, object)

    def __init__(self, model: InterventionTableModel, parent=None) -> None:
        super().__init__(parent)
        self.setProperty("role", "panel")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 12)
        layout.setSpacing(8)
        toolbar = QHBoxLayout()
        toolbar.setSpacing(6)
        self.inject = QPushButton("Inject")
        self.replace = QPushButton("Replace")
        self.suppress = QPushButton("Suppress")
        self.group = QPushButton("Group")
        self.preview = QPushButton("Preview")
        self.arm = QPushButton("Arm Stack")
        self.bake = QPushButton("Bake Stack")
        self.clear = QPushButton("Clear")
        self.inject.setProperty("operation", "inject")
        self.replace.setProperty("operation", "replace")
        self.suppress.setProperty("operation", "suppress")
        self.arm.setProperty("role", "primary")
        self.clear.setProperty("role", "danger")
        for button in (
            self.inject,
            self.replace,
            self.suppress,
            self.group,
            self.preview,
            self.arm,
            self.bake,
            self.clear,
        ):
            toolbar.addWidget(button)
        toolbar.insertStretch(3, 1)
        toolbar.insertStretch(8, 1)
        self.table = QTableView(self)
        self.table.setModel(model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.clicked.connect(self._toggle_enabled_cell)
        self.table.customContextMenuRequested.connect(self._show_context_menu)
        layout.addLayout(toolbar)
        layout.addWidget(self.table, 1)

    def build_context_menu(self, rows: tuple[int, ...]) -> QMenu:
        rows = tuple(
            sorted(
                {
                    row
                    for row in rows
                    if 0 <= row < self.table.model().rowCount()
                }
            )
        )
        menu = QMenu(self)
        records = [self.table.model().record(row) for row in rows]
        enable_label = (
            "Disable"
            if records and all(row.enabled for row in records)
            else "Enable"
        )
        single = len(rows) == 1
        specifications = (
            (
                enable_label,
                "disable" if enable_label == "Disable" else "enable",
                bool(rows),
            ),
            ("Edit…", "edit", single),
            ("Duplicate", "duplicate", bool(rows)),
            ("Preview", "preview", single),
            ("Move Up", "move-up", single and rows[0] > 0),
            (
                "Move Down",
                "move-down",
                single and rows[0] < self.table.model().rowCount() - 1,
            ),
            ("Remove", "remove", bool(rows)),
        )
        for label, command, enabled in specifications:
            action = menu.addAction(label)
            action.setEnabled(enabled)
            action.triggered.connect(
                lambda _checked=False, value=command, selected=rows: (
                    self.action_requested.emit(value, selected)
                )
            )
        return menu

    def _show_context_menu(self, position) -> None:
        index = self.table.indexAt(position)
        if not index.isValid():
            return
        selected = {row.row() for row in self.table.selectionModel().selectedRows()}
        if index.row() not in selected:
            self.table.clearSelection()
            self.table.selectRow(index.row())
        rows = tuple(
            sorted(row.row() for row in self.table.selectionModel().selectedRows())
        )
        self.build_context_menu(rows).popup(
            self.table.viewport().mapToGlobal(position)
        )

    def _toggle_enabled_cell(self, index) -> None:
        if not index.isValid() or index.column() != 0:
            return
        model = self.table.model()
        state = model.data(index, Qt.ItemDataRole.CheckStateRole)
        next_state = (
            Qt.CheckState.Unchecked
            if state == Qt.CheckState.Checked
            else Qt.CheckState.Checked
        )
        model.setData(index, next_state, Qt.ItemDataRole.CheckStateRole)
