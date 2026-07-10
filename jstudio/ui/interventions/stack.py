"""Intervention stack table and controls."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QHBoxLayout, QPushButton, QTableView, QVBoxLayout, QWidget

from jstudio.ui.models import InterventionTableModel


class InterventionStackView(QWidget):
    def __init__(self, model: InterventionTableModel, parent=None) -> None:
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        toolbar = QHBoxLayout()
        self.inject = QPushButton("Inject")
        self.replace = QPushButton("Replace")
        self.suppress = QPushButton("Suppress")
        self.group = QPushButton("Group")
        self.preview = QPushButton("Preview")
        self.arm = QPushButton("Arm Stack")
        self.clear = QPushButton("Clear")
        for button in (
            self.inject,
            self.replace,
            self.suppress,
            self.group,
            self.preview,
            self.arm,
            self.clear,
        ):
            toolbar.addWidget(button)
        toolbar.addStretch(1)
        self.table = QTableView(self)
        self.table.setModel(model)
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.table.clicked.connect(self._toggle_enabled_cell)
        layout.addLayout(toolbar)
        layout.addWidget(self.table, 1)

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
