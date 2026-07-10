"""Startup model and lens selection."""

from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from jstudio.services.hf_cache import CachedModel, LensState, scan_hf_cache


class StartupModelDialog(QDialog):
    def __init__(
        self,
        models: tuple[CachedModel, ...],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.models = models
        self._manual_lens_path: Path | None = None
        self._auto_fit_requested = False
        self.setWindowTitle("Choose cached model")
        self.resize(920, 520)

        layout = QVBoxLayout(self)
        title = QLabel(
            "Choose a Hugging Face cached model. Use an existing Stable lens, "
            "point to a lens file, or let J Studio fit one after loading."
        )
        title.setWordWrap(True)
        layout.addWidget(title)

        self.table = QTableWidget(len(models), 4, self)
        self.table.setHorizontalHeaderLabels(("Model", "Lens", "Cache", "Detail"))
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        self.table.itemSelectionChanged.connect(self._selection_changed)
        self.table.doubleClicked.connect(lambda _index: self.accept())
        self.table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.table, 1)

        lens_row = QHBoxLayout()
        self.lens_path = QLineEdit(self)
        self.lens_path.setPlaceholderText("Optional lens file override")
        self.lens_path.setReadOnly(True)
        self.browse_lens_button = QPushButton("Browse Lens…", self)
        self.browse_lens_button.clicked.connect(self._browse_lens)
        self.clear_lens_button = QPushButton("Use Auto-Fit", self)
        self.clear_lens_button.clicked.connect(self._clear_lens)
        lens_row.addWidget(QLabel("Lens:", self))
        lens_row.addWidget(self.lens_path, 1)
        lens_row.addWidget(self.browse_lens_button)
        lens_row.addWidget(self.clear_lens_button)
        layout.addLayout(lens_row)

        self.action_label = QLabel("", self)
        self.action_label.setWordWrap(True)
        layout.addWidget(self.action_label)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, self)
        self.open_button = buttons.addButton(
            "Load Model", QDialogButtonBox.ButtonRole.AcceptRole
        )
        self.open_button.clicked.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._populate()
        if models:
            self.table.selectRow(0)
        else:
            self.open_button.setEnabled(False)
            self.action_label.setText(
                "No Hugging Face model snapshots were found in the local cache. "
                "Start with --model and --allow-download to fetch one."
            )

    def selected_model_id(self) -> str | None:
        row = self._selected_row()
        if row is None:
            return None
        return self.models[row].model_id

    def selected_lens_path(self) -> Path | None:
        if self._auto_fit_requested:
            return None
        if self._manual_lens_path is not None:
            return self._manual_lens_path
        row = self._selected_row()
        if row is None:
            return None
        model = self.models[row]
        if model.lens_state is LensState.STABLE:
            return model.lens_path
        return None

    def _populate(self) -> None:
        for row, model in enumerate(self.models):
            values = (
                model.model_id,
                self._lens_label(model),
                str(model.cache_path),
                model.lens_detail,
            )
            for col, value in enumerate(values):
                item = QTableWidgetItem(value)
                if col == 1 and model.lens_state is LensState.STABLE:
                    item.setData(Qt.ItemDataRole.UserRole, "stable")
                self.table.setItem(row, col, item)
        self.table.resizeColumnsToContents()

    def _selected_row(self) -> int | None:
        selected = self.table.selectionModel().selectedRows()
        if not selected:
            return None
        return selected[0].row()

    def _selection_changed(self) -> None:
        self._manual_lens_path = None
        self._auto_fit_requested = False
        row = self._selected_row()
        if row is None:
            self.open_button.setEnabled(False)
            self.action_label.setText("Select a cached model.")
            self.lens_path.clear()
            return
        self.open_button.setEnabled(True)
        model = self.models[row]
        if model.lens_state is LensState.STABLE and model.lens_path is not None:
            self.lens_path.setText(str(model.lens_path))
            self.action_label.setText(
                "Stable lens found. J Studio will load this model with the existing "
                "calibrated J-space lens."
            )
        else:
            self.lens_path.clear()
            self.action_label.setText(
                "No usable Stable lens found. J Studio will load the cached model and "
                "generate a Stable lens in the background."
            )

    def _browse_lens(self) -> None:
        path, _filter = QFileDialog.getOpenFileName(
            self,
            "Choose Jacobian lens",
            str(Path.home()),
            "Lens files (*.pt *.pth);;All files (*)",
        )
        if path:
            self._auto_fit_requested = False
            self._manual_lens_path = Path(path)
            self.lens_path.setText(path)
            self.action_label.setText(
                "Manual lens selected. J Studio will validate it against the model "
                "before arming J-space interventions."
            )

    def _clear_lens(self) -> None:
        self._manual_lens_path = None
        self._auto_fit_requested = True
        row = self._selected_row()
        if row is not None:
            self.lens_path.clear()
            self.action_label.setText(
                "J Studio will generate or resume a Stable lens after loading."
            )

    @staticmethod
    def _lens_label(model: CachedModel) -> str:
        if model.lens_state is LensState.STABLE:
            return "Stable"
        if model.lens_state is LensState.NEEDS_FIT:
            return "Needs fit"
        if model.lens_state is LensState.UNREADABLE:
            return "Unreadable"
        return "Missing"


def choose_startup_model(parent: QWidget | None = None) -> tuple[str, Path | None] | None:
    dialog = StartupModelDialog(scan_hf_cache(), parent)
    if dialog.exec() != QDialog.DialogCode.Accepted:
        return None
    model_id = dialog.selected_model_id()
    if model_id is None:
        return None
    return model_id, dialog.selected_lens_path()
