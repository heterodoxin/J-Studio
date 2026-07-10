"""Virtualized Qt item models used throughout J Studio."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from PySide6.QtCore import QAbstractTableModel, QModelIndex, Qt, Signal

from jstudio.domain import (
    ConceptActivation,
    ExperimentRecord,
    InterventionEntry,
    ModelSessionSummary,
    RuleRecord,
)


class ActivationTableModel(QAbstractTableModel):
    HEADERS = ("Term", "Score", "Previous")

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._rows: list[ConceptActivation] = []

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or not 0 <= index.row() < len(self._rows):
            return None
        activation = self._rows[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return activation
        if role == Qt.ItemDataRole.AccessibleTextRole:
            return (
                f"{activation.term}, score {activation.score:+.2f}, "
                f"layer {activation.layer}, token {activation.token_index}"
            )
        if role == Qt.ItemDataRole.TextAlignmentRole and index.column() > 0:
            return int(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        if index.column() == 0:
            return activation.term
        if index.column() == 1:
            return f"{activation.score:+.2f}"
        return (
            "" if activation.previous_score is None else f"{activation.previous_score:+.2f}"
        )

    def replace_rows(self, rows: Sequence[ConceptActivation]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def update_activation(self, activation: ConceptActivation) -> None:
        for row, current in enumerate(self._rows):
            if current.term == activation.term:
                self._rows[row] = activation
                self.dataChanged.emit(
                    self.index(row, 0),
                    self.index(row, self.columnCount() - 1),
                    [Qt.ItemDataRole.DisplayRole, Qt.ItemDataRole.UserRole],
                )
                return
        row = len(self._rows)
        self.beginInsertRows(QModelIndex(), row, row)
        self._rows.append(activation)
        self.endInsertRows()

    def activation(self, row: int) -> ConceptActivation:
        return self._rows[row]


class _RecordTableModel(QAbstractTableModel):
    HEADERS: tuple[str, ...] = ()

    def __init__(self, rows: Sequence[Any] = (), parent=None) -> None:
        super().__init__(parent)
        self._rows = list(rows)

    def rowCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self._rows)

    def columnCount(self, parent: QModelIndex | None = None) -> int:
        return 0 if parent is not None and parent.isValid() else len(self.HEADERS)

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.HEADERS[section]
        return None

    def replace_rows(self, rows: Sequence[Any]) -> None:
        self.beginResetModel()
        self._rows = list(rows)
        self.endResetModel()

    def record(self, row: int):
        return self._rows[row]


class InterventionTableModel(_RecordTableModel):
    enabled_changed = Signal(int, bool)
    HEADERS = (
        "Enabled",
        "Operation",
        "Match",
        "Result",
        "Max Budget",
        "Layers",
        "Duration",
        "Trigger",
        "Status",
    )

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and index.column() == 0:
            flags |= Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or index.column() != 0:
            return False
        if role != Qt.ItemDataRole.CheckStateRole:
            return False
        from dataclasses import replace

        enabled = value == Qt.CheckState.Checked
        self._rows[index.row()] = replace(self._rows[index.row()], enabled=enabled)
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
        self.enabled_changed.emit(index.row(), enabled)
        return True

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        entry: InterventionEntry = self._rows[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return entry
        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 0:
            return Qt.CheckState.Checked if entry.enabled else Qt.CheckState.Unchecked
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        draft = entry.draft
        values = (
            "",
            draft.operation.value.title(),
            draft.source_term or "",
            draft.target_term or "",
            f"{draft.strength:.2f}",
            f"{draft.layer_start}–{draft.layer_end}",
            draft.duration,
            draft.trigger,
            entry.status_detail,
        )
        return values[index.column()]


class SessionTableModel(_RecordTableModel):
    HEADERS = ("Model", "Revision", "Backend", "Device", "Precision", "Lens Status")

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        session: ModelSessionSummary = self._rows[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return session
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        values = (
            session.display_name or session.model_id,
            session.revision,
            session.backend_kind.value,
            session.device,
            session.precision,
            "Compatible" if session.lens_id else "Missing",
        )
        return values[index.column()]


class RuleTableModel(_RecordTableModel):
    enabled_changed = Signal(int, bool)
    HEADERS = ("Enabled", "Name", "Trigger", "Priority", "Last Result", "Failures")

    def flags(self, index):
        flags = super().flags(index)
        if index.isValid() and index.column() == 0:
            flags |= Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEditable
        return flags

    def setData(self, index, value, role=Qt.ItemDataRole.EditRole):
        if not index.isValid() or index.column() != 0:
            return False
        if role != Qt.ItemDataRole.CheckStateRole:
            return False
        from dataclasses import replace

        enabled = value == Qt.CheckState.Checked
        self._rows[index.row()] = replace(self._rows[index.row()], enabled=enabled)
        self.dataChanged.emit(index, index, [Qt.ItemDataRole.CheckStateRole])
        self.enabled_changed.emit(index.row(), enabled)
        return True

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return None
        rule: RuleRecord = self._rows[index.row()]
        if role == Qt.ItemDataRole.UserRole:
            return rule
        if role == Qt.ItemDataRole.CheckStateRole and index.column() == 0:
            return Qt.CheckState.Checked if rule.enabled else Qt.CheckState.Unchecked
        if role != Qt.ItemDataRole.DisplayRole:
            return None
        values = (
            "",
            rule.name,
            rule.trigger.value,
            str(rule.priority),
            rule.last_result,
            str(rule.consecutive_failures),
        )
        return values[index.column()]


class TraceEventModel(_RecordTableModel):
    HEADERS = ("Step", "Type", "Token", "Layer", "Detail")

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        event = self._rows[index.row()]
        return str(event.get(self.HEADERS[index.column()].lower(), ""))


class ExperimentRunModel(_RecordTableModel):
    HEADERS = ("Run", "Prompt", "Mode", "Stack", "Rules", "Status", "Duration", "Output")

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or role != Qt.ItemDataRole.DisplayRole:
            return None
        record: ExperimentRecord | dict = self._rows[index.row()]
        if isinstance(record, dict):
            return str(record.get(self.HEADERS[index.column()].lower(), ""))
        return record.name if index.column() == 0 else ""
