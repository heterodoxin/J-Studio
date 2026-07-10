"""Modeless validated J-space intervention editor."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from jstudio.domain import InterventionDraft, InterventionOperation, ModelSessionSummary


class InterventionEditor(QDialog):
    draft_added = Signal(object)
    preview_requested = Signal(object)

    def __init__(
        self,
        session: ModelSessionSummary,
        *,
        operation: str = "inject",
        term: str = "",
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.session = session
        self.setWindowTitle("Add / Edit J-Space Intervention")
        self.setModal(False)
        self.resize(620, 620)
        root = QVBoxLayout(self)
        form_group = QGroupBox("Intervention", self)
        form = QFormLayout(form_group)
        self.operation = QComboBox(form_group)
        self.operation.addItems(["Inject", "Replace", "Suppress"])
        self.source_term = QLineEdit(form_group)
        self.target_term = QLineEdit(form_group)
        self.match_mode = QComboBox(form_group)
        self.match_mode.addItems(
            ["Exact Term", "Case-Insensitive", "Regular Expression", "Concept ID"]
        )
        self.strength = QDoubleSpinBox(form_group)
        self.strength.setRange(
            session.capabilities.strength_min, session.capabilities.strength_max
        )
        self.strength.setDecimals(3)
        self.strength.setSingleStep(0.05)
        self.strength.setValue(session.capabilities.strength_max)
        self.strength.setToolTip(
            "Maximum search budget. J Studio applies the minimum effective "
            "residual/J-space scale found within this limit."
        )
        layer_widget = QWidget(form_group)
        layer_layout = QHBoxLayout(layer_widget)
        layer_layout.setContentsMargins(0, 0, 0, 0)
        self.layer_start = QSpinBox(layer_widget)
        self.layer_end = QSpinBox(layer_widget)
        self.layer_start.setRange(0, session.layer_count - 1)
        self.layer_end.setRange(0, session.layer_count - 1)
        self.layer_end.setValue(session.layer_count - 1)
        layer_layout.addWidget(self.layer_start)
        layer_layout.addWidget(QLabel("to"))
        layer_layout.addWidget(self.layer_end)
        self.duration = QComboBox(form_group)
        self.duration.addItems(
            ["Current Token", "Next Token", "N Steps", "Entire Generation"]
        )
        self.duration.setCurrentText("Next Token")
        self.step_count = QSpinBox(form_group)
        self.step_count.setRange(1, 10000)
        self.step_count.setValue(1)
        self.trigger = QComboBox(form_group)
        self.trigger.addItems(["Manual", "Before Token", "After Match", "Rule"])
        form.addRow("Operation", self.operation)
        self.source_label = QLabel("Match Term")
        self.target_label = QLabel("Target Term")
        form.addRow(self.source_label, self.source_term)
        form.addRow(self.target_label, self.target_term)
        form.addRow("Match mode", self.match_mode)
        form.addRow("Max strength budget", self.strength)
        form.addRow("Layer scope", layer_widget)
        form.addRow("Duration", self.duration)
        form.addRow("Step count", self.step_count)
        form.addRow("Trigger", self.trigger)
        root.addWidget(form_group)

        self.preview = QLabel(self)
        self.preview.setWordWrap(True)
        self.preview.setFrameShape(QLabel.Shape.StyledPanel)
        self.preview.setMinimumHeight(72)
        self.error_label = QLabel(self)
        self.error_label.setStyleSheet(
            "color: palette(bright-text); background: #9b1c1c; padding: 4px"
        )
        self.error_label.hide()
        root.addWidget(QLabel("Preview"))
        root.addWidget(self.preview)
        root.addWidget(self.error_label)
        root.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Cancel, parent=self)
        self.save_button = buttons.addButton(
            "Save Draft", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.preview_button = buttons.addButton(
            "Preview", QDialogButtonBox.ButtonRole.ActionRole
        )
        self.add_button = buttons.addButton(
            "Add to Intervention List", QDialogButtonBox.ButtonRole.AcceptRole
        )
        root.addWidget(buttons)
        buttons.rejected.connect(self.close)
        self.add_button.clicked.connect(self._add)
        self.save_button.clicked.connect(self._add)
        self.preview_button.clicked.connect(self._preview)
        self.operation.currentTextChanged.connect(self._sync_fields)
        self.duration.currentTextChanged.connect(self._sync_fields)
        for widget in (
            self.source_term,
            self.target_term,
            self.strength,
            self.layer_start,
            self.layer_end,
            self.trigger,
        ):
            if hasattr(widget, "textChanged"):
                widget.textChanged.connect(self._update_preview)
            if hasattr(widget, "valueChanged"):
                widget.valueChanged.connect(self._update_preview)
            if hasattr(widget, "currentTextChanged"):
                widget.currentTextChanged.connect(self._update_preview)

        self.operation.setCurrentText(operation.title())
        op = InterventionOperation(operation.lower())
        if op is InterventionOperation.INJECT:
            self.target_term.setText(term)
        else:
            self.source_term.setText(term)
        self._sync_fields()

    def _sync_fields(self) -> None:
        operation = InterventionOperation(self.operation.currentText().lower())
        inject = operation is InterventionOperation.INJECT
        suppress = operation is InterventionOperation.SUPPRESS
        self.source_label.setVisible(not inject)
        self.source_term.setVisible(not inject)
        self.target_label.setVisible(not suppress)
        self.target_term.setVisible(not suppress)
        self.target_label.setText("Injected Term" if inject else "Replacement Term")
        self.match_mode.setEnabled(not inject)
        self.step_count.setVisible(self.duration.currentText() == "N Steps")
        self._update_preview()

    def _draft(self) -> InterventionDraft:
        operation = InterventionOperation(self.operation.currentText().lower())
        duration_map = {
            "Current Token": "current-token",
            "Next Token": "next-token",
            "N Steps": "steps",
            "Entire Generation": "generation",
        }
        return InterventionDraft(
            operation=operation,
            source_term=self.source_term.text().strip() or None,
            target_term=self.target_term.text().strip() or None,
            strength=self.strength.value(),
            layer_start=self.layer_start.value(),
            layer_end=self.layer_end.value(),
            duration=duration_map[self.duration.currentText()],
            step_count=self.step_count.value()
            if self.duration.currentText() == "N Steps"
            else None,
            match_mode=self.match_mode.currentText().lower().replace(" ", "-"),
            trigger=self.trigger.currentText().lower().replace(" ", "-"),
        )

    def _validate(self) -> InterventionDraft | None:
        self.error_label.hide()
        try:
            draft = self._draft()
        except ValueError as exc:
            self.error_label.setText(str(exc))
            self.error_label.show()
            message = str(exc).lower()
            self.activateWindow()
            if "target" in message:
                self.target_term.setFocus(Qt.FocusReason.OtherFocusReason)
            elif "source" in message:
                self.source_term.setFocus(Qt.FocusReason.OtherFocusReason)
            elif "layer" in message:
                self.layer_start.setFocus(Qt.FocusReason.OtherFocusReason)
            return None
        if self.match_mode.currentText() == "Regular Expression":
            import re

            try:
                re.compile(draft.source_term or "")
            except re.error as exc:
                self.error_label.setText(f"Invalid regular expression: {exc}")
                self.error_label.show()
                self.source_term.setFocus()
                return None
            if len(draft.source_term or "") > 256:
                self.error_label.setText("Regular expression is too complex")
                self.error_label.show()
                return None
        return draft

    def _add(self) -> None:
        draft = self._validate()
        if draft is None:
            return
        self.draft_added.emit(draft)
        self.close()

    def _preview(self) -> None:
        draft = self._validate()
        if draft is not None:
            self.preview_requested.emit(draft)

    def _update_preview(self) -> None:
        operation = self.operation.currentText()
        source = self.source_term.text().strip()
        target = self.target_term.text().strip()
        if operation == "Replace":
            phrase = f"Replace {source or '…'} with {target or '…'}"
        elif operation == "Inject":
            phrase = f"Inject {target or '…'}"
        else:
            phrase = f"Suppress {source or '…'}"
        self.preview.setText(
            f"{phrase}; auto-search minimum effective scale up to "
            f"{self.strength.value():.3g} in layers "
            f"{self.layer_start.value()}–{self.layer_end.value()}, "
            f"duration {self.duration.currentText().lower()}."
        )
