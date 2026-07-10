"""Permanent active-model identity strip."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton

from jstudio.domain import LensFitState, LensFitStatus, ModelSessionSummary


class SessionBar(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sessionBar")
        self.setFixedHeight(36)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(4, 2, 4, 2)
        layout.setSpacing(6)
        self.select_button = QToolButton(self)
        self.select_button.setText("▣")
        self.select_button.setToolTip("Select a Model Session (Ctrl+K)")
        self.select_button.setAccessibleName("Select model session")
        self.identity = QLabel("No Model Selected", self)
        self.identity.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.identity.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.backend_badge = QLabel("Disconnected", self)
        self.backend_badge.setFrameShape(QFrame.Shape.StyledPanel)
        self.status = QLabel("Disconnected", self)
        self.lens_status = QLabel("Lens unavailable", self)
        self.overflow = QToolButton(self)
        self.overflow.setText("⋮")
        self.overflow.setToolTip("Model session actions")
        self.overflow.setAccessibleName("Model session actions")
        self.overflow.setEnabled(False)
        layout.addWidget(self.select_button)
        layout.addWidget(self.identity, 1)
        layout.addWidget(self.backend_badge)
        layout.addWidget(self.status)
        layout.addWidget(self.lens_status)
        layout.addWidget(self.overflow)

    def set_session(self, session: ModelSessionSummary | None) -> None:
        if session is None:
            self.identity.setText("No Model Selected")
            self.backend_badge.setText("Disconnected")
            self.status.setText("Disconnected")
            self.lens_status.setText("Lens unavailable")
            self.overflow.setEnabled(False)
            return
        name = session.display_name or session.model_id
        lens = session.lens_id or "No compatible lens"
        self.identity.setText(
            f"{name}  ·  {session.revision}  ·  {lens}  ·  "
            f"{session.device}/{session.precision}"
        )
        self.backend_badge.setText(session.backend_kind.value.replace("-", " ").title())
        self.status.setText(session.state.value.title())
        self.overflow.setEnabled(True)

    def set_fit_status(self, status: LensFitStatus) -> None:
        if status.state in {LensFitState.MISSING, LensFitState.FAILED}:
            text = "Lens missing" if status.state is LensFitState.MISSING else "Lens failed"
        elif status.total:
            text = f"{status.stage} {status.completed}/{status.total}"
        else:
            text = status.stage or status.state.value.title()
        if status.quality == "passed":
            text += " · passed"
        self.lens_status.setText(text)
        self.lens_status.setToolTip(status.detail)
