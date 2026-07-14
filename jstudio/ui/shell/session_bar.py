"""Permanent active-model identity strip."""

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QHBoxLayout, QLabel, QToolButton

from jstudio.domain import LensFitState, LensFitStatus, ModelSessionSummary


class SessionBar(QFrame):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("sessionBar")
        self.setProperty("role", "session")
        self.setFixedHeight(46)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(10, 5, 10, 5)
        layout.setSpacing(8)
        self.select_button = QToolButton(self)
        self.select_button.setText("J")
        self.select_button.setProperty("role", "primary")
        self.select_button.setToolTip("Select a Model Session (Ctrl+K)")
        self.select_button.setAccessibleName("Select model session")
        self.identity = QLabel("No Model Selected", self)
        self.identity.setProperty("role", "heading")
        self.identity.setAlignment(Qt.AlignmentFlag.AlignVCenter)
        self.identity.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.backend_badge = QLabel("Disconnected", self)
        self.status = QLabel("Disconnected", self)
        self.lens_status = QLabel("Lens unavailable", self)
        for badge in (self.backend_badge, self.status, self.lens_status):
            badge.setProperty("role", "statusPill")
        self.overflow = QToolButton(self)
        self.overflow.setText("⋮")
        self.overflow.setProperty("role", "ghost")
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
