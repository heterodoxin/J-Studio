"""Compact activity status surface."""

from PySide6.QtWidgets import QLabel, QToolButton, QWidget


class ActivityIndicator(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.label = QLabel("Ready", self)
        self.button = QToolButton(self)
        self.button.setText("Activity")
        self.button.setToolTip("Show background activity")
        self.button.setAccessibleName("Show background activity")
