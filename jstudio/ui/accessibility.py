"""Accessibility and display-density utilities."""

from PySide6.QtCore import QCoreApplication
from PySide6.QtGui import QFont


def tr(text: str) -> str:
    return QCoreApplication.translate("JStudio", text)


def apply_text_scale(application, percent: int) -> None:
    if not 90 <= percent <= 160:
        raise ValueError("text scale must lie between 90 and 160 percent")
    font = QFont(application.font())
    base = font.pointSizeF() if font.pointSizeF() > 0 else 10.0
    font.setPointSizeF(base * percent / 100)
    application.setFont(font)


def ensure_accessible_button(button, name: str, tooltip: str) -> None:
    button.setAccessibleName(name)
    button.setToolTip(tooltip)
