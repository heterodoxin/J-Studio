"""System, light, and dark palette helpers."""

from PySide6.QtGui import QColor, QPalette


def _palette(window: str, panel: str, text: str, secondary: str) -> QPalette:
    palette = QPalette()
    palette.setColor(QPalette.ColorRole.Window, QColor(window))
    palette.setColor(QPalette.ColorRole.WindowText, QColor(text))
    palette.setColor(QPalette.ColorRole.Base, QColor(panel))
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(window))
    palette.setColor(QPalette.ColorRole.Text, QColor(text))
    palette.setColor(QPalette.ColorRole.Button, QColor(panel))
    palette.setColor(QPalette.ColorRole.ButtonText, QColor(text))
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(panel))
    palette.setColor(QPalette.ColorRole.ToolTipText, QColor(text))
    palette.setColor(QPalette.ColorRole.PlaceholderText, QColor(secondary))
    palette.setColor(QPalette.ColorRole.Highlight, QColor("#2563eb"))
    palette.setColor(QPalette.ColorRole.HighlightedText, QColor("#ffffff"))
    palette.setColor(QPalette.ColorRole.Link, QColor("#2563eb"))
    palette.setColor(
        QPalette.ColorGroup.Disabled, QPalette.ColorRole.Text, QColor(secondary)
    )
    return palette


def apply_light_palette(application) -> None:
    application.setPalette(_palette("#f4f6f8", "#ffffff", "#17202a", "#5f6b78"))


def apply_dark_palette(application) -> None:
    application.setPalette(_palette("#171a1f", "#20242b", "#f2f4f7", "#aab2bd"))


def apply_system_palette(application) -> None:
    from PySide6.QtWidgets import QStyleFactory

    style = QStyleFactory.create(application.style().objectName())
    if style is not None:
        application.setPalette(style.standardPalette())
