"""System palettes and the shared semantic J Studio workbench theme."""

from PySide6.QtGui import QColor, QPalette

_TOKENS = {
    "dark": {
        "window": "#111318",
        "panel": "#191c23",
        "panel_alt": "#20242d",
        "data": "#0d1015",
        "border": "#303642",
        "border_strong": "#454d5c",
        "text": "#f4f1fb",
        "muted": "#9da5b4",
        "accent": "#8b5cf6",
        "accent_hover": "#9f7aea",
        "accent_soft": "#2b2144",
        "danger": "#ef6b73",
        "success": "#4ade80",
        "warning": "#fbbf24",
    },
    "light": {
        "window": "#eef0f5",
        "panel": "#ffffff",
        "panel_alt": "#f7f7fb",
        "data": "#ffffff",
        "border": "#d5d8e2",
        "border_strong": "#b8becc",
        "text": "#17151d",
        "muted": "#687083",
        "accent": "#7c3aed",
        "accent_hover": "#6d28d9",
        "accent_soft": "#ede9fe",
        "danger": "#c2414c",
        "success": "#15803d",
        "warning": "#b45309",
    },
}


def _stylesheet(tokens: dict[str, str]) -> str:
    return f"""
QWidget {{
    color: {tokens['text']};
    font-size: 13px;
}}
QMainWindow, QDialog, QWidget#applicationRoot {{
    background: {tokens['window']};
}}
QWidget[role="panel"], QFrame[role="panel"] {{
    background: {tokens['panel']};
    border: 1px solid {tokens['border']};
    border-radius: 9px;
}}
QWidget[role="data"], QFrame[role="data"] {{
    background: {tokens['data']};
    border: 1px solid {tokens['border']};
    border-radius: 7px;
}}
QLabel[role="heading"] {{
    color: {tokens['text']};
    font-size: 15px;
    font-weight: 650;
}}
QLabel[role="muted"] {{ color: {tokens['muted']}; }}
QLabel[role="statusPill"] {{
    background: {tokens['panel_alt']};
    border: 1px solid {tokens['border']};
    border-radius: 9px;
    color: {tokens['muted']};
    padding: 2px 8px;
}}
QLabel[status="success"] {{ color: {tokens['success']}; }}
QLabel[status="warning"] {{ color: {tokens['warning']}; }}
QFrame#sessionBar {{
    background: {tokens['panel']};
    border: 0;
    border-bottom: 1px solid {tokens['border']};
}}
QPushButton, QToolButton {{
    background: {tokens['panel_alt']};
    border: 1px solid {tokens['border_strong']};
    border-radius: 6px;
    min-height: 30px;
    padding: 0 11px;
}}
QPushButton:hover, QToolButton:hover {{
    background: {tokens['accent_soft']};
    border-color: {tokens['accent']};
}}
QPushButton:pressed, QToolButton:pressed {{ background: {tokens['data']}; }}
QPushButton:disabled, QToolButton:disabled {{
    color: {tokens['muted']};
    background: {tokens['panel']};
    border-color: {tokens['border']};
}}
QPushButton[role="primary"], QToolButton[role="primary"] {{
    color: #ffffff;
    background: {tokens['accent']};
    border-color: {tokens['accent']};
    font-weight: 650;
}}
QPushButton[role="primary"]:hover, QToolButton[role="primary"]:hover {{
    background: {tokens['accent_hover']};
}}
QPushButton[role="danger"] {{ color: {tokens['danger']}; }}
QPushButton[operation="inject"] {{
    color: #c4b5fd;
    border-color: #6d4ac7;
}}
QPushButton[operation="replace"] {{
    color: #fb923c;
    border-color: #9a5428;
}}
QPushButton[operation="suppress"] {{
    color: {tokens['danger']};
    border-color: #8f4148;
}}
QPushButton[role="ghost"], QToolButton[role="ghost"] {{
    background: transparent;
    border-color: transparent;
}}
QLineEdit, QPlainTextEdit, QTextEdit, QComboBox, QSpinBox {{
    background: {tokens['data']};
    border: 1px solid {tokens['border']};
    border-radius: 6px;
    selection-background-color: {tokens['accent']};
    padding: 5px 8px;
}}
QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QComboBox:focus, QSpinBox:focus {{ border: 1px solid {tokens['accent']}; }}
QComboBox::drop-down {{ border: 0; width: 24px; }}
QTableView, QListView, QTreeView {{
    background: {tokens['data']};
    alternate-background-color: {tokens['panel']};
    border: 1px solid {tokens['border']};
    border-radius: 7px;
    gridline-color: {tokens['border']};
    selection-background-color: {tokens['accent_soft']};
    selection-color: {tokens['text']};
}}
QHeaderView::section {{
    background: {tokens['panel_alt']};
    border: 0;
    border-right: 1px solid {tokens['border']};
    border-bottom: 1px solid {tokens['border']};
    color: {tokens['muted']};
    font-weight: 600;
    padding: 6px 8px;
}}
QTabWidget#workspaceTabs::pane {{ border: 0; background: {tokens['window']}; }}
QTabWidget#workspaceTabs > QTabBar::tab {{
    background: transparent;
    border: 0;
    color: {tokens['muted']};
    min-width: 78px;
    min-height: 34px;
    padding: 0 12px;
}}
QTabWidget#workspaceTabs > QTabBar::tab:selected {{
    color: {tokens['text']};
    border-bottom: 2px solid {tokens['accent']};
    font-weight: 650;
}}
QTabWidget#workspaceTabs > QTabBar::tab:hover {{ color: {tokens['text']}; }}
QTabWidget[role="subtabs"]::pane {{
    border: 1px solid {tokens['border']};
    border-radius: 7px;
    background: {tokens['panel']};
}}
QTabWidget[role="subtabs"] > QTabBar::tab {{
    background: transparent;
    border: 0;
    color: {tokens['muted']};
    padding: 7px 12px;
}}
QTabWidget[role="subtabs"] > QTabBar::tab:selected {{
    color: {tokens['text']};
    border-bottom: 2px solid {tokens['accent']};
}}
QGroupBox {{
    border: 1px solid {tokens['border']};
    border-radius: 7px;
    margin-top: 10px;
    padding: 10px 8px 8px 8px;
    font-weight: 600;
}}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px; }}
QSplitter::handle {{ background: transparent; }}
QSplitter::handle:hover {{ background: {tokens['accent_soft']}; }}
QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
QScrollBar::handle:vertical {{
    background: {tokens['border_strong']}; border-radius: 4px; min-height: 24px;
}}
QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
QScrollBar::handle:horizontal {{
    background: {tokens['border_strong']}; border-radius: 4px; min-width: 24px;
}}
QScrollBar::add-line, QScrollBar::sub-line {{ width: 0; height: 0; }}
QMenuBar {{ background: {tokens['panel']}; border-bottom: 1px solid {tokens['border']}; }}
QMenuBar::item {{ padding: 5px 9px; background: transparent; }}
QMenuBar::item:selected, QMenu::item:selected {{ background: {tokens['accent_soft']}; }}
QMenu {{
    background: {tokens['panel']};
    border: 1px solid {tokens['border']};
    padding: 5px;
}}
QMenu::item {{ padding: 6px 24px 6px 10px; border-radius: 4px; }}
QStatusBar {{
    background: {tokens['panel']};
    color: {tokens['muted']};
    border-top: 1px solid {tokens['border']};
}}
QToolTip {{
    background: {tokens['panel_alt']};
    color: {tokens['text']};
    border: 1px solid {tokens['border_strong']};
    padding: 5px;
}}
"""


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


def apply_jstudio_theme(application, mode: str = "dark") -> None:
    """Install the coherent workbench palette and semantic Qt stylesheet."""
    from PySide6.QtWidgets import QStyleFactory

    if mode not in {"dark", "light", "system"}:
        raise ValueError("theme mode must be dark, light, or system")
    style = QStyleFactory.create("Fusion")
    if style is not None:
        application.setStyle(style)
    if mode == "light":
        apply_light_palette(application)
        tokens = _TOKENS["light"]
    elif mode == "system":
        apply_system_palette(application)
        dark = application.palette().color(QPalette.ColorRole.Window).lightness() < 128
        tokens = _TOKENS["dark" if dark else "light"]
    else:
        apply_dark_palette(application)
        tokens = _TOKENS["dark"]
    application.setStyleSheet(_stylesheet(tokens))
