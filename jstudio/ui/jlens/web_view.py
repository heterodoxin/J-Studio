"""Secure host and native action bridge for the original J-Lens HTML page."""

from __future__ import annotations

import os
from pathlib import Path

from PySide6.QtCore import QObject, QUrl, Signal, Slot
from PySide6.QtWebChannel import QWebChannel
from PySide6.QtWebEngineCore import QWebEnginePage
from PySide6.QtWebEngineWidgets import QWebEngineView

from jstudio.services.protocols import SlicePage


class SecureSlicePage(QWebEnginePage):
    _ALLOWED_SCHEMES = {"about", "qrc"}

    def __init__(self, trusted_root: Path, parent=None) -> None:
        super().__init__(parent)
        self.trusted_root = trusted_root.resolve()

    def allows(self, url: QUrl) -> bool:
        scheme = url.scheme().lower()
        if scheme in self._ALLOWED_SCHEMES:
            return True
        if not url.isLocalFile():
            return False
        try:
            Path(url.toLocalFile()).resolve().relative_to(self.trusted_root)
        except (OSError, ValueError):
            return False
        return True

    def acceptNavigationRequest(self, url, navigation_type, is_main_frame):
        return self.allows(url)


class JLensBridge(QObject):
    coordinate_selected = Signal(int, int)
    term_pinned = Signal(str)
    intervention_requested = Signal(str, int, int)

    @Slot(int, int)
    def select(self, position: int, layer: int) -> None:
        if position >= 0 and layer >= 0:
            self.coordinate_selected.emit(position, layer)

    @Slot(str)
    def pin(self, term: str) -> None:
        if term.strip():
            self.term_pinned.emit(term.strip())

    @Slot(str, int, int)
    def intervene(self, term: str, layer: int, position: int) -> None:
        if term.strip() and layer >= 0 and position >= 0:
            self.intervention_requested.emit(term.strip(), layer, position)


class JLensWebView(QWebEngineView):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Interactive J-Lens slice")
        self._page_root = (
            Path.home() / ".cache" / "jstudio" / "slices" / str(os.getpid())
        )
        self._page_root.mkdir(parents=True, exist_ok=True)
        self.setPage(SecureSlicePage(self._page_root, self))
        self.bridge = JLensBridge(self)
        self.channel = QWebChannel(self.page())
        self.channel.registerObject("jstudioBridge", self.bridge)
        self.page().setWebChannel(self.channel)
        self.current_generation = 0
        self.last_html = ""

    def set_page(self, page: SlicePage) -> None:
        self.current_generation = page.generation
        self.last_html = page.html
        page_path = self._page_root / f"slice-{page.generation}.html"
        page_path.write_text(page.html, encoding="utf-8")
        self.setUrl(QUrl.fromLocalFile(str(page_path)))

    def closeEvent(self, event) -> None:
        page = self.page()
        page.setWebChannel(None)
        page.triggerAction(QWebEnginePage.WebAction.Stop)
        super().closeEvent(event)
        page.deleteLater()
