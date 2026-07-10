"""Native controls around the repository's original J-Lens renderer."""

from __future__ import annotations

from dataclasses import replace

from PySide6.QtCore import QObject, QSize, Signal
from PySide6.QtWidgets import (
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from jstudio.domain import LensFitState, LensFitStatus
from jstudio.services.protocols import JStudioServices, SliceRequest
from jstudio.ui.jlens.web_view import JLensWebView


class JLensSelectionModel(QObject):
    changed = Signal(int, int)
    pins_changed = Signal(object)

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.run_id: str | None = None
        self.position = 0
        self.layer = 0
        self.pinned_terms: tuple[str, ...] = ()

    def set_run(self, run_id: str, *, position: int = 0, layer: int = 0) -> None:
        self.run_id = run_id
        self.select(position=position, layer=layer)

    def select(self, *, position: int, layer: int, record_history: bool = True) -> None:
        self.position, self.layer = max(0, position), max(0, layer)
        self.changed.emit(self.position, self.layer)

    def pin(self, term: str) -> None:
        if term and term not in self.pinned_terms:
            self.pinned_terms = (*self.pinned_terms, term)
            self.pins_changed.emit(self.pinned_terms)


class JLensWorkspace(QWidget):
    FULL_SLICE_TOKEN_LIMIT = 256
    intervention_requested = Signal(str, int, int)
    _slice_finished = Signal(int, object)

    def __init__(self, services: JStudioServices, parent=None) -> None:
        super().__init__(parent)
        self.services = services
        self.selection = JLensSelectionModel(self)
        self._request: SliceRequest | None = None
        self._request_serial = 0

        root = QVBoxLayout(self)
        root.setContentsMargins(4, 4, 4, 4)
        toolbar = QHBoxLayout()
        self.heading = QLabel("J-Lens", self)
        self.status = QLabel("No run selected", self)
        self.lens_badge = QLabel("No lens", self)
        self.lens_badge.setAccessibleName("Active lens and readout reliability")
        self.fit_status = QLabel("Lens status unavailable", self)
        self.fit_button = QPushButton("Start Fit", self)
        self.refresh_button = QPushButton("Refresh", self)
        self.refresh_button.setAccessibleName("Refresh J-Lens slice")
        self.export_button = QPushButton("Export", self)
        self.export_button.setAccessibleName("Export J-Lens slice")
        toolbar.addWidget(self.heading)
        toolbar.addWidget(self.status, 1)
        toolbar.addWidget(self.lens_badge)
        toolbar.addWidget(self.fit_status)
        toolbar.addWidget(self.fit_button)
        toolbar.addWidget(self.refresh_button)
        toolbar.addWidget(self.export_button)
        root.addLayout(toolbar)

        self.fit_panel = QWidget(self)
        fit_layout = QVBoxLayout(self.fit_panel)
        fit_layout.setContentsMargins(12, 8, 12, 8)
        self.fit_headline = QLabel("Fitting Jacobian lens", self.fit_panel)
        self.fit_headline.setStyleSheet("font-weight: 600;")
        self.fit_bar = QProgressBar(self.fit_panel)
        self.fit_bar.setTextVisible(True)
        self.fit_estimate = QLabel("", self.fit_panel)
        self.fit_estimate.setAccessibleName("Lens fit progress and time estimate")
        fit_layout.addWidget(self.fit_headline)
        fit_layout.addWidget(self.fit_bar)
        fit_layout.addWidget(self.fit_estimate)
        self.fit_panel.hide()
        root.addWidget(self.fit_panel)

        self.web = JLensWebView(self)
        root.addWidget(self.web, 1)
        self.refresh_button.clicked.connect(self.refresh)
        self.fit_button.clicked.connect(self._toggle_fit)
        self.web.bridge.coordinate_selected.connect(self._select)
        self.web.bridge.term_pinned.connect(self.selection.pin)
        self.web.bridge.intervention_requested.connect(self.intervention_requested)
        self._slice_finished.connect(self._accept_slice)
        self._fit_state = LensFitState.MISSING

    @property
    def run_id(self) -> str | None:
        return self.selection.run_id

    def sizeHint(self) -> QSize:
        return QSize(1200, 780)

    def inspect(
        self,
        run_id: str,
        text: str,
        title: str,
        *,
        position: int = 0,
    ) -> None:
        last_n_tokens = (
            self.FULL_SLICE_TOKEN_LIMIT
            if len(text.split()) > self.FULL_SLICE_TOKEN_LIMIT
            else None
        )
        self._request = SliceRequest(
            run_id,
            text,
            title,
            last_n_tokens=last_n_tokens,
        )
        self.selection.set_run(run_id, position=position)
        self.heading.setText(f"J-Lens — {title}")
        self.refresh()

    def set_run(self, run_id: str, frames, *, title: str = "J-Lens") -> None:
        text = "".join(frame.token_text for frame in frames) or "No captured text"
        position = frames[-1].token_index if frames else 0
        self.inspect(run_id, text, title, position=position)

    def stream_text(self, run_id: str, text: str) -> None:
        if self._request is None or self._request.run_id != run_id:
            return
        if not text or text == self._request.text:
            return
        self._request = replace(self._request, text=text)
        token_count = len(text.split())
        self.status.setText(
            f"Streaming source text · {token_count} tokens · refresh to rerender"
        )

    def refresh(self) -> None:
        if self._request is None:
            return
        self._request_serial += 1
        serial = self._request_serial
        self.status.setText("Loading slice…")
        future = self.services.lens.request_slice(self._request)
        future.add_done_callback(
            lambda completed: self._slice_finished.emit(serial, completed)
        )

    def _accept_slice(self, serial: int, future) -> None:
        if serial != self._request_serial:
            return
        try:
            page = future.result()
        except Exception as exc:
            self.status.setText(f"Slice failed: {exc}")
            return
        self.web.set_page(page)
        self.status.setText("Ready")

    def _select(self, position: int, layer: int) -> None:
        self.selection.select(position=position, layer=layer)

    def set_lens_identity(self, lens_id: str | None) -> None:
        if not lens_id:
            self.lens_badge.setText("No lens")
            self.lens_badge.setStyleSheet("")
            self.lens_badge.setToolTip("No compatible Jacobian lens is active.")
            return
        if lens_id.startswith("sketched"):
            self.lens_badge.setText(f"⚠ {lens_id}")
            self.lens_badge.setStyleSheet("color: #d97706; font-weight: 600;")
            self.lens_badge.setToolTip(
                "Sketched low-rank transport: readout ranks carry amplified "
                "projection noise and are unreliable. Interventions are "
                "unaffected. Fit a dense lens for trustworthy readout."
            )
        else:
            self.lens_badge.setText(lens_id)
            self.lens_badge.setStyleSheet("")
            self.lens_badge.setToolTip(
                "Dense Jacobian transport: readout ranks reflect the fitted "
                "average Jacobian."
            )

    def set_fit_status(self, status: LensFitStatus) -> None:
        self._fit_state = status.state
        progress = f" {status.completed}/{status.total}" if status.total else ""
        self.fit_status.setText(f"{status.stage or status.state.value.title()}{progress}")
        self.fit_status.setToolTip(status.detail)
        fitting = status.state in {
            LensFitState.WAITING,
            LensFitState.REFINING,
        }
        self.fit_button.setText("Cancel Fit" if fitting else "Resume Fit")
        self._update_fit_panel(status, fitting)
        viewable = status.state in {
            LensFitState.PREVIEW,
            LensFitState.STABLE,
            LensFitState.FAILED,
        }
        self.web.setEnabled(viewable)
        if (
            status.state in {LensFitState.PREVIEW, LensFitState.STABLE}
            and self._request is not None
        ):
            self.refresh()

    def _update_fit_panel(self, status: LensFitStatus, fitting: bool) -> None:
        if not fitting:
            self.fit_panel.hide()
            return
        self.fit_headline.setText(f"Fitting Jacobian lens · {status.stage or 'preparing'}")
        if status.total:
            self.fit_bar.setRange(0, status.total)
            self.fit_bar.setValue(status.completed)
        else:
            self.fit_bar.setRange(0, 0)  # indeterminate while preparing
        self.fit_estimate.setText(self._fit_estimate_text(status))
        self.fit_panel.show()

    @staticmethod
    def _fit_estimate_text(status: LensFitStatus) -> str:
        elapsed = int(status.elapsed_seconds)
        parts = [f"{status.completed}/{status.total} prompts"] if status.total else []
        parts.append(f"{elapsed // 60}m {elapsed % 60}s elapsed")
        if status.completed and status.total and status.elapsed_seconds > 0:
            remaining = status.elapsed_seconds / status.completed * (
                status.total - status.completed
            )
            parts.append(f"~{int(remaining) // 60}m {int(remaining) % 60}s remaining")
        return " · ".join(parts)

    def _toggle_fit(self) -> None:
        if self._fit_state in {LensFitState.WAITING, LensFitState.REFINING}:
            self.services.lens.cancel_fit()
        else:
            self.services.lens.start_fit()
