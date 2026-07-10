"""Custom-painted rank plots."""

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import QWidget


class RankPlot(QWidget):
    def __init__(self, axis: str, parent=None) -> None:
        super().__init__(parent)
        self.axis = axis
        self.crosshair = 0
        self._frames = ()
        self._terms = ()
        self.setMinimumHeight(120)
        self.setAccessibleName(f"Pinned term rank by {axis}")

    def set_data(self, frames, terms, crosshair: int) -> None:
        self._frames = tuple(frames)
        self._terms = tuple(terms)
        self.crosshair = crosshair
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setPen(self.palette().mid().color())
        painter.drawRect(self.rect().adjusted(0, 0, -1, -1))
        if not self._frames or not self._terms:
            painter.drawText(self.rect(), Qt.AlignmentFlag.AlignCenter, "No pinned terms")
            return
        colors = (QColor("#d000d0"), QColor("#2563eb"), QColor("#15803d"))
        count = max(2, len(self._frames))
        for term_index, term in enumerate(self._terms):
            points = QPolygonF()
            for index, frame in enumerate(self._frames):
                activation = next((a for a in frame.activations if a.term == term), None)
                rank = (
                    activation.rank if activation and activation.rank is not None else 100
                )
                x = index * (self.width() - 1) / (count - 1)
                y = min(1.0, rank / 100) * (self.height() - 1)
                points.append(QPointF(x, y))
            painter.setPen(QPen(colors[term_index % len(colors)], 2))
            painter.drawPolyline(points)
        denominator = max(1, (len(self._frames) - 1) if self.axis == "position" else 63)
        x = min(self.width() - 1, max(0, self.crosshair / denominator * self.width()))
        painter.setPen(QPen(self.palette().text().color(), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(x), 0, int(x), self.height())
