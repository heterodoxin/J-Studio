"""Custom-painted layer × position views."""

from __future__ import annotations

from PySide6.QtCore import QPoint, QRect, QSize, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QAbstractScrollArea, QToolTip, QWidget


class JLensMatrixView(QAbstractScrollArea):
    cell_selected = Signal(int, int)
    term_pinned = Signal(str)
    intervention_requested = Signal(str, int, int)

    CELL_W = 76
    CELL_H = 23
    HEADER_W = 34
    HEADER_H = 22

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("layerPositionMatrix")
        self.setMouseTracking(True)
        self._frames = ()
        self._cells = {}
        self._positions: list[int] = []
        self._layers: list[int] = []
        self.selected = (0, 0)
        self.setAccessibleName("Layer by position J-lens matrix")
        self.setMinimumSize(380, 250)

    def set_frames(self, frames) -> None:
        self._frames = tuple(frames)
        self._positions = sorted({frame.token_index for frame in frames})
        layer_count = max((frame.layer_count for frame in frames), default=1)
        self._layers = list(range(layer_count - 1, -1, -1))
        self._cells = {}
        for frame in frames:
            per_layer = {}
            for activation in frame.activations:
                current = per_layer.get(activation.layer)
                if current is None or activation.rank < current.rank:
                    per_layer[activation.layer] = activation
            for layer, activation in per_layer.items():
                self._cells[(frame.token_index, layer)] = activation
        self.horizontalScrollBar().setRange(
            0,
            max(
                0,
                self.HEADER_W
                + len(self._positions) * self.CELL_W
                - self.viewport().width(),
            ),
        )
        self.verticalScrollBar().setRange(
            0,
            max(
                0,
                self.HEADER_H + len(self._layers) * self.CELL_H - self.viewport().height(),
            ),
        )
        self.viewport().update()

    def sizeHint(self) -> QSize:
        return QSize(650, 360)

    def set_selected(self, position: int, layer: int) -> None:
        self.selected = (position, layer)
        self.viewport().update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self.viewport())
        painter.fillRect(self.viewport().rect(), self.palette().base())
        x_offset = self.horizontalScrollBar().value()
        y_offset = self.verticalScrollBar().value()
        font = QFont(self.font())
        font.setPointSizeF(max(8.0, font.pointSizeF() - 1))
        painter.setFont(font)
        visible = self.viewport().rect()
        for column, position in enumerate(self._positions):
            x = self.HEADER_W + column * self.CELL_W - x_offset
            if x + self.CELL_W < 0 or x > visible.right():
                continue
            painter.setPen(self.palette().text().color())
            painter.drawText(
                QRect(x, 0, self.CELL_W, self.HEADER_H),
                Qt.AlignmentFlag.AlignCenter,
                str(position),
            )
        for row, layer in enumerate(self._layers):
            y = self.HEADER_H + row * self.CELL_H - y_offset
            if y + self.CELL_H < 0 or y > visible.bottom():
                continue
            painter.drawText(
                QRect(0, y, self.HEADER_W, self.CELL_H),
                Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter,
                str(layer),
            )
            for column, position in enumerate(self._positions):
                x = self.HEADER_W + column * self.CELL_W - x_offset
                if x + self.CELL_W < 0 or x > visible.right():
                    continue
                rect = QRect(x, y, self.CELL_W, self.CELL_H)
                activation = self._cells.get((position, layer))
                painter.setPen(self.palette().mid().color())
                painter.drawRect(rect.adjusted(0, 0, -1, -1))
                if activation is not None:
                    strength = min(abs(activation.score), 1.0)
                    color = QColor(37, 99, 235, int(28 + strength * 90))
                    painter.fillRect(rect.adjusted(1, 1, -1, -1), color)
                    painter.setPen(self.palette().text().color())
                    painter.drawText(
                        rect.adjusted(3, 0, -3, 0),
                        Qt.AlignmentFlag.AlignVCenter,
                        f"{activation.term}  "
                        f"{activation.rank if activation.rank is not None else ''}",
                    )
                if (position, layer) == self.selected:
                    painter.setPen(QPen(QColor("#d000d0"), 2))
                    painter.drawRect(rect.adjusted(1, 1, -2, -2))

    def _cell_at(self, point: QPoint):
        x = point.x() + self.horizontalScrollBar().value() - self.HEADER_W
        y = point.y() + self.verticalScrollBar().value() - self.HEADER_H
        if x < 0 or y < 0:
            return None
        column, row = x // self.CELL_W, y // self.CELL_H
        if column >= len(self._positions) or row >= len(self._layers):
            return None
        return self._positions[column], self._layers[row]

    def mousePressEvent(self, event) -> None:
        cell = self._cell_at(event.position().toPoint())
        if cell is None:
            return
        position, layer = cell
        self.cell_selected.emit(position, layer)
        activation = self._cells.get(cell)
        if event.modifiers() & Qt.KeyboardModifier.ShiftModifier and activation:
            self.term_pinned.emit(activation.term)

    def mouseDoubleClickEvent(self, event) -> None:
        cell = self._cell_at(event.position().toPoint())
        activation = self._cells.get(cell) if cell else None
        if cell and activation:
            self.intervention_requested.emit(activation.term, cell[1], cell[0])

    def mouseMoveEvent(self, event) -> None:
        cell = self._cell_at(event.position().toPoint())
        activation = self._cells.get(cell) if cell else None
        if activation is None:
            QToolTip.hideText()
            return
        QToolTip.showText(
            event.globalPosition().toPoint(),
            f"{activation.term}\nScore {activation.score:+.3f} · rank {activation.rank}\n"
            f"Layer {activation.layer} · position {activation.token_index}\n"
            f"Source {activation.source.value}",
            self,
        )


class PinnedHeatmap(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("pinnedTermHeatmap")
        self._frames = ()
        self._terms = ()
        self._selected = (0, 0)
        self.setMinimumHeight(130)
        self.setAccessibleName("Pinned term rank heatmap")

    def set_data(self, frames, terms, selected) -> None:
        self._frames = tuple(frames)
        self._terms = tuple(terms)
        self._selected = selected
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        if not self._frames:
            painter.drawText(
                self.rect(), Qt.AlignmentFlag.AlignCenter, "Pin terms to show ranks"
            )
            return
        positions = sorted({frame.token_index for frame in self._frames})
        layers = max(frame.layer_count for frame in self._frames)
        cell_w = max(2, self.width() // max(1, len(positions)))
        cell_h = max(1, self.height() // max(1, layers))
        for p_index, position in enumerate(positions):
            frame = next(frame for frame in self._frames if frame.token_index == position)
            ranks = {
                a.term: (a.rank if a.rank is not None else 100) for a in frame.activations
            }
            for layer in range(layers):
                if self._terms:
                    rank = min(ranks.get(term, 100) for term in self._terms)
                    value = max(0, 255 - min(rank, 100) * 2)
                    color = QColor(208, 40 + value // 3, 208, 80 + value // 2)
                else:
                    color = self.palette().alternateBase().color()
                painter.fillRect(
                    p_index * cell_w,
                    self.height() - (layer + 1) * cell_h,
                    cell_w,
                    cell_h,
                    color,
                )
