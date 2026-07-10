"""Focused secondary research tools opened from J Studio's Tools menu."""

from __future__ import annotations

from PySide6.QtCore import QPointF, Qt
from PySide6.QtGui import QColor, QPainter, QPen
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QMainWindow,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableView,
    QTabWidget,
    QToolBar,
    QVBoxLayout,
    QWidget,
)

from jstudio.ui.models import ExperimentRunModel, TraceEventModel


class _ToolWindow(QMainWindow):
    def __init__(self, title: str, parent=None) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(860, 620)
        self.setAttribute(Qt.WidgetAttribute.WA_DeleteOnClose, False)


class ModelViewWindow(_ToolWindow):
    def __init__(self, parent=None) -> None:
        super().__init__("Model View", parent)
        self.menuBar().addMenu("File")
        self.menuBar().addMenu("Search")
        self.menuBar().addMenu("View")
        self.menuBar().addMenu("Generate")
        self.response = QPlainTextEdit(self)
        self.response.setReadOnly(True)
        self.response.setPlainText("No response selected. Generate or select a prior run.")
        self.setCentralWidget(self.response)


class LayerExplorerWindow(_ToolWindow):
    def __init__(self, parent=None) -> None:
        super().__init__("Layer Explorer", parent)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        self.provenance = QLabel(
            "Run — · Model — · Revision — · Lens — · Layer — · Position —", central
        )
        self.details = QPlainTextEdit(central)
        self.details.setReadOnly(True)
        self.details.setPlainText("Select a layer/position cell in J-Lens.")
        actions = QHBoxLayout()
        self.actions = {}
        for label in (
            "Inject at Selection",
            "Replace at Selection",
            "Suppress at Selection",
            "Trace Influence",
            "Copy Coordinates",
        ):
            button = QPushButton(label, central)
            self.actions[label] = button
            actions.addWidget(button)
        layout.addWidget(self.provenance)
        layout.addWidget(self.details, 1)
        layout.addLayout(actions)
        self.setCentralWidget(central)


class InfluenceGraph(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("influenceGraph")
        self.setMinimumSize(420, 300)
        self.setAccessibleName("Estimated concept influence graph")
        self.terms = ["No", "trace", "selected"]

    def set_terms(self, terms) -> None:
        self.terms = list(terms)[:4] or ["No selection"]
        self.update()

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.fillRect(self.rect(), self.palette().base())
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        coordinates = ((0.15, 0.50), (0.45, 0.25), (0.48, 0.70), (0.80, 0.48))
        nodes = tuple((*coordinates[index], term) for index, term in enumerate(self.terms))
        painter.setPen(QPen(self.palette().mid().color(), 2))
        for start, stop in ((index, index + 1) for index in range(len(nodes) - 1)):
            a, b = nodes[start], nodes[stop]
            painter.drawLine(
                QPointF(a[0] * self.width(), a[1] * self.height()),
                QPointF(b[0] * self.width(), b[1] * self.height()),
            )
        for x, y, label in nodes:
            point = QPointF(x * self.width(), y * self.height())
            painter.setBrush(QColor("#2563eb"))
            painter.drawEllipse(point, 22, 22)
            painter.setPen(self.palette().text().color())
            painter.drawText(
                int(point.x() - 35),
                int(point.y() + 38),
                70,
                18,
                Qt.AlignmentFlag.AlignCenter,
                label,
            )


class InfluenceTraceWindow(_ToolWindow):
    def __init__(self, parent=None) -> None:
        super().__init__("Influence Trace", parent)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        setup = QGroupBox("Trace Setup", central)
        form = QFormLayout(setup)
        self.seed_term = QLineEdit("injection", setup)
        self.direction = QComboBox(setup)
        self.direction.addItems(["Forward", "Backward", "Both"])
        self.max_nodes = QSpinBox(setup)
        self.max_nodes.setRange(1, 10000)
        self.max_nodes.setValue(500)
        self.run_button = QPushButton("Run Trace", setup)
        form.addRow("Seed Term", self.seed_term)
        form.addRow("Direction", self.direction)
        form.addRow("Maximum Nodes", self.max_nodes)
        form.addRow(self.run_button)
        splitter = QSplitter(Qt.Orientation.Horizontal, central)
        self.graph = InfluenceGraph(splitter)
        self.results = QTableView(splitter)
        self.results.setModel(TraceEventModel(parent=self.results))
        splitter.addWidget(self.graph)
        splitter.addWidget(self.results)
        self.disclaimer = QLabel(
            "Links are estimated influence, not causation unless a causal "
            "intervention result is reported.",
            central,
        )
        layout.addWidget(setup)
        layout.addWidget(splitter, 1)
        layout.addWidget(self.disclaimer)
        self.setCentralWidget(central)


class GenerationTraceWindow(_ToolWindow):
    def __init__(self, parent=None) -> None:
        super().__init__("Generation Trace", parent)
        toolbar = QToolBar(self)
        self.actions = {}
        for label in (
            "Generate",
            "One Token",
            "Pause",
            "Resume",
            "Stop",
            "Baseline",
            "With Stack",
            "Compare",
        ):
            self.actions[label] = toolbar.addAction(label)
        self.addToolBar(toolbar)
        table = QTableView(self)
        table.setModel(TraceEventModel(parent=table))
        self.setCentralWidget(table)


class AdvancedSweepWindow(_ToolWindow):
    def __init__(self, parent=None) -> None:
        super().__init__("Advanced J-Lens Sweep", parent)
        central = QWidget(self)
        layout = QHBoxLayout(central)
        left = QTabWidget(central)
        left.addTab(QPlainTextEdit("Activation timeline"), "Sweep 1")
        setup = QGroupBox("Sweep Setup", central)
        form = QFormLayout(setup)
        self.prompt = QPlainTextEdit(setup)
        self.run_button = QPushButton("Run Sweep", setup)
        form.addRow("Prompt", self.prompt)
        form.addRow("Readout Position", QComboBox(setup))
        form.addRow("Minimum |score|", QLineEdit("0.10", setup))
        form.addRow(self.run_button)
        layout.addWidget(left, 1)
        layout.addWidget(setup)
        self.setCentralWidget(central)


class ExperimentsWindow(_ToolWindow):
    def __init__(self, parent=None) -> None:
        super().__init__("Experiments", parent)
        self.tabs = QTabWidget(self)
        setup = QWidget(self.tabs)
        setup_layout = QFormLayout(setup)
        setup_layout.addRow("Session", QComboBox(setup))
        setup_layout.addRow("Prompt Set", QComboBox(setup))
        setup_layout.addRow("Run Matrix", QPlainTextEdit(setup))
        self.run_button = QPushButton("Run Baseline + Stack", setup)
        setup_layout.addRow(self.run_button)
        runs = QTableView(self.tabs)
        runs.setModel(ExperimentRunModel(parent=runs))
        compare = QPlainTextEdit(self.tabs)
        compare.setReadOnly(True)
        compare.setPlainText("Aligned baseline/intervention outputs and activation deltas")
        report = QPlainTextEdit(self.tabs)
        report.setPlainText("Reports preserve exact provenance and aggregation method.")
        for title, widget in (
            ("Setup", setup),
            ("Runs", runs),
            ("Compare", compare),
            ("Report", report),
        ):
            self.tabs.addTab(widget, title)
        self.setCentralWidget(self.tabs)


class SnapshotManagerWindow(_ToolWindow):
    def __init__(self, parent=None) -> None:
        super().__init__("Snapshot Manager", parent)
        central = QWidget(self)
        layout = QVBoxLayout(central)
        layout.addWidget(QLabel("Captured immutable rule and J-lens snapshots"))
        self.snapshots = QListWidget(central)
        self.capture_button = QPushButton("Capture Current Frame", central)
        layout.addWidget(self.snapshots, 1)
        layout.addWidget(self.capture_button)
        self.setCentralWidget(central)


TOOL_CLASSES = {
    "model_view": ModelViewWindow,
    "layer_explorer": LayerExplorerWindow,
    "jlens_sweep": AdvancedSweepWindow,
    "influence_trace": InfluenceTraceWindow,
    "generation_trace": GenerationTraceWindow,
    "experiments": ExperimentsWindow,
    "snapshot_manager": SnapshotManagerWindow,
}
