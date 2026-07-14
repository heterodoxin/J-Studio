"""Three-pane Rules workspace and isolated Test Bench."""

from __future__ import annotations

import hashlib
from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace

from PySide6.QtCore import QObject, Qt, Signal
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QMenu,
    QPlainTextEdit,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableView,
    QTabWidget,
    QToolBar,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from jstudio.domain import RuleRecord, RuleTrigger
from jstudio.project import ProjectDocument
from jstudio.rules.protocol import RuleEvaluationRequest, SandboxLimits
from jstudio.ui.models import RuleTableModel
from jstudio.ui.rules.editor import RuleSourceEditor

DEFAULT_SOURCE = """const TERM = "cat";

function run(ctx) {
  if (ctx.jspace.has(TERM, {minScore: 0.65})) {
    return [jspace.inject(TERM, {
      layers: "current",
      duration: "next-token"
    })];
  }
  return [];
}"""

_API_REFERENCE = """Rule source pattern
Define constants in your rule source. There is no separate config tab.

const TERM = "cat";

function run(ctx) {
  if (ctx.jspace.has(TERM, {minScore: 0.65})) {
    return [jspace.inject(TERM, {
      layers: "current",
      duration: "next-token"
    })];
  }
  return [];
}

Context
ctx.event.type              current trigger name
ctx.event.sequence          event sequence number
ctx.model.id                model id
ctx.model.revision          model revision
ctx.model.layerCount        layer count
ctx.lens.id                 active lens id
ctx.layer.index             current layer, when applicable
ctx.token.index/text        current token, when applicable
ctx.generation.step         generation step
ctx.generation.prompt       original prompt
ctx.generation.outputText   generated text so far
ctx.jspace.activations      [{term, score, layer, position, rank}]
ctx.stack.entries           active intervention stack
ctx.tags                    rule-created tags

J-space helpers
ctx.jspace.has(term, {minScore})
ctx.jspace.score(term)
ctx.jspace.top(n)
ctx.jspace.find(pattern)

Actions
jspace.inject(term, {layers, duration, strength, label})
jspace.replace(source, target, {layers, duration, strength, matchMode, label})
jspace.suppress(term, {layers, duration, strength, label})

strength is optional. It is a maximum search budget, not the applied scale.
If omitted, J Studio searches up to 16.0 and applies the minimum passing
residual/J-space scale found by the intervention engine.
generation.stop({reason})
rule.log(message)
rule.tag(name, value)

Layers
"current"
"all"
{from: 12, to: 24}

Duration
"next-token"
"read"
{steps: 8}

Limits
50 ms wall time · 25 ms QuickJS · 16 MiB heap · 32 actions
"""


class _ResultBridge(QObject):
    completed = Signal(object)


class RulesWorkspace(QWidget):
    def __init__(self, sandbox, project: ProjectDocument, parent=None) -> None:
        super().__init__(parent)
        self.sandbox = sandbox
        self.project = project
        self.current_rule_id: str | None = None
        self.last_test_passed = False
        self._executor = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="jstudio-rules-ui"
        )
        self._bridge = _ResultBridge(self)
        self._bridge.completed.connect(self._test_completed)
        root = QVBoxLayout(self)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(8)
        toolbar = QToolBar(self)
        toolbar.setObjectName("rulesToolbar")
        self.new_action = toolbar.addAction("New Rule")
        self.save_action = toolbar.addAction("Save")
        self.enable_action = QAction("Enable", self)
        toolbar.addAction(self.enable_action)
        self.more_button = QToolButton(toolbar)
        self.more_button.setText("More  ⋮")
        self.more_button.setPopupMode(QToolButton.ToolButtonPopupMode.InstantPopup)
        self.more_button.setAccessibleName("More rule actions")
        more_menu = QMenu(self.more_button)
        for label in (
            "New Folder",
            "Disable",
            "Duplicate",
            "Export",
            "Import",
            "Delete",
        ):
            more_menu.addAction(label)
        self.more_button.setMenu(more_menu)
        toolbar.addWidget(self.more_button)
        root.addWidget(toolbar)
        self.main_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self.rule_list_panel = self._build_rule_list()
        self.main_splitter.addWidget(self.rule_list_panel)
        self.editor = RuleSourceEditor(self.main_splitter)
        self.editor.setObjectName("ruleSourceEditor")
        self.editor.setProperty("role", "data")
        self.editor.setAccessibleName("JavaScript rule source")
        self.main_splitter.addWidget(self.editor)
        self.side_panel = self._build_side_panel()
        self.main_splitter.addWidget(self.side_panel)
        self.main_splitter.setSizes([260, 520, 340])
        root.addWidget(self.main_splitter, 1)
        self.output_tabs = QTabWidget(self)
        self.output_tabs.setProperty("role", "subtabs")
        self.problems = QPlainTextEdit(self.output_tabs)
        self.returned_actions = QPlainTextEdit(self.output_tabs)
        self.execution_log = QPlainTextEdit(self.output_tabs)
        for title, widget in (
            ("Problems", self.problems),
            ("Returned Actions", self.returned_actions),
            ("Execution Log", self.execution_log),
        ):
            widget.setReadOnly(True)
            self.output_tabs.addTab(widget, title)
        self.output_tabs.setFixedHeight(160)
        root.addWidget(self.output_tabs)
        self.new_action.triggered.connect(lambda: self.new_rule("New Rule"))
        self.save_action.triggered.connect(self.save_current)
        self.enable_action.triggered.connect(self.enable_current)
        self.editor.textChanged.connect(self._source_changed)
        self.trigger.currentIndexChanged.connect(self._source_changed)
        self.priority.valueChanged.connect(self._source_changed)
        self.editor.test_requested.connect(self.test_current)
        self.test_button.clicked.connect(self.test_current)
        self.rule_model.enabled_changed.connect(self._rule_enabled_changed)
        self.rule_table.selectionModel().selectionChanged.connect(self._selection_changed)
        self.enable_action.setEnabled(False)

    def _build_rule_list(self) -> QWidget:
        widget = QWidget(self.main_splitter)
        widget.setProperty("role", "panel")
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        self.rule_model = RuleTableModel(self.project.rules, widget)
        self.rule_table = QTableView(widget)
        self.rule_table.setModel(self.rule_model)
        self.rule_table.setSelectionBehavior(QTableView.SelectionBehavior.SelectRows)
        self.rule_table.setSelectionMode(QTableView.SelectionMode.SingleSelection)
        self.rule_table.setColumnHidden(2, True)
        self.rule_table.setColumnHidden(3, True)
        self.test_button = QPushButton("Test Rule", widget)
        self.test_button.setProperty("role", "primary")
        layout.addWidget(self.rule_table, 1)
        layout.addWidget(self.test_button)
        return widget

    def _build_side_panel(self) -> QWidget:
        panel = QWidget(self.main_splitter)
        panel.setProperty("role", "panel")
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)
        settings = QGroupBox("Rule Settings", panel)
        form = QFormLayout(settings)
        self.trigger = QComboBox(settings)
        for trigger in RuleTrigger:
            self.trigger.addItem(trigger.value, trigger)
        self.priority = QSpinBox(settings)
        self.priority.setRange(-100000, 100000)
        self.priority.setValue(100)
        self.sandbox_status = QLabel(
            "Ready" if self.sandbox.available else self.sandbox.unavailable_reason,
            settings,
        )
        form.addRow("Trigger", self.trigger)
        form.addRow("Priority", self.priority)
        form.addRow("Sandbox", self.sandbox_status)
        layout.addWidget(settings)

        tabs = QTabWidget(panel)
        tabs.setProperty("role", "subtabs")
        self.api_reference = QPlainTextEdit(tabs)
        self.api_reference.setReadOnly(True)
        self.api_reference.setPlainText(_API_REFERENCE)
        test = QWidget(tabs)
        test_layout = QVBoxLayout(test)
        self.snapshot = QComboBox(test)
        self.snapshot.addItems(["Captured prompt-injection frame", "Empty frame"])
        self.test_summary = QLabel("Testing never applies actions.", test)
        test_layout.addWidget(QLabel("Immutable snapshot"))
        test_layout.addWidget(self.snapshot)
        test_layout.addWidget(self.test_summary)
        test_layout.addStretch(1)
        tabs.addTab(self.api_reference, "API")
        tabs.addTab(test, "Test")
        layout.addWidget(tabs, 1)
        self.side_tabs = tabs
        return panel

    def _hash(self, source: str | None = None) -> str:
        source = self.editor.toPlainText() if source is None else source
        trigger = RuleTrigger(self.trigger.currentData()).value
        value = f"{source}\0{trigger}\0{self.priority.value()}"
        return hashlib.sha256(value.encode()).hexdigest()

    def _config(self) -> dict:
        return {}

    def new_rule(self, name: str) -> RuleRecord:
        return self.add_rule(
            name=name,
            source=DEFAULT_SOURCE,
            trigger=RuleTrigger.JSPACE_FRAME,
            enabled=False,
        )

    def add_rule(
        self,
        *,
        name: str,
        source: str,
        trigger: RuleTrigger,
        enabled: bool = False,
    ) -> RuleRecord:
        from uuid import uuid4

        rule = RuleRecord(
            rule_id=f"rule-{uuid4().hex}",
            name=name,
            source=source,
            trigger=trigger,
            enabled=enabled,
            trusted=True,
        )
        self.project.rules.append(rule)
        self.project.dirty = True
        self.rule_model.replace_rows(self.project.rules)
        row = len(self.project.rules) - 1
        self.rule_table.selectRow(row)
        self._load_rule(rule)
        return rule

    def rule(self, rule_id: str) -> RuleRecord:
        return next(rule for rule in self.project.rules if rule.rule_id == rule_id)

    def _replace_rule(self, updated: RuleRecord) -> None:
        self.project.rules[:] = [
            updated if rule.rule_id == updated.rule_id else rule
            for rule in self.project.rules
        ]
        self.rule_model.replace_rows(self.project.rules)
        self.project.dirty = True

    def _rule_enabled_changed(self, row: int, enabled: bool) -> None:
        self.project.rules[row] = replace(self.project.rules[row], enabled=enabled)
        self.project.dirty = True

    def _selection_changed(self) -> None:
        rows = self.rule_table.selectionModel().selectedRows()
        if rows:
            self._load_rule(self.rule_model.record(rows[0].row()))

    def _load_rule(self, rule: RuleRecord) -> None:
        self.current_rule_id = rule.rule_id
        self.editor.blockSignals(True)
        self.trigger.blockSignals(True)
        self.priority.blockSignals(True)
        self.editor.setPlainText(rule.source)
        self.trigger.setCurrentIndex(max(0, self.trigger.findData(rule.trigger.value)))
        self.priority.setValue(rule.priority)
        self.priority.blockSignals(False)
        self.trigger.blockSignals(False)
        self.editor.blockSignals(False)
        self.last_test_passed = bool(rule.tested_hash and rule.tested_hash == self._hash())
        self.enable_action.setEnabled(self.last_test_passed and rule.trusted)

    def _source_changed(self) -> None:
        self.last_test_passed = False
        self.enable_action.setEnabled(False)

    def save_current(self) -> None:
        if self.current_rule_id is None:
            return
        rule = self.rule(self.current_rule_id)
        updated = replace(
            rule,
            source=self.editor.toPlainText(),
            trigger=RuleTrigger(self.trigger.currentData()),
            priority=self.priority.value(),
            source_hash=self._hash(),
            config={},
        )
        self._replace_rule(updated)
        self.project.dirty = True

    def test_current(self) -> None:
        if self.current_rule_id is None or not self.sandbox.available:
            return
        self.save_current()
        self.last_test_passed = False
        self.enable_action.setEnabled(False)
        self.test_button.setEnabled(False)
        self.test_summary.setText("Testing in isolated worker…")
        request = RuleEvaluationRequest(
            source=self.editor.toPlainText(),
            trigger=RuleTrigger(self.trigger.currentData()).value,
            context={
                "event": {
                    "type": RuleTrigger(self.trigger.currentData()).value,
                    "sequence": 1,
                },
                "model": {"id": "qwen", "revision": "main", "layerCount": 64},
                "lens": {"id": "jlens"},
                "layer": {"index": 42},
                "token": {"index": 7, "text": " result"},
                "generation": {"step": 7, "outputText": "bounded"},
                "jspace": {
                    "activations": [
                        {"term": "injection", "score": 0.91},
                        {"term": "warning", "score": 0.55},
                    ]
                },
                "stack": {"entries": []},
                "tags": {},
                "config": self._config(),
            },
            layer_count=64,
            limits=SandboxLimits(),
        )
        future = self._executor.submit(self.sandbox.evaluate, request)
        future.add_done_callback(
            lambda completed: self._bridge.completed.emit(completed.result())
        )

    def _test_completed(self, result) -> None:
        self.test_button.setEnabled(True)
        self.problems.setPlainText(result.error or "No problems")
        self.returned_actions.setPlainText(
            "\n".join(
                f"{index + 1}. {action.kind}: {action.payload}"
                for index, action in enumerate(result.actions)
            )
            or "No actions"
        )
        self.execution_log.setPlainText(
            f"Success: {result.success}\n"
            f"Wall: {result.metrics.wall_ms:.2f} ms\n"
            f"QuickJS: {result.metrics.execution_ms:.2f} ms\n"
            f"Worker memory: {result.metrics.peak_worker_bytes} bytes\n"
            "Input/output: "
            f"{result.metrics.input_bytes}/{result.metrics.output_bytes} bytes"
        )
        self.test_summary.setText(
            "Test passed — actions remain unapplied"
            if result.success
            else "Test failed — no actions applied"
        )
        if self.current_rule_id is None:
            return
        rule = self.rule(self.current_rule_id)
        tested_hash = self._hash() if result.success else None
        updated = replace(
            rule,
            source=self.editor.toPlainText(),
            source_hash=self._hash(),
            tested_hash=tested_hash,
            last_result="Passed" if result.success else result.error,
            consecutive_failures=0 if result.success else rule.consecutive_failures + 1,
        )
        self._replace_rule(updated)
        self.last_test_passed = result.success
        self.enable_action.setEnabled(result.success and updated.trusted)

    def enable_current(self) -> None:
        if self.current_rule_id is None or not self.last_test_passed:
            return
        rule = self.rule(self.current_rule_id)
        self._replace_rule(replace(rule, enabled=True))

    def record_execution_failure(self, rule_id: str, detail: str) -> None:
        rule = self.rule(rule_id)
        failures = rule.consecutive_failures + 1
        self._replace_rule(
            replace(
                rule,
                consecutive_failures=failures,
                enabled=False if failures >= 3 else rule.enabled,
                last_result=(
                    f"Auto-disabled after {failures} failures: {detail}"
                    if failures >= 3
                    else detail
                ),
            )
        )

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False, cancel_futures=True)
