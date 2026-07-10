from PySide6.QtCore import Qt

from jstudio.domain import (
    ConceptActivation,
    InterventionDraft,
    InterventionEntry,
    InterventionOperation,
    RuleRecord,
    RuleTrigger,
)
from jstudio.ui.models import ActivationTableModel, InterventionTableModel, RuleTableModel


def test_activation_model_exposes_rows_and_stable_ids():
    model = ActivationTableModel()
    rows = (
        ConceptActivation("injection", 0.91, 0.94, 42, 7, previous_score=0.78),
        ConceptActivation("warning", -0.38, 0.74, 39, 7, previous_score=-0.21),
    )

    model.replace_rows(rows)

    assert model.rowCount() == 2
    assert model.columnCount() == 3
    assert model.data(model.index(0, 0), Qt.ItemDataRole.DisplayRole) == "injection"
    assert model.data(model.index(0, 1), Qt.ItemDataRole.DisplayRole) == "+0.91"
    assert model.data(model.index(0, 0), Qt.ItemDataRole.UserRole) == rows[0]


def test_activation_model_updates_one_score_without_reset(qtbot):
    model = ActivationTableModel()
    model.replace_rows((ConceptActivation("term", 0.1, 0.8, 3, 2),))
    with qtbot.waitSignal(model.dataChanged):
        model.update_activation(ConceptActivation("term", 0.5, 0.9, 4, 3))
    assert model.data(model.index(0, 1)) == "+0.50"


def test_intervention_enabled_checkbox_is_editable(qtbot):
    entry = InterventionEntry.from_draft(
        InterventionDraft(
            InterventionOperation.INJECT, None, "cat", 0.5, 1, 2
        )
    )
    model = InterventionTableModel((entry,))

    with qtbot.waitSignal(model.enabled_changed) as changed:
        accepted = model.setData(
            model.index(0, 0), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole
        )

    assert accepted
    assert changed.args == [0, True]
    assert model.record(0).enabled


def test_rule_enabled_checkbox_is_editable(qtbot):
    rule = RuleRecord(
        "rule-1",
        "Guard",
        "function run(ctx) { return []; }",
        RuleTrigger.BEFORE_TOKEN,
    )
    model = RuleTableModel((rule,))

    accepted = model.setData(
        model.index(0, 0), Qt.CheckState.Checked, Qt.ItemDataRole.CheckStateRole
    )

    assert accepted
    assert model.record(0).enabled
    assert (
        model.data(model.index(0, 0), Qt.ItemDataRole.CheckStateRole)
        == Qt.CheckState.Checked
    )
