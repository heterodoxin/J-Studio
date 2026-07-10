import json

from jstudio.domain import (
    InterventionDraft,
    InterventionEntry,
    InterventionOperation,
    InterventionState,
    RuleRecord,
    RuleTrigger,
)
from jstudio.project import ProjectDocument


def test_imported_project_starts_safe(qtbot, window, tmp_path):
    project = ProjectDocument.new("Imported")
    project.interventions.append(
        InterventionEntry(
            intervention_id="i1",
            draft=InterventionDraft(
                InterventionOperation.INJECT,
                None,
                "caution",
                0.6,
                1,
                5,
            ),
            label="Caution",
            enabled=True,
            state=InterventionState.ARMED,
        )
    )
    project.rules.append(
        RuleRecord(
            "r1",
            "Rule",
            "function run(ctx) { return []; }",
            RuleTrigger.JSPACE_FRAME,
            enabled=True,
            trusted=True,
        )
    )
    path = tmp_path / "imported.jstudio.json"
    path.write_text(json.dumps(project.to_dict()))

    window.open_project(path, imported=True)

    assert all(not row.enabled for row in window.project.interventions)
    assert all(not rule.enabled and not rule.trusted for rule in window.project.rules)
    assert window.main_workspace.intervention_model.rowCount() == 1


def test_save_project_updates_title_and_clears_dirty(window, tmp_path):
    window.project.dirty = True
    path = tmp_path / "saved.jstudio.json"

    window.save_project(path)

    assert path.exists()
    assert not window.project.dirty
    assert "saved.jstudio.json" in window.windowTitle()
