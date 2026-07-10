import json
from dataclasses import replace

import pytest

from jstudio.domain import (
    GenerationBackend,
    InterventionDraft,
    InterventionEntry,
    InterventionOperation,
    InterventionState,
    RuleRecord,
    RuleTrigger,
    RunMode,
    RunRecord,
)
from jstudio.project import ProjectDocument, ProjectFormatError


def _imported_payload():
    draft = InterventionDraft(
        operation=InterventionOperation.REPLACE,
        source_term="injection",
        target_term="trusted",
        strength=0.8,
        layer_start=18,
        layer_end=26,
    )
    project = ProjectDocument.new("Imported")
    project.interventions.append(
        InterventionEntry(
            intervention_id="i1",
            draft=draft,
            label="Guard",
            enabled=True,
            state=InterventionState.ARMED,
        )
    )
    project.rules.append(
        RuleRecord(
            rule_id="r1",
            name="Guard rule",
            source="function run(ctx) { return []; }",
            trigger=RuleTrigger.JSPACE_FRAME,
            enabled=True,
            trusted=True,
        )
    )
    return project.to_dict()


def test_import_disarms_interventions_and_distrusts_rules(tmp_path):
    path = tmp_path / "imported.jstudio.json"
    path.write_text(json.dumps(_imported_payload()))

    project = ProjectDocument.load(path, imported=True)

    assert all(
        not entry.enabled and entry.state is InterventionState.DRAFT
        for entry in project.interventions
    )
    assert all(not rule.enabled and not rule.trusted for rule in project.rules)


def test_project_round_trip_is_atomic(tmp_path):
    project = ProjectDocument.new("Research")
    project.prompts.extend(["one", "two"])
    path = tmp_path / "research.jstudio.json"

    project.save(path)
    loaded = ProjectDocument.load(path)

    assert loaded.name == "Research"
    assert loaded.prompts == ["one", "two"]
    assert not path.with_suffix(path.suffix + ".tmp").exists()


def test_project_round_trips_generation_provenance(tmp_path):
    project = ProjectDocument.new("Timing")
    project.runs.append(
        replace(
            RunRecord.create(prompt="hello", mode=RunMode.BASELINE),
            generation_backend=GenerationBackend.FAST,
            quantization="Q4_K_M",
            ttft_seconds=0.2,
            decode_tokens_per_second=72.5,
        )
    )
    path = tmp_path / "timing.jstudio.json"
    project.save(path)

    loaded = ProjectDocument.load(path)

    assert loaded.runs[0].generation_backend is GenerationBackend.FAST
    assert loaded.runs[0].quantization == "Q4_K_M"
    assert loaded.runs[0].ttft_seconds == 0.2
    assert loaded.runs[0].decode_tokens_per_second == 72.5


def test_project_rejects_secrets_unknown_schema_and_nonfinite_numbers():
    with pytest.raises(ProjectFormatError):
        ProjectDocument.from_dict({"schema": 99, "access_token": "secret"})
    with pytest.raises(ProjectFormatError, match="secret"):
        ProjectDocument.from_dict(
            {"schema": 1, "name": "x", "credentials": {"token": "secret"}}
        )
    with pytest.raises(ProjectFormatError, match="finite"):
        ProjectDocument.from_json('{"schema":1,"name":"x","layout":{"zoom":NaN}}')
