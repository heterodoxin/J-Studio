"""Versioned, secret-free J Studio project documents."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any
from uuid import uuid4

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
    RunState,
)


class ProjectFormatError(ValueError):
    pass


_TOP_LEVEL_KEYS = {
    "schema",
    "project_id",
    "name",
    "session",
    "prompts",
    "interventions",
    "rules",
    "runs",
    "experiments",
    "trace_references",
    "layout",
}
_SECRET_MARKERS = ("token", "password", "credential", "secret", "api_key")


def _reject_secrets(value: Any, path: str = "project") -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            lowered = str(key).lower()
            if any(marker in lowered for marker in _SECRET_MARKERS):
                raise ProjectFormatError(
                    f"secret-bearing field is not allowed: {path}.{key}"
                )
            _reject_secrets(child, f"{path}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _reject_secrets(child, f"{path}[{index}]")


def _reject_nonfinite(value: Any) -> None:
    if isinstance(value, float) and not math.isfinite(value):
        raise ProjectFormatError("project numbers must be finite")
    if isinstance(value, dict):
        for child in value.values():
            _reject_nonfinite(child)
    elif isinstance(value, list):
        for child in value:
            _reject_nonfinite(child)


def _draft_to_dict(draft: InterventionDraft) -> dict[str, Any]:
    return {
        "operation": draft.operation.value,
        "source_term": draft.source_term,
        "target_term": draft.target_term,
        "strength": draft.strength,
        "layer_start": draft.layer_start,
        "layer_end": draft.layer_end,
        "duration": draft.duration,
        "step_count": draft.step_count,
        "match_mode": draft.match_mode,
        "trigger": draft.trigger,
    }


def _entry_to_dict(entry: InterventionEntry) -> dict[str, Any]:
    return {
        "intervention_id": entry.intervention_id,
        "draft": _draft_to_dict(entry.draft),
        "label": entry.label,
        "enabled": entry.enabled,
        "state": entry.state.value,
        "status_detail": entry.status_detail,
        "applied_run_id": entry.applied_run_id,
    }


def _rule_to_dict(rule: RuleRecord) -> dict[str, Any]:
    return {
        "rule_id": rule.rule_id,
        "name": rule.name,
        "source": rule.source,
        "trigger": rule.trigger.value,
        "priority": rule.priority,
        "enabled": rule.enabled,
        "trusted": rule.trusted,
        "source_hash": rule.source_hash,
        "tested_hash": rule.tested_hash,
        "consecutive_failures": rule.consecutive_failures,
        "last_result": rule.last_result,
        "config": rule.config,
    }


def _run_to_dict(run: RunRecord) -> dict[str, Any]:
    return {
        "run_id": run.run_id,
        "prompt": run.prompt,
        "mode": run.mode.value,
        "state": run.state.value,
        "created_at": run.created_at,
        "baseline_run_id": run.baseline_run_id,
        "intervention_ids": list(run.intervention_ids),
        "rule_ids": list(run.rule_ids),
        "output_text": run.output_text,
        "inspection_text": run.inspection_text,
        "partial": run.partial,
        "generation_backend": run.generation_backend.value,
        "quantization": run.quantization,
        "ttft_seconds": run.ttft_seconds,
        "decode_rate": run.decode_tokens_per_second,
    }


@dataclass(slots=True)
class ProjectDocument:
    project_id: str
    name: str
    session: dict[str, Any] | None = None
    prompts: list[str] = field(default_factory=list)
    interventions: list[InterventionEntry] = field(default_factory=list)
    rules: list[RuleRecord] = field(default_factory=list)
    runs: list[RunRecord] = field(default_factory=list)
    experiments: list[dict[str, Any]] = field(default_factory=list)
    trace_references: list[str] = field(default_factory=list)
    layout: dict[str, Any] = field(default_factory=dict)
    dirty: bool = False
    path: Path | None = None

    @classmethod
    def new(cls, name: str = "Untitled") -> ProjectDocument:
        return cls(project_id=f"project-{uuid4().hex}", name=name)

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": 1,
            "project_id": self.project_id,
            "name": self.name,
            "session": self.session,
            "prompts": list(self.prompts),
            "interventions": [_entry_to_dict(entry) for entry in self.interventions],
            "rules": [_rule_to_dict(rule) for rule in self.rules],
            "runs": [_run_to_dict(run) for run in self.runs],
            "experiments": self.experiments,
            "trace_references": self.trace_references,
            "layout": self.layout,
        }

    @classmethod
    def from_json(cls, source: str, *, imported: bool = False) -> ProjectDocument:
        try:
            value = json.loads(
                source,
                parse_constant=lambda constant: (_ for _ in ()).throw(
                    ProjectFormatError(f"project numbers must be finite: {constant}")
                ),
            )
        except ProjectFormatError:
            raise
        except (json.JSONDecodeError, TypeError) as exc:
            raise ProjectFormatError(f"invalid project JSON: {exc}") from exc
        return cls.from_dict(value, imported=imported)

    @classmethod
    def from_dict(cls, value: dict[str, Any], *, imported: bool = False) -> ProjectDocument:
        if not isinstance(value, dict):
            raise ProjectFormatError("project root must be an object")
        _reject_secrets(value)
        _reject_nonfinite(value)
        unknown = set(value) - _TOP_LEVEL_KEYS
        if unknown:
            raise ProjectFormatError(f"unknown project fields: {sorted(unknown)}")
        if value.get("schema") != 1:
            raise ProjectFormatError(f"unsupported project schema: {value.get('schema')!r}")
        name = value.get("name")
        if not isinstance(name, str) or not name:
            raise ProjectFormatError("project name is required")

        interventions = []
        for raw in value.get("interventions", []):
            draft_raw = raw["draft"]
            draft = InterventionDraft(
                operation=InterventionOperation(draft_raw["operation"]),
                source_term=draft_raw.get("source_term"),
                target_term=draft_raw.get("target_term"),
                strength=float(draft_raw["strength"]),
                layer_start=int(draft_raw["layer_start"]),
                layer_end=int(draft_raw["layer_end"]),
                duration=draft_raw.get("duration", "next-token"),
                step_count=draft_raw.get("step_count"),
                match_mode=draft_raw.get("match_mode", "exact"),
                trigger=draft_raw.get("trigger", "manual"),
            )
            entry = InterventionEntry(
                intervention_id=raw["intervention_id"],
                draft=draft,
                label=raw.get("label", draft.operation.value.title()),
                enabled=bool(raw.get("enabled", False)),
                state=InterventionState(raw.get("state", "draft")),
                status_detail=raw.get("status_detail", "Draft"),
                applied_run_id=raw.get("applied_run_id"),
            )
            if imported:
                entry = replace(
                    entry,
                    enabled=False,
                    state=InterventionState.DRAFT,
                    status_detail="Imported — review required",
                    applied_run_id=None,
                )
            interventions.append(entry)

        rules = []
        for raw in value.get("rules", []):
            rule = RuleRecord(
                rule_id=raw["rule_id"],
                name=raw["name"],
                source=raw["source"],
                trigger=RuleTrigger(raw["trigger"]),
                priority=int(raw.get("priority", 100)),
                enabled=bool(raw.get("enabled", False)),
                trusted=bool(raw.get("trusted", False)),
                source_hash=raw.get("source_hash"),
                tested_hash=raw.get("tested_hash"),
                consecutive_failures=int(raw.get("consecutive_failures", 0)),
                last_result=raw.get("last_result", "Never tested"),
                config=dict(raw.get("config", {})),
            )
            if imported:
                rule = replace(
                    rule,
                    enabled=False,
                    trusted=False,
                    tested_hash=None,
                    last_result="Imported — review and test required",
                )
            rules.append(rule)

        runs = [
            RunRecord(
                run_id=raw["run_id"],
                prompt=raw["prompt"],
                mode=RunMode(raw["mode"]),
                state=RunState(raw["state"]),
                created_at=raw["created_at"],
                baseline_run_id=raw.get("baseline_run_id"),
                intervention_ids=tuple(raw.get("intervention_ids", [])),
                rule_ids=tuple(raw.get("rule_ids", [])),
                output_text=raw.get("output_text", ""),
                inspection_text=raw.get("inspection_text", ""),
                partial=bool(raw.get("partial", False)),
                generation_backend=GenerationBackend(
                    raw.get("generation_backend", "exact-bf16")
                ),
                quantization=raw.get("quantization", "BF16"),
                ttft_seconds=raw.get("ttft_seconds"),
                decode_tokens_per_second=raw.get("decode_rate"),
            )
            for raw in value.get("runs", [])
        ]
        return cls(
            project_id=value.get("project_id") or f"project-{uuid4().hex}",
            name=name,
            session=value.get("session"),
            prompts=list(value.get("prompts", [])),
            interventions=interventions,
            rules=rules,
            runs=runs,
            experiments=list(value.get("experiments", [])),
            trace_references=list(value.get("trace_references", [])),
            layout=dict(value.get("layout", {})),
        )

    @classmethod
    def load(cls, path: str | Path, *, imported: bool = False) -> ProjectDocument:
        path = Path(path)
        if path.stat().st_size > 16 * 1024 * 1024:
            raise ProjectFormatError("project exceeds 16 MiB")
        project = cls.from_json(path.read_text(encoding="utf-8"), imported=imported)
        project.path = path
        return project

    def save(self, path: str | Path | None = None) -> None:
        destination = Path(path) if path is not None else self.path
        if destination is None:
            raise ValueError("project has no save path")
        _reject_secrets(self.to_dict())
        temporary = destination.with_suffix(destination.suffix + ".tmp")
        temporary.write_text(
            json.dumps(self.to_dict(), ensure_ascii=False, indent=2, allow_nan=False),
            encoding="utf-8",
        )
        temporary.replace(destination)
        self.path = destination
        self.dirty = False
