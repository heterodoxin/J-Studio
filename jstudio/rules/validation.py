"""Static source checks and strict declarative action validation."""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

from jstudio.rules.protocol import RejectedRuleAction, ValidatedRuleAction

DEFAULT_MAX_STRENGTH = 16.0


@dataclass(frozen=True, slots=True)
class SourceValidation:
    valid: bool
    problems: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ActionValidation:
    validated: tuple[ValidatedRuleAction, ...]
    rejected: tuple[RejectedRuleAction, ...]


_ENTRY = re.compile(r"\A\s*function\s+run\s*\(\s*ctx\s*\)\s*\{[\s\S]*\}\s*\Z")
_FORBIDDEN = (
    (re.compile(r"\beval\s*\("), "eval is forbidden"),
    (re.compile(r"\bFunction\s*\("), "Function is forbidden"),
    (re.compile(r"\bimport\s*(?:\(|[\w{*])"), "imports are forbidden"),
    (re.compile(r"\basync\s+function\b"), "async functions are forbidden"),
    (re.compile(r"\bfunction\s*\*"), "generators are forbidden"),
    (re.compile(r"\bPromise\b"), "promises are forbidden"),
    (re.compile(r"\bWebAssembly\b"), "WebAssembly is forbidden"),
    (re.compile(r"\bDate\b"), "clock access is forbidden"),
    (re.compile(r"\bMath\s*\.\s*random\b"), "randomness is forbidden"),
    (
        re.compile(r"\b(?:process|require|Deno|Bun|fetch|XMLHttpRequest)\b"),
        "host capability is forbidden",
    ),
)


def validate_rule_source(source: str, *, max_bytes: int = 128 * 1024) -> SourceValidation:
    problems = []
    if not isinstance(source, str):
        return SourceValidation(False, ("source must be text",))
    if len(source.encode("utf-8")) > max_bytes:
        problems.append(f"source exceeds {max_bytes} bytes")
    if not _ENTRY.fullmatch(source):
        problems.append(
            "source must contain exactly function run(ctx) and no top-level statements"
        )
    for pattern, message in _FORBIDDEN:
        if pattern.search(source):
            problems.append(message)
    return SourceValidation(not problems, tuple(problems))


def _exact_keys(action: dict, allowed: set[str], required: set[str]) -> str | None:
    missing = required - set(action)
    extra = set(action) - allowed
    if missing:
        return f"missing keys: {sorted(missing)}"
    if extra:
        return f"unknown keys: {sorted(extra)}"
    return None


def _text(value: Any, name: str, *, maximum: int = 4096) -> str | None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        return f"{name} must be non-empty text up to {maximum} characters"
    return None


def _strength(value: Any) -> str | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return "strength must be a finite number"
    if not math.isfinite(float(value)) or not 0 <= float(value) <= 16:
        return "strength must be finite and lie in [0, 16]"
    return None


def _layers(value: Any, layer_count: int) -> str | None:
    if value in ("current", "all"):
        return None
    if not isinstance(value, dict) or set(value) != {"from", "to"}:
        return "layers must be current, all, or an exact {from, to} object"
    start, stop = value["from"], value["to"]
    if not isinstance(start, int) or not isinstance(stop, int):
        return "layer bounds must be integers"
    if start < 0 or stop < start or stop >= layer_count:
        return f"layer range must lie within 0..{layer_count - 1}"
    return None


def _duration(value: Any) -> str | None:
    if value in ("current-token", "next-token", "generation"):
        return None
    if (
        isinstance(value, dict)
        and set(value) == {"steps"}
        and isinstance(value["steps"], int)
        and 1 <= value["steps"] <= 10000
    ):
        return None
    return "invalid duration"


def _validate_action(action: Any, layer_count: int) -> str | None:
    if not isinstance(action, dict):
        return "action must be an object"
    kind = action.get("type")
    if kind == "inject":
        problem = _exact_keys(
            action,
            {"type", "term", "strength", "layers", "duration", "label"},
            {"type", "term", "layers", "duration"},
        )
        return (
            problem
            or _text(action["term"], "term")
            or (
                _strength(action["strength"])
                if "strength" in action
                else None
            )
            or _layers(action["layers"], layer_count)
            or _duration(action["duration"])
        )
    if kind == "replace":
        problem = _exact_keys(
            action,
            {
                "type",
                "matchTerm",
                "replacementTerm",
                "strength",
                "layers",
                "duration",
                "matchMode",
            },
            {
                "type",
                "matchTerm",
                "replacementTerm",
                "layers",
                "duration",
                "matchMode",
            },
        )
        if problem:
            return problem
        if action["matchMode"] not in ("exact", "case-insensitive"):
            return "matchMode must be exact or case-insensitive"
        return (
            _text(action["matchTerm"], "matchTerm")
            or _text(action["replacementTerm"], "replacementTerm")
            or (
                _strength(action["strength"])
                if "strength" in action
                else None
            )
            or _layers(action["layers"], layer_count)
            or _duration(action["duration"])
        )
    if kind == "suppress":
        problem = _exact_keys(
            action,
            {"type", "term", "strength", "layers", "duration"},
            {"type", "term", "layers", "duration"},
        )
        return (
            problem
            or _text(action["term"], "term")
            or (
                _strength(action["strength"])
                if "strength" in action
                else None
            )
            or _layers(action["layers"], layer_count)
            or _duration(action["duration"])
        )
    if kind == "stop":
        return _exact_keys(action, {"type", "reason"}, {"type", "reason"}) or _text(
            action["reason"], "reason"
        )
    if kind == "log":
        problem = _exact_keys(
            action, {"type", "level", "message"}, {"type", "level", "message"}
        )
        if problem:
            return problem
        if action["level"] not in ("debug", "info", "warn", "error"):
            return "invalid log level"
        return _text(action["message"], "message", maximum=8192)
    if kind == "tag":
        problem = _exact_keys(action, {"type", "name", "value"}, {"type", "name", "value"})
        if problem:
            return problem
        if isinstance(action["value"], (dict, list)):
            return "tag value must be a JSON scalar"
        if isinstance(action["value"], float) and not math.isfinite(action["value"]):
            return "tag value must be finite"
        return _text(action["name"], "name", maximum=128)
    return f"unknown action type: {kind!r}"


def validate_actions(
    actions: Any, *, layer_count: int, max_actions: int = 32
) -> ActionValidation:
    if not isinstance(actions, list):
        return ActionValidation((), (RejectedRuleAction(-1, "rule must return an array"),))
    if len(actions) > max_actions:
        return ActionValidation(
            (),
            (
                RejectedRuleAction(
                    -1, f"rule returned {len(actions)} actions; limit is {max_actions}"
                ),
            ),
        )
    validated = []
    rejected = []
    for index, action in enumerate(actions):
        problem = _validate_action(action, layer_count)
        if problem:
            rejected.append(RejectedRuleAction(index, problem))
        else:
            payload = dict(action)
            if payload["type"] in {"inject", "replace", "suppress"}:
                payload.setdefault("strength", DEFAULT_MAX_STRENGTH)
            validated.append(ValidatedRuleAction(payload["type"], payload))
    order = {"stop": 0, "replace": 1, "suppress": 2, "inject": 3, "tag": 4, "log": 5}
    validated.sort(key=lambda item: order[item.kind])
    return ActionValidation(tuple(validated), tuple(rejected))
