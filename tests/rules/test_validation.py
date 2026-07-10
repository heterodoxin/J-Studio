import math

import pytest

from jstudio.rules.validation import (
    DEFAULT_MAX_STRENGTH,
    validate_actions,
    validate_rule_source,
)


@pytest.mark.parametrize(
    "forbidden",
    [
        "eval('1')",
        "Function('return 1')",
        "import('x')",
        "async function bad() {}",
        "function* bad() {}",
        "WebAssembly.compile()",
        "Date.now()",
        "Math.random()",
        "new Promise(() => {})",
    ],
)
def test_forbidden_source_fails_closed(forbidden):
    result = validate_rule_source(f"function run(ctx) {{ {forbidden}; return []; }}")
    assert not result.valid
    assert result.problems


def test_source_requires_exact_single_entry_function():
    assert not validate_rule_source("return [];").valid
    assert not validate_rule_source("const x = 1; function run(ctx) { return []; }").valid
    assert validate_rule_source("function run(ctx) { return []; }").valid


def test_action_validator_rejects_nonfinite_strength_and_extra_keys():
    result = validate_actions(
        [
            {
                "type": "inject",
                "term": "x",
                "strength": math.inf,
                "layers": "all",
                "duration": "next-token",
                "escape": True,
            }
        ],
        layer_count=64,
    )
    assert result.validated == ()
    assert len(result.rejected) == 1


def test_actions_are_sorted_by_declared_conflict_order():
    raw = [
        {"type": "log", "level": "info", "message": "done"},
        {
            "type": "inject",
            "term": "caution",
            "strength": 0.5,
            "layers": "current",
            "duration": "next-token",
        },
        {"type": "stop", "reason": "unsafe"},
        {
            "type": "replace",
            "matchTerm": "injection",
            "replacementTerm": "trusted",
            "strength": 0.8,
            "layers": {"from": 18, "to": 26},
            "duration": "next-token",
            "matchMode": "exact",
        },
    ]
    result = validate_actions(raw, layer_count=64)
    assert [action.kind for action in result.validated] == [
        "stop",
        "replace",
        "inject",
        "log",
    ]


def test_jspace_actions_default_to_max_strength_budget():
    result = validate_actions(
        [
            {
                "type": "inject",
                "term": "cat",
                "layers": "current",
                "duration": "next-token",
            },
            {
                "type": "replace",
                "matchTerm": "dog",
                "replacementTerm": "cat",
                "layers": "all",
                "duration": "next-token",
                "matchMode": "exact",
            },
            {
                "type": "suppress",
                "term": "dog",
                "layers": "all",
                "duration": "next-token",
            },
        ],
        layer_count=64,
    )

    assert result.rejected == ()
    assert [action.payload["strength"] for action in result.validated] == [
        DEFAULT_MAX_STRENGTH,
        DEFAULT_MAX_STRENGTH,
        DEFAULT_MAX_STRENGTH,
    ]
