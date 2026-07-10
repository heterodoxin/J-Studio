import threading
import time

import pytest

from jstudio.rules.protocol import RuleEvaluationRequest, SandboxLimits
from jstudio.rules.sandbox import QuickJSSandbox, _decode


def context():
    return {
        "event": {"type": "jspace.frame", "sequence": 4},
        "model": {"id": "qwen", "revision": "main", "layerCount": 64},
        "lens": {"id": "lens"},
        "layer": {"index": 42},
        "token": {"index": 7, "text": " result"},
        "generation": {"step": 7, "outputText": "bounded output"},
        "jspace": {
            "activations": [
                {"term": "injection", "score": 0.91},
                {"term": "warning", "score": 0.55},
            ]
        },
        "stack": {"entries": []},
        "tags": {},
    }


def request(source, *, limits=None):
    return RuleEvaluationRequest(
        source=source,
        trigger="jspace.frame",
        context=context(),
        layer_count=64,
        limits=limits or SandboxLimits(wall_time_ms=300),
    )


def test_valid_rule_returns_declarative_actions():
    sandbox = QuickJSSandbox()
    result = sandbox.evaluate(
        request(
            """function run(ctx) {
              if (ctx.jspace.has("injection", {minScore: 0.7})) {
                return [jspace.replace("injection", "trusted", {
                  strength: 0.8, layers: {from: 18, to: 26},
                  duration: "next-token", matchMode: "exact"
                }), rule.log("info", "guarded")];
              }
              return [];
            }"""
        )
    )
    assert result.success, result.error
    assert [action.kind for action in result.actions] == ["replace", "log"]
    assert result.metrics.execution_ms >= 0


def test_rule_actions_can_omit_strength():
    result = QuickJSSandbox().evaluate(
        request(
            """function run(ctx) {
              return [jspace.inject("cat", {
                layers: "current",
                duration: "next-token"
              })];
            }"""
        )
    )

    assert result.success, result.error
    assert result.actions[0].payload["strength"] == 16.0


@pytest.mark.parametrize(
    "source,error_fragment",
    [
        ("function run(ctx) { while (true) {} }", "time"),
        ("function run(ctx) { throw new Error('boom'); }", "boom"),
        (
            "function run(ctx) { return Array.from({length:33}, (_,i) => "
            "jspace.inject('x'+i,{strength:.5,layers:'all',duration:'next-token'})); }",
            "32",
        ),
        (
            "function run(ctx) { return [jspace.inject('x',"
            "{strength:17,layers:'all',duration:'next-token'})]; }",
            "strength",
        ),
        (
            "function run(ctx) { return [jspace.inject('x',"
            "{strength:.5,layers:{from:2,to:99},duration:'next-token'})]; }",
            "layer",
        ),
    ],
)
def test_rule_failures_apply_no_actions(source, error_fragment):
    result = QuickJSSandbox().evaluate(request(source))
    assert not result.success
    assert result.actions == ()
    assert error_fragment.lower() in result.error.lower()


def test_forbidden_source_never_starts_worker():
    result = QuickJSSandbox().evaluate(request("function run(ctx) { return [eval('1')]; }"))
    assert not result.success
    assert result.actions == ()
    assert "forbidden" in result.error.lower()


def test_pre_cancelled_request_fails_closed():
    cancelled = threading.Event()
    cancelled.set()
    result = QuickJSSandbox().evaluate(
        request("function run(ctx) { return []; }"), cancel_event=cancelled
    )
    assert not result.success
    assert result.actions == ()
    assert "cancel" in result.error.lower()


def test_limits_match_security_defaults():
    limits = SandboxLimits()
    assert limits.wall_time_ms == 50
    assert limits.execution_time_ms == 25
    assert limits.heap_bytes == 16 * 1024 * 1024
    assert limits.max_actions == 32


def test_rule_worker_startup_grace_is_longer_than_cold_python_imports():
    sandbox = QuickJSSandbox()

    assert sandbox.startup_timeout_ms >= 5000


def test_default_wall_limit_allows_small_valid_rule():
    result = QuickJSSandbox().evaluate(
        request("function run(ctx) { return []; }", limits=SandboxLimits())
    )
    assert result.success, result.error


def test_oversized_input_and_output_fail_closed():
    input_result = QuickJSSandbox().evaluate(
        request(
            "function run(ctx) { return []; }",
            limits=SandboxLimits(max_input_bytes=64, wall_time_ms=300),
        )
    )
    output_result = QuickJSSandbox().evaluate(
        request(
            "function run(ctx) { return [rule.log('info', 'x'.repeat(2048))]; }",
            limits=SandboxLimits(max_output_bytes=128, wall_time_ms=300),
        )
    )

    assert not input_result.success and input_result.actions == ()
    assert "input exceeds" in input_result.error
    assert not output_result.success and output_result.actions == ()
    assert "output exceeds" in output_result.error


def test_non_serializable_worker_output_is_rejected():
    result = QuickJSSandbox().evaluate(
        request("function run(ctx) { const value = []; value.push(value); return value; }")
    )

    assert not result.success
    assert result.actions == ()
    assert "circular" in result.error.lower()


@pytest.mark.parametrize(
    "source",
    [
        "function run(ctx) { import('x'); return []; }",
        "function run(ctx) { return [fetch('x')]; }",
        "function run(ctx) { return [Function('return 1')()]; }",
        "function run(ctx) { return [Math.random()]; }",
    ],
)
def test_host_and_dynamic_capabilities_are_forbidden(source):
    result = QuickJSSandbox().evaluate(request(source))

    assert not result.success
    assert result.actions == ()
    assert "forbidden" in result.error.lower()


def test_recursion_and_heap_exhaustion_fail_closed():
    recursion = QuickJSSandbox().evaluate(
        request(
            "function run(ctx) { function recurse() { return recurse(); } "
            "return recurse(); }"
        )
    )
    memory = QuickJSSandbox().evaluate(
        request(
            "function run(ctx) { const x = new Array(4000000).fill('xxxxxxxx'); "
            "return x; }",
            limits=SandboxLimits(heap_bytes=1024 * 1024, wall_time_ms=300),
        )
    )

    assert not recursion.success and recursion.actions == ()
    assert not memory.success and memory.actions == ()


def test_parent_can_cancel_running_worker():
    cancelled = threading.Event()

    def cancel_soon():
        time.sleep(0.02)
        cancelled.set()

    thread = threading.Thread(target=cancel_soon)
    thread.start()
    result = QuickJSSandbox().evaluate(
        request(
            "function run(ctx) { while (true) {} }",
            limits=SandboxLimits(wall_time_ms=500, execution_time_ms=400),
        ),
        cancel_event=cancelled,
    )
    thread.join()

    assert not result.success
    assert result.actions == ()
    assert "cancel" in result.error.lower()


def test_protocol_decoder_rejects_truncated_and_oversized_worker_messages():
    with pytest.raises(ValueError):
        _decode(b"\x00\x00\x00\x05{}", 100)
    with pytest.raises(ValueError):
        _decode(b"\x00\x00\x00\x02{}", 1)
