"""Spawned QuickJS worker. This module has no Qt or application imports."""

from __future__ import annotations

import json
import os
import struct
import time


def _encode(value: dict) -> bytes:
    payload = json.dumps(value, ensure_ascii=False, allow_nan=False).encode("utf-8")
    return struct.pack(">I", len(payload)) + payload


def _decode(message: bytes, maximum: int) -> dict:
    if len(message) < 4:
        raise ValueError("malformed length-prefixed message")
    length = struct.unpack(">I", message[:4])[0]
    if length != len(message) - 4 or length > maximum:
        raise ValueError("malformed or oversized message")
    value = json.loads(message[4:].decode("utf-8"))
    if not isinstance(value, dict):
        raise ValueError("request must be an object")
    return value


def _set_resource_limits(limits: dict) -> None:
    try:
        import resource

        address = int(limits["address_space_bytes"])
        stack = int(limits["stack_bytes"])
        resource.setrlimit(resource.RLIMIT_AS, (address, address))
        resource.setrlimit(resource.RLIMIT_STACK, (stack, stack))
    except (ImportError, OSError, ValueError):
        pass


def _bootstrap(context: dict) -> str:
    raw_json = json.dumps(context, ensure_ascii=False, allow_nan=False)
    encoded = json.dumps(raw_json)
    return f"""
      'use strict';
      const __raw = JSON.parse({encoded});
      function __deepFreeze(value) {{
        if (value && typeof value === 'object' && !Object.isFrozen(value)) {{
          Object.freeze(value);
          Object.keys(value).forEach(k => __deepFreeze(value[k]));
        }}
        return value;
      }}
      const __activations = ((__raw.jspace || {{}}).activations || []).slice(0, 100);
      const __jspaceContext = {{
        has(term, options={{}}) {{
          const minScore = options.minScore === undefined ? -Infinity : options.minScore;
          return __activations.some(item => item.term === term && item.score >= minScore);
        }},
        score(term) {{ const item=__activations.find(value => value.term === term); return item ? item.score : null; }},
        top(limit) {{ const n=Math.max(1,Math.min(100,Number(limit)||1)); return __deepFreeze(__activations.slice(0,n).map(value=>({{...value}}))); }},
        find(pattern) {{ const text=String(pattern); return __deepFreeze(__activations.filter(value=>value.term.includes(text)).slice(0,100).map(value=>({{...value}}))); }}
      }};
      const ctx = __deepFreeze({{
        ...__raw,
        jspace: __jspaceContext,
        stack: {{active() {{ return __deepFreeze((((__raw.stack||{{}}).entries)||[]).slice(0,100).map(value=>({{...value}}))); }}}},
        tags: {{get(name) {{ return Object.prototype.hasOwnProperty.call(__raw.tags||{{}},name) ? __raw.tags[name] : null; }}}}
      }});
      const jspace = __deepFreeze({{
        inject(term, options={{}}) {{ return __deepFreeze({{type:'inject',term,...options}}); }},
        replace(matchTerm, replacementTerm, options={{}}) {{ return __deepFreeze({{type:'replace',matchTerm,replacementTerm,...options}}); }},
        suppress(term, options={{}}) {{ return __deepFreeze({{type:'suppress',term,...options}}); }}
      }});
      const generation = __deepFreeze({{stop(reason) {{ return __deepFreeze({{type:'stop',reason}}); }}}});
      const rule = __deepFreeze({{
        log(level,message) {{ return __deepFreeze({{type:'log',level,message}}); }},
        tag(name,value) {{ return __deepFreeze({{type:'tag',name,value}}); }}
      }});
      globalThis.eval=undefined; globalThis.Function=undefined; globalThis.Promise=undefined;
      globalThis.WebAssembly=undefined; globalThis.Date=undefined;
      globalThis.fetch=undefined; globalThis.XMLHttpRequest=undefined;
      Math.random=undefined; Object.freeze(Math);
    """


def worker_main(request_receiver, response_sender) -> None:
    response = {"success": False, "error": "worker did not complete", "raw_json": "[]"}
    try:
        import quickjs

        message = request_receiver.recv_bytes()
        request = _decode(message, 1024 * 1024)
        limits = request["limits"]
        os.environ.clear()
        _set_resource_limits(limits)
        context = quickjs.Context()
        context.set_memory_limit(int(limits["heap_bytes"]))
        context.set_max_stack_size(int(limits["stack_bytes"]))
        context.set_time_limit(float(limits["execution_time_ms"]) / 1000.0)
        response_sender.send_bytes(_encode({"ready": True}))
        started = time.perf_counter()
        context.eval(_bootstrap(request["context"]))
        context.eval(request["source"])
        raw_json = context.eval("JSON.stringify(run(ctx))")
        execution_ms = (time.perf_counter() - started) * 1000
        if raw_json is None:
            raise ValueError("rule returned undefined")
        output_bytes = len(raw_json.encode("utf-8"))
        if output_bytes > int(limits["max_output_bytes"]):
            raise ValueError(f"rule output exceeds {limits['max_output_bytes']} bytes")
        response = {
            "success": True,
            "error": "",
            "raw_json": raw_json,
            "execution_ms": execution_ms,
            "peak_worker_bytes": int(context.memory().get("memory_used_size", 0)),
            "output_bytes": output_bytes,
        }
    except BaseException as exc:
        error = str(exc) or type(exc).__name__
        if "interrupted" in error.lower():
            error = "QuickJS execution time limit exceeded"
        response = {
            "success": False,
            "error": error,
            "raw_json": "[]",
            "execution_ms": 0.0,
            "peak_worker_bytes": 0,
            "output_bytes": 2,
        }
    try:
        response_sender.send_bytes(_encode(response))
    finally:
        request_receiver.close()
        response_sender.close()
