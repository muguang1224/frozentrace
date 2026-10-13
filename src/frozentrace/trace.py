"""Trace (recorded run) data model, loading/saving, summarising and diffing."""

from __future__ import annotations

import datetime as _dt
import json
import os
from dataclasses import dataclass, field
from typing import Any, Optional

from . import errors

FORMAT = "frozentrace/1"


def _now_iso() -> str:
    return _dt.datetime.now(_dt.timezone.utc).replace(microsecond=0).isoformat()


@dataclass
class TraceStep:
    signature: str
    kind: str
    name: str
    arguments: dict
    result: Any
    occurrence: int = 0
    streaming: bool = False
    input_tokens: Optional[int] = None
    output_tokens: Optional[int] = None
    duration_ms: Optional[float] = None

    def to_dict(self) -> dict:
        return {
            "signature": self.signature,
            "kind": self.kind,
            "name": self.name,
            "arguments": self.arguments,
            "result": self.result,
            "occurrence": self.occurrence,
            "streaming": self.streaming,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "duration_ms": self.duration_ms,
        }

    @classmethod
    def from_dict(cls, d: dict) -> "TraceStep":
        return cls(
            signature=d["signature"],
            kind=d.get("kind", "call"),
            name=d.get("name", "call"),
            arguments=d.get("arguments", {}),
            result=d.get("result"),
            occurrence=d.get("occurrence", 0),
            streaming=d.get("streaming", False),
            input_tokens=d.get("input_tokens"),
            output_tokens=d.get("output_tokens"),
            duration_ms=d.get("duration_ms"),
        )


@dataclass
class Trace:
    steps: list = field(default_factory=list)
    meta: dict = field(default_factory=dict)

    # -- persistence ---------------------------------------------------------
    def to_dict(self) -> dict:
        return {
            "format": FORMAT,
            "meta": self.meta,
            "steps": [s.to_dict() for s in self.steps],
        }

    def save(self, path: str, redact=None) -> None:
        steps = self.steps
        if redact:
            steps = [TraceStep.from_dict(redact_step(s.to_dict(), redact)) for s in self.steps]
        data = {"format": FORMAT, "meta": self.meta, "steps": [s.to_dict() for s in steps]}
        parent = os.path.dirname(os.path.abspath(path))
        if parent:
            os.makedirs(parent, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(data, fh, indent=2, ensure_ascii=False)
            fh.write("\n")

    @classmethod
    def load(cls, path: str) -> "Trace":
        if not os.path.exists(path):
            raise errors.TraceNotFound(f"no trace file at {path!r}")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return cls(
            steps=[TraceStep.from_dict(s) for s in data.get("steps", [])],
            meta=data.get("meta", {}),
        )

    # -- helpers -------------------------------------------------------------
    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def total_input_tokens(self) -> int:
        return sum(s.input_tokens or 0 for s in self.steps)

    @property
    def total_output_tokens(self) -> int:
        return sum(s.output_tokens or 0 for s in self.steps)

    @property
    def total_tokens(self) -> int:
        return self.total_input_tokens + self.total_output_tokens

    @property
    def duration_ms(self) -> float:
        return sum(s.duration_ms or 0.0 for s in self.steps)


# --------------------------------------------------------------------------- #
# redaction
# --------------------------------------------------------------------------- #
def redact_step(step: dict, keys, replacement: str = "****") -> dict:
    keys = set(keys or [])
    out = dict(step)
    out["arguments"] = _redact(out.get("arguments"), keys, replacement)
    out["result"] = _redact(out.get("result"), keys, replacement)
    return out


def _redact(obj, keys, replacement):
    if isinstance(obj, dict):
        return {
            k: (replacement if k in keys else _redact(v, keys, replacement))
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [_redact(v, keys, replacement) for v in obj]
    return obj


# --------------------------------------------------------------------------- #
# reporting
# --------------------------------------------------------------------------- #
def summarize(trace: Trace) -> dict:
    by_kind: dict = {}
    by_name: dict = {}
    for s in trace.steps:
        by_kind[s.kind] = by_kind.get(s.kind, 0) + 1
        key = f"{s.kind}:{s.name}"
        by_name[key] = by_name.get(key, 0) + 1
    return {
        "steps": trace.num_steps,
        "streaming_steps": sum(1 for s in trace.steps if s.streaming),
        "by_kind": by_kind,
        "by_name": by_name,
        "input_tokens": trace.total_input_tokens,
        "output_tokens": trace.total_output_tokens,
        "total_tokens": trace.total_tokens,
        "duration_ms": round(trace.duration_ms, 3),
        "meta": trace.meta,
    }


def diff_traces(a: Trace, b: Trace) -> dict:
    from collections import Counter

    ca = Counter(s.signature for s in a.steps)
    cb = Counter(s.signature for s in b.steps)
    added = cb - ca
    removed = ca - cb
    return {
        "added_calls": sum(added.values()),
        "removed_calls": sum(removed.values()),
        "changed_calls": sum(
            1
            for sa, sb in zip(a.steps, b.steps)
            if sa.signature == sb.signature and sa.result != sb.result
        ),
        "step_delta": b.num_steps - a.num_steps,
        "token_delta": b.total_tokens - a.total_tokens,
        "identical": ca == cb and [s.result for s in a.steps] == [s.result for s in b.steps],
    }
