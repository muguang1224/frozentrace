"""Recording and replay sessions.

State lives in a module-level stack guarded by a lock, so the active session is
visible from sync code, ``async`` code (same thread) and worker threads alike.
"""

from __future__ import annotations

import os
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

from . import errors
from .canonical import jsonable, signature
from .trace import Trace, TraceStep

_LOCK = threading.RLock()
_STACK = []


def current_session():
    """Return the active recording/replay session, or ``None``."""
    with _LOCK:
        return _STACK[-1] if _STACK else None


def _push(session) -> None:
    with _LOCK:
        _STACK.append(session)


def _pop(session) -> None:
    with _LOCK:
        if _STACK and _STACK[-1] is session:
            _STACK.pop()
        elif session in _STACK:
            _STACK.remove(session)


def _is_stream(obj) -> bool:
    return hasattr(obj, "__next__") and not isinstance(obj, (str, bytes, bytearray))


def _extract_usage(result):
    """Return (input_tokens, output_tokens) if the result carries exact usage."""
    usage = None
    if isinstance(result, dict):
        usage = result.get("usage")
        if usage is None and isinstance(result.get("response"), dict):
            usage = result["response"].get("usage")
    if isinstance(usage, dict):
        it = usage.get("prompt_tokens", usage.get("input_tokens"))
        ot = usage.get("completion_tokens", usage.get("output_tokens"))
        if it is not None or ot is not None:
            return it, ot
    return None, None


async def _aiter(chunks):
    for c in chunks:
        yield c


class Session:
    """Common base. ``fn`` is the real callable; replay ignores it."""

    def on_call_sync(self, fn, name, kind, args, kwargs, stream, key):
        raise NotImplementedError

    async def on_call_async(self, fn, name, kind, args, kwargs, stream, key):
        raise NotImplementedError

    async def on_call_agen(self, fn, name, kind, args, kwargs, stream, key):
        raise NotImplementedError


class Recorder(Session):
    def __init__(self, path=None, model=None, redact=None):
        self.path = path
        self.model = model
        self.redact = list(redact or [])
        self.steps: list = []
        self._occ = defaultdict(int)
        self._lock = threading.RLock()

    # -- internals -----------------------------------------------------------
    def _finish(self, kind, name, args, kwargs, key, payload, streaming, it, ot, dur):
        sig = signature(kind, name, args, kwargs, key)
        with self._lock:
            occ = self._occ[sig]
            self._occ[sig] += 1
            self.steps.append(
                TraceStep(
                    signature=sig,
                    kind=kind,
                    name=name,
                    arguments={"args": jsonable(list(args)), "kwargs": jsonable(dict(kwargs))},
                    result=payload,
                    occurrence=occ,
                    streaming=streaming,
                    input_tokens=it,
                    output_tokens=ot,
                    duration_ms=dur,
                )
            )

    def trace(self) -> Trace:
        from .trace import _now_iso

        return Trace(
            steps=list(self.steps),
            meta={
                "recorded_at": _now_iso(),
                "model": self.model,
                "redacts": list(self.redact),
                "recorder": "agentvcr",
            },
        )

    # -- Session -------------------------------------------------------------
    def on_call_sync(self, fn, name, kind, args, kwargs, stream, key):
        t0 = time.perf_counter()
        result = fn(*args, **kwargs)
        dur = (time.perf_counter() - t0) * 1000.0
        streaming = stream if stream is not None else _is_stream(result)
        if streaming:
            chunks = list(result)
            self._finish(kind, name, args, kwargs, key,
                         {"__stream__": [jsonable(c) for c in chunks]}, True, None, None, dur)
            return iter(chunks)
        it, ot = _extract_usage(result)
        self._finish(kind, name, args, kwargs, key, jsonable(result), False, it, ot, dur)
        return result

    async def on_call_async(self, fn, name, kind, args, kwargs, stream, key):
        t0 = time.perf_counter()
        result = await fn(*args, **kwargs)
        dur = (time.perf_counter() - t0) * 1000.0
        if hasattr(result, "__anext__"):  # coroutine returned an async generator
            chunks = [c async for c in result]
            self._finish(kind, name, args, kwargs, key,
                         {"__astream__": [jsonable(c) for c in chunks]}, True, None, None, dur)
            return _aiter(chunks)
        streaming = stream if stream is not None else _is_stream(result)
        if streaming:
            chunks = list(result)
            self._finish(kind, name, args, kwargs, key,
                         {"__stream__": [jsonable(c) for c in chunks]}, True, None, None, dur)
            return iter(chunks)
        it, ot = _extract_usage(result)
        self._finish(kind, name, args, kwargs, key, jsonable(result), False, it, ot, dur)
        return result

    async def on_call_agen(self, fn, name, kind, args, kwargs, stream, key):
        t0 = time.perf_counter()
        chunks = [c async for c in fn(*args, **kwargs)]
        dur = (time.perf_counter() - t0) * 1000.0
        self._finish(kind, name, args, kwargs, key,
                     {"__astream__": [jsonable(c) for c in chunks]}, True, None, None, dur)
        return _aiter(chunks)


class Player(Session):
    def __init__(self, trace: Trace, strict: bool = False):
        self.trace = trace
        self.strict = strict
        self._index = defaultdict(list)
        for step in trace.steps:
            self._index[step.signature].append(step)
        self._counters = defaultdict(int)
        self._consumed = set()
        self._lock = threading.RLock()
        self.divergences: list = []

    # -- internals -----------------------------------------------------------
    def _serve(self, kind, name, args, kwargs, key) -> TraceStep:
        sig = signature(kind, name, args, kwargs, key)
        with self._lock:
            lst = self._index.get(sig)
            occ = self._counters.get(sig, 0)
            if not lst:
                self.divergences.append({"kind": kind, "name": name, "reason": "unknown"})
                raise errors.ReplayMiss(
                    f"replay miss: no recorded {kind}:{name} call with these arguments"
                )
            if occ >= len(lst):
                self.divergences.append({"kind": kind, "name": name, "reason": "exhausted"})
                raise errors.ReplayMiss(
                    f"replay miss: {kind}:{name} called more times than it was recorded"
                )
            step = lst[occ]
            self._counters[sig] = occ + 1
            self._consumed.add((sig, occ))
            return step

    @staticmethod
    def _materialize(step: TraceStep):
        payload = step.result
        if isinstance(payload, dict):
            if "__astream__" in payload:
                return "aiter", payload["__astream__"]
            if "__stream__" in payload:
                return "iter", payload["__stream__"]
        if step.streaming:
            return "iter", payload
        return "value", payload

    # -- Session -------------------------------------------------------------
    def on_call_sync(self, fn, name, kind, args, kwargs, stream, key):
        step = self._serve(kind, name, args, kwargs, key)
        mode, payload = self._materialize(step)
        return iter(payload) if mode == "iter" else payload

    async def on_call_async(self, fn, name, kind, args, kwargs, stream, key):
        step = self._serve(kind, name, args, kwargs, key)
        mode, payload = self._materialize(step)
        if mode == "aiter":
            return _aiter(payload)
        return iter(payload) if mode == "iter" else payload

    async def on_call_agen(self, fn, name, kind, args, kwargs, stream, key):
        step = self._serve(kind, name, args, kwargs, key)
        mode, payload = self._materialize(step)
        if mode == "aiter":
            return _aiter(payload)
        return iter(payload) if mode == "iter" else payload

    # -- determinism contract -------------------------------------------------
    def _all_keys(self):
        return {(sig, i) for sig, lst in self._index.items() for i in range(len(lst))}

    def unconsumed(self):
        with self._lock:
            return self._all_keys() - self._consumed

    def assert_complete(self) -> None:
        left = self.unconsumed()
        if left:
            names = sorted(f"{sig}@{i}" for sig, i in left)[:5]
            raise errors.DivergenceError(
                f"{len(left)} recorded call(s) were never replayed — the agent took a "
                f"different path than the recording: {names}"
            )


# --------------------------------------------------------------------------- #
# context managers
# --------------------------------------------------------------------------- #
@contextmanager
def record(path=None, *, model=None, redact=None):
    """Record every intercepted call made inside the block into ``path``."""
    rec = Recorder(path=path, model=model, redact=redact)
    _push(rec)
    try:
        yield rec
    except BaseException:
        _pop(rec)
        raise
    else:
        _pop(rec)
        if path is not None:
            rec.trace().save(path, redact=rec.redact)


@contextmanager
def replay(path, *, strict: bool = False):
    """Serve recorded results for intercepted calls; never runs the real ones."""
    trace = Trace.load(path)
    player = Player(trace, strict=strict)
    _push(player)
    try:
        yield player
    except BaseException:
        _pop(player)
        raise
    else:
        _pop(player)
        if strict:
            player.assert_complete()


@contextmanager
def replay_from_env(var: str = "AGENTVCR_REPLAY", strict=None):
    """Replay the trace named by an env var, if set. Otherwise a no-op.

    Lets a CLI drive a user's script into replay mode without code changes.
    """
    path = os.environ.get(var)
    if not path:
        yield None
        return
    if strict is None:
        strict = os.environ.get("AGENTVCR_STRICT", "").lower() not in ("", "0", "false", "no")
    with replay(path, strict=strict) as player:
        yield player
