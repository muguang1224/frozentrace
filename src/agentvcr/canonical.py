"""Canonicalisation and call signatures.

The signature is the heart of agentvcr's order-independence. Two calls with the
same ``kind``, ``name`` and arguments produce the same signature no matter *when*
they happen, so replay does not have to care about the order in which an agent's
calls complete.
"""

from __future__ import annotations

import hashlib
import json
import re

_ADDR = re.compile(r"0x[0-9a-fA-F]+")


def safe_repr(obj: object) -> str:
    """A repr with memory addresses scrubbed so it is stable across runs."""
    return _ADDR.sub("0xADDR", repr(obj))


def _default(obj: object) -> dict:
    return {"__repr__": safe_repr(obj)}


def canonical(obj: object) -> str:
    """Deterministic JSON string for an arbitrary object."""
    try:
        return json.dumps(
            obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=_default
        )
    except Exception:
        return json.dumps({"__repr__": safe_repr(obj)}, separators=(",", ":"), ensure_ascii=False)


def jsonable(obj: object) -> object:
    """Best-effort conversion to a JSON-round-trippable value."""
    try:
        return json.loads(json.dumps(obj, ensure_ascii=False, default=_default))
    except Exception:
        return {"__repr__": safe_repr(obj)}


def signature(kind: str, name: str, args, kwargs, key=None) -> str:
    """Stable 16-hex-char signature for a call.

    ``key`` is an optional caller-supplied discriminator. It exists for the one
    case order-independence cannot solve by itself: the *same* call made
    concurrently that legitimately returns *different* results. Give those calls
    a stable ``key`` and they replay deterministically.
    """
    parts = [
        str(kind),
        str(name),
        canonical(list(args)),
        canonical(dict(kwargs)) if kwargs else "{}",
        canonical(key) if key is not None else "",
    ]
    h = hashlib.sha1()
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]
