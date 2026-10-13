"""frozentrace — deterministic record & replay for AI agent runs.

Unlike VCR-style tools that match calls by *position*, frozentrace matches by a
stable call signature, so a run replays identically **even when the agent runs
its calls concurrently** (``asyncio.gather``, threads) or streams responses.
"""

from __future__ import annotations

from .errors import AgentVcrError, DivergenceError, ReplayMiss, TraceNotFound
from .intercept import intercept
from .session import (
    Player,
    Recorder,
    current_session,
    record,
    replay,
    replay_from_env,
)
from .testing import assert_deterministic
from .trace import Trace, TraceStep, diff_traces, summarize

__version__ = "0.1.0"

__all__ = [
    "intercept",
    "record",
    "replay",
    "replay_from_env",
    "current_session",
    "Recorder",
    "Player",
    "Trace",
    "TraceStep",
    "summarize",
    "diff_traces",
    "assert_deterministic",
    "AgentVcrError",
    "ReplayMiss",
    "DivergenceError",
    "TraceNotFound",
    "__version__",
]
