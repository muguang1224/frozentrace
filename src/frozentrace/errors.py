"""Exception types for frozentrace."""

from __future__ import annotations


class AgentVcrError(Exception):
    """Base class for all frozentrace errors."""


class TraceNotFound(AgentVcrError):
    """Raised when loading a trace file that does not exist."""


class ReplayMiss(AgentVcrError):
    """Raised during replay when the agent makes a call that was never recorded.

    This is the loud failure that makes replay trustworthy: instead of silently
    returning ``None`` or calling the real (networked) function, frozentrace stops.
    """


class DivergenceError(AgentVcrError):
    """Raised in strict replay when the run diverged from the recording.

    Triggered when recorded calls were never replayed, i.e. the agent took a
    different path than it did when it was recorded.
    """
