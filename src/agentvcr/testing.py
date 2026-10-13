"""Helper for asserting that a replayed agent run is deterministic."""

from __future__ import annotations

from .session import replay


def assert_deterministic(run, trace_path, *, repeats: int = 20, strict: bool = True):
    """Run ``run()`` ``repeats`` times under replay and assert identical results.

    This is the measurable guarantee agentvcr exists to provide. If the agent
    schedules its work non-deterministically (e.g. concurrent tool calls that
    finish in a different order each time), signature-based replay still yields
    the same result on every run — and this helper proves it.

    Returns the (common) result so callers can assert on it too.
    """
    results = []
    for _ in range(repeats):
        with replay(trace_path, strict=strict):
            results.append(run())
    first = results[0]
    for i, other in enumerate(results):
        if other != first:
            raise AssertionError(
                f"non-deterministic replay: run #0 and run #{i} differ\n"
                f"  run  0 -> {first!r}\n"
                f"  run {i} -> {other!r}"
            )
    return first
