from __future__ import annotations

import asyncio
import json
import random
import time
from concurrent.futures import ThreadPoolExecutor, as_completed

import pytest

import frozentrace
from frozentrace import DivergenceError, ReplayMiss, Trace
from frozentrace.cli import main as cli_main

# --------------------------------------------------------------------------- #
# seams, wrapped once at import
# --------------------------------------------------------------------------- #
_COUNTED = {"n": 0}


@frozentrace.intercept(kind="llm", name="model")
def model(prompt):
    return {"text": prompt.upper(), "usage": {"prompt_tokens": 3, "output_tokens": 5}}


@frozentrace.intercept(kind="llm", name="amodel")
async def amodel(prompt):
    return {"text": f"async:{prompt}", "usage": {"prompt_tokens": 2, "output_tokens": 4}}


@frozentrace.intercept(kind="tool", name="slow_tool")
def slow_tool(x):
    time.sleep(random.random() * 0.005)  # inside the seam: recorded latency
    return {"x": x, "double": x * 2}


@frozentrace.intercept(kind="tool", name="counted")
def counted(x):
    _COUNTED["n"] += 1
    return x


@frozentrace.intercept(kind="llm", name="stream_model")
def stream_model(prompt):
    for token in prompt.split():
        yield {"token": token}


@frozentrace.intercept(kind="llm", name="astream_model")
async def astream_model(prompt):
    for token in prompt.split():
        yield {"token": token}


@frozentrace.intercept(kind="tool", name="kw_tool")
def kw_tool(**kwargs):
    return {"ok": True}


async def _concurrent_agent(topics):
    async def one(topic):
        # Jitter is OUTSIDE the seam, so it is not recorded — completion order
        # is randomised on every run.
        await asyncio.sleep(random.random() * 0.005)
        return await amodel(topic)

    results = await asyncio.gather(*(one(t) for t in topics))
    return {t: r["text"] for t, r in zip(topics, results)}


def _threaded(items):
    out = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futures = {ex.submit(slow_tool, it): it for it in items}
        for fut in as_completed(futures):
            it = futures[fut]
            out[it] = fut.result()["double"]
    return dict(sorted(out.items()))


# --------------------------------------------------------------------------- #
# basics
# --------------------------------------------------------------------------- #
def test_passthrough_without_session():
    assert model("hi")["text"] == "HI"


def test_roundtrip_sync(tmp_path):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path):
        recorded = model("hello")
    assert recorded == {"text": "HELLO", "usage": {"prompt_tokens": 3, "output_tokens": 5}}
    with frozentrace.replay(path):
        assert model("hello") == recorded


def test_replay_does_not_call_real_function(tmp_path):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path):
        counted(1)
    before = _COUNTED["n"]
    with frozentrace.replay(path):
        counted(1)
    assert _COUNTED["n"] == before  # real function never ran during replay


def test_async_roundtrip(tmp_path):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path):
        recorded = asyncio.run(amodel("x"))
    with frozentrace.replay(path):
        assert asyncio.run(amodel("x")) == recorded


def test_usage_tokens_captured(tmp_path):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path):
        model("a")
    trace = Trace.load(path)
    assert trace.total_input_tokens == 3
    assert trace.total_output_tokens == 5
    assert trace.total_tokens == 8


# --------------------------------------------------------------------------- #
# the headline guarantee: determinism under concurrency
# --------------------------------------------------------------------------- #
def test_order_independent_concurrent_async(tmp_path):
    path = str(tmp_path / "t.json")
    topics = ["a", "b", "c", "d", "e", "f"]
    with frozentrace.record(path):
        recorded = asyncio.run(_concurrent_agent(topics))
    assert recorded == {t: f"async:{t}" for t in topics}

    out = frozentrace.assert_deterministic(
        lambda: asyncio.run(_concurrent_agent(topics)), path, repeats=40
    )
    assert out == recorded


def test_order_independent_threads(tmp_path):
    path = str(tmp_path / "t.json")
    items = list(range(12))
    with frozentrace.record(path):
        recorded = _threaded(items)
    assert recorded == {i: i * 2 for i in items}

    out = frozentrace.assert_deterministic(lambda: _threaded(items), path, repeats=40)
    assert out == recorded


# --------------------------------------------------------------------------- #
# streaming
# --------------------------------------------------------------------------- #
def test_streaming_roundtrip(tmp_path):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path):
        recorded = list(stream_model("hello world"))
    assert recorded == [{"token": "hello"}, {"token": "world"}]
    with frozentrace.replay(path):
        assert list(stream_model("hello world")) == recorded


def test_async_streaming_roundtrip(tmp_path):
    path = str(tmp_path / "t.json")

    async def consume(prompt):
        return [chunk async for chunk in astream_model(prompt)]

    with frozentrace.record(path):
        recorded = asyncio.run(consume("aa bb cc"))
    assert recorded == [{"token": "aa"}, {"token": "bb"}, {"token": "cc"}]
    with frozentrace.replay(path):
        assert asyncio.run(consume("aa bb cc")) == recorded


# --------------------------------------------------------------------------- #
# strictness: loud failures, no silent drift
# --------------------------------------------------------------------------- #
def test_replay_miss(tmp_path):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path):
        model("only")
    with pytest.raises(ReplayMiss):
        with frozentrace.replay(path):
            model("something-else")


def test_strict_divergence_on_unreplayed_call(tmp_path):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path):
        model("a")
        model("b")
    with pytest.raises(DivergenceError):
        with frozentrace.replay(path, strict=True):
            model("a")  # never replays "b"


def test_non_strict_allows_partial_replay(tmp_path):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path):
        model("a")
        model("b")
    with frozentrace.replay(path):  # best-effort: no error
        assert model("a")["text"] == "A"


# --------------------------------------------------------------------------- #
# redaction
# --------------------------------------------------------------------------- #
def test_redaction_at_save(tmp_path):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path, redact=["api_key"]):
        kw_tool(api_key="s3cr3t-value", query="hi")
    raw = json.dumps(Trace.load(path).to_dict())
    assert "s3cr3t-value" not in raw
    assert "****" in raw


# --------------------------------------------------------------------------- #
# CLI + reporting
# --------------------------------------------------------------------------- #
def test_cli_inspect(tmp_path, capsys):
    path = str(tmp_path / "t.json")
    with frozentrace.record(path):
        model("a")
    assert cli_main(["inspect", path]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["steps"] == 1
    assert data["by_kind"] == {"llm": 1}


def test_cli_diff_identical(tmp_path, capsys):
    a, b = str(tmp_path / "a.json"), str(tmp_path / "b.json")
    with frozentrace.record(a):
        model("x")
    with frozentrace.record(b):
        model("x")
    assert cli_main(["diff", a, b]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["identical"] is True
