# frozentrace

> **Deterministic record & replay for AI agent runs — stable even under concurrency and streaming.**

Testing an AI agent is painful. Live model calls are **expensive** (every test run costs money), **non-deterministic** (a different answer each time) and **slow** (seconds per call). The usual fix is record & replay: capture a real run once, replay it forever as a mock.

The problem is that most record & replay tools — the HTTP VCR family, and most agent cassettes — match calls by **position**: the *n-th* intercepted call gets the *n-th* recorded result. That holds together right up until your agent does anything interesting. The moment it calls tools **concurrently** (`asyncio.gather`, a thread pool) the completion order changes from run to run, the recorded positions no longer line up, and replay silently drifts.

frozentrace matches calls by a **stable signature** instead of by position. The same call — same name, same arguments — gets the same result no matter *when* it happens, so replay is identical even when the agent's scheduling is not. Zero dependencies, stdlib only.

```python
import frozentrace

call_model = frozentrace.intercept(call_model, kind="llm")

# Record a real run once:
with frozentrace.record("traces/flight.json"):
    my_agent.run("Find flights to NYC under $300")

# Replay it forever — no network, no tokens, same result every time:
with frozentrace.replay("traces/flight.json"):
    my_agent.run("Find flights to NYC under $300")
```

## Why not just use a VCR?

|                                             | HTTP VCR (`vcrpy`…)| positional agent cassette | **frozentrace** |
|---------------------------------------------|:------------------:|:-------------------------:|:------------:|
| matched by                                  | position           | position                  | **signature**|
| deterministic under concurrency             | no                 | best-effort (documented)  | **yes**      |
| streaming responses                         | partial            | not handled               | **yes**      |
| async generators                            | no                 | no                        | **yes**      |
| runtime dependencies                        | varies             | zero                      | **zero**     |

Positional cassettes are honest about the gap — they recommend you "record such sections sequentially if you need strict determinism." frozentrace closes it.

## Install

```bash
pip install frozentrace
```

Requires Python 3.9+. No runtime dependencies, ever.

## Quickstart

### 1. Wrap the seams you want captured

Wrap your model call (and any tools). Outside a record/replay block the wrapper is a **transparent pass-through**, so this is safe to leave in production code.

```python
import frozentrace

@frozentrace.intercept(kind="llm")
def call_model(prompt):
    return openai_client.chat.completions.create(...)

@frozentrace.intercept(kind="tool")     # async works too
async def search_web(query):
    ...
```

### 2. Record

```python
with frozentrace.record("traces/run.json", model="gpt-4o"):
    my_agent.run(task)
```

### 3. Replay

```python
with frozentrace.replay("traces/run.json"):
    my_agent.run(task)
```

During replay an intercepted call returns its recorded result and the real function is **never called**.

## The guarantee: determinism under concurrency

This is what frozentrace is for. Given an agent that fires its calls concurrently, replay must produce the same answer no matter how the calls interleave:

```python
import asyncio, random, frozentrace

@frozentrace.intercept(kind="llm")
async def call_model(prompt): ...

async def agent(topics):
    async def one(t):
        await asyncio.sleep(random.random() * 0.01)   # schedule jitter, NOT recorded
        return await call_model(t)
    results = await asyncio.gather(*(one(t) for t in topics))
    return {t: r["text"] for t, r in zip(topics, results)}
```

Record once, then **prove** it replays identically, every time, under a different schedule:

```python
frozentrace.assert_deterministic(lambda: asyncio.run(agent(topics)), "traces/run.json", repeats=50)
```

or from the command line, over a script that opts into replay via `frozentrace.replay_from_env()`:

```console
$ frozentrace determinism traces/run.json --runs 50 -- python examples/agent_demo.py
runs=50 distinct_outputs=1 deterministic=yes
```

`distinct_outputs=1` across 50 differently-scheduled runs is the whole point.

## How it works

Each intercepted call is reduced to a **signature**: `sha1(kind, name, canonical(args), canonical(kwargs), key)`. `canonical` produces deterministic JSON (sorted keys, memory addresses scrubbed from reprs). On record, results are stored under their signature. On replay, a call computes its own signature and is served the recorded result — **independent of when it runs**. Repeated identical calls are disambiguated by a per-signature occurrence counter.

The one case a signature cannot resolve on its own is a *concurrent* call that is genuinely identical but legitimately returns *different* results. For that, pass a stable `key`:

```python
@frozentrace.intercept(kind="tool", key=lambda user: user.id)
def fetch_profile(user):
    ...
```

## Streaming, sync and async

Streamed responses are captured as their chunks and replayed as an iterator, so your `for chunk in ...` code is unchanged. `async def` coroutines and `async for` async generators are both supported.

```python
@frozentrace.intercept(kind="llm", name="stream_model")
def stream_model(prompt):
    for token in ...:
        yield {"token": token}
```

## Fail loudly, never drift

Replay never invents data. A call with no recorded match raises `ReplayMiss` — never a silent `None`, never an accidental network call. In `strict=True` mode, replay additionally fails with `DivergenceError` if the agent took a path that left recorded calls unreplayed, turning a trace into a behavioural contract:

```python
with frozentrace.replay("traces/run.json", strict=True):
    my_agent.run(task)          # raises if the agent drifted from the recording
```

## Redacting secrets

```python
with frozentrace.record("traces/run.json", redact=["api_key", "authorization"]):
    my_agent.run(task)
```

Keys are scrubbed at any depth before the trace is written.

## CLI

```console
frozentrace inspect     traces/run.json                     # steps, tokens, duration
frozentrace diff        traces/v1.json traces/v2.json       # what changed between runs
frozentrace determinism traces/run.json --runs 50 -- python agent.py
```

## Known limitations

- Replayed values round-trip through JSON, so you get dicts/lists/primitives back, not the original SDK objects. For typical LLM responses this is exactly what you want.
- A *concurrent* call with an identical signature but genuinely different results needs an explicit `key=` (see above); everything else is order-independent by construction.
- Traces are single-threaded JSON; very large runs (>~10k calls) would want a streamed or SQLite backend.

## Roadmap

- Deterministic clock and `random` seams (freeze time / seed RNG during replay).
- Recorded HTTP transport layer, so un-instrumented clients (`requests`, `httpx`) are captured too.
- Trace export to OpenTelemetry.

## License

MIT — see [LICENSE](LICENSE).
