"""A tiny demo agent for the README and for `agentvcr determinism`.

Record a run:

    python examples/agent_demo.py --record demo.json

Then prove replay is deterministic no matter how the concurrent calls finish:

    agentvcr determinism demo.json -- python examples/agent_demo.py
"""

from __future__ import annotations

import argparse
import asyncio
import random

import agentvcr


@agentvcr.intercept(kind="llm", name="call_model")
async def call_model(prompt):
    return {
        "text": f"answer: {prompt}",
        "usage": {"prompt_tokens": 4, "output_tokens": 6},
    }


async def agent(topics):
    async def one(topic):
        # Schedule jitter lives OUTSIDE the intercepted seam, so it is not
        # recorded — this randomises completion order across runs.
        await asyncio.sleep(random.random() * 0.01)
        return await call_model(topic)

    results = await asyncio.gather(*(one(t) for t in topics))
    return {t: r["text"] for t, r in zip(topics, results)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--record", metavar="TRACE", help="record a run into this trace file")
    args = parser.parse_args()

    topics = ["math", "geography", "history"]
    if args.record:
        with agentvcr.record(args.record):
            asyncio.run(agent(topics))
        print(f"recorded -> {args.record}")
        return

    # Under `agentvcr determinism` this reads AGENTVCR_REPLAY; otherwise it is a
    # no-op and the real functions run.
    with agentvcr.replay_from_env():
        result = asyncio.run(agent(topics))
    print(result)


if __name__ == "__main__":
    main()
