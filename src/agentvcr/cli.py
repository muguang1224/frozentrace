"""agentvcr command line interface.

    agentvcr inspect      TRACE          summarise a recorded run
    agentvcr diff         A B            compare two runs
    agentvcr determinism  TRACE [--runs N] -- CMD
                                         replay CMD N times and check it is identical
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys

from . import __version__
from .trace import Trace, diff_traces, summarize


def cmd_inspect(args) -> int:
    trace = Trace.load(args.trace)
    print(json.dumps(summarize(trace), indent=2, ensure_ascii=False))
    return 0


def cmd_diff(args) -> int:
    result = diff_traces(Trace.load(args.a), Trace.load(args.b))
    print(json.dumps(result, indent=2, ensure_ascii=False))
    return 0 if result["identical"] else 1


def cmd_determinism(args) -> int:
    command = list(getattr(args, "command", []) or [])
    if command and command[0] == "--":
        command = command[1:]
    if not command:
        print(
            "error: no command given. Usage: agentvcr determinism TRACE -- python agent.py",
            file=sys.stderr,
        )
        return 2

    env = dict(os.environ)
    env["AGENTVCR_REPLAY"] = os.path.abspath(args.trace)
    if args.strict:
        env["AGENTVCR_STRICT"] = "1"

    outputs = []
    for i in range(args.runs):
        proc = subprocess.run(command, capture_output=True, text=True, env=env)
        if proc.returncode != 0:
            print(f"run #{i} exited {proc.returncode}; stderr:", file=sys.stderr)
            sys.stderr.write(proc.stderr)
            return 1
        outputs.append(proc.stdout)

    distinct = sorted(set(outputs))
    deterministic = len(distinct) == 1
    verdict = "yes" if deterministic else "NO"
    print(f"runs={args.runs} distinct_outputs={len(distinct)} deterministic={verdict}")
    if not deterministic:
        print("\nfirst differing outputs:")
        for i, out in enumerate(outputs):
            if out != outputs[0]:
                print(f"--- run #0 ---\n{outputs[0]}\n--- run #{i} ---\n{out}")
                break
        return 1
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="agentvcr", description="Deterministic record & replay for AI agents."
    )
    parser.add_argument("--version", action="version", version=f"agentvcr {__version__}")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect", help="summarise a recorded run")
    p.add_argument("trace")
    p.set_defaults(func=cmd_inspect)

    p = sub.add_parser("diff", help="compare two recorded runs")
    p.add_argument("a")
    p.add_argument("b")
    p.set_defaults(func=cmd_diff)

    p = sub.add_parser("determinism", help="replay a command N times and check it is identical")
    p.add_argument("trace")
    p.add_argument("--runs", type=int, default=10)
    p.add_argument("--strict", action="store_true")
    p.set_defaults(func=cmd_determinism, command=[])

    return parser


def main(argv=None) -> int:
    raw = list(sys.argv[1:] if argv is None else argv)

    # Everything after a standalone `--` is the sub-command to replay.
    command = None
    if "--" in raw:
        idx = raw.index("--")
        command = raw[idx + 1:]
        raw = raw[:idx]

    args = build_parser().parse_args(raw)
    if command is not None and getattr(args, "func", None) is cmd_determinism:
        args.command = command
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
