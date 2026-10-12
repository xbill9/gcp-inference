"""Aggregate decode throughput with 1, 4 and 16 requests in flight, timed here.

    python3 parallel.py --json results/2026-10-12/parallel.json

For each backend and each level N, N threads each send the bench prompt (256-token limit) three
times back to back. Aggregate tok/s = completion tokens returned / wall seconds for the level, so
it includes the network path. Every number is summed or divided here.
"""

import argparse
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor

import backends
from bench_chat import PROMPT


def level(b, n, rounds, max_tokens):
    def worker(_):
        got = 0
        for _ in range(rounds):
            got += backends.chat_once(b, PROMPT, max_tokens=max_tokens)["completion_tokens"]
        return got
    t0 = time.perf_counter()
    with ThreadPoolExecutor(n) as pool:
        tokens = sum(pool.map(worker, range(n)))
    wall = time.perf_counter() - t0
    return {"backend": b.name, "parallel": n, "tokens": tokens, "seconds": round(wall, 2),
            "aggregate_tok_s": round(tokens / wall, 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--levels", default="1,4,16")
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = []
    for b in backends.configured():
        backends.chat_once(b, "Hi", max_tokens=4)
        for n in (int(x) for x in a.levels.split(",")):
            rows.append(level(b, n, a.rounds, a.max_tokens))
            r = rows[-1]
            print(f"{r['backend']:10s} parallel {r['parallel']:>3d}  {r['tokens']:>6d} tokens in {r['seconds']:>7}s"
                  f"  = {r['aggregate_tok_s']:>7} tok/s", flush=True)
    if a.json:
        json.dump({"prompt": PROMPT, "max_tokens": a.max_tokens, "rounds": a.rounds, "rows": rows},
                  open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
