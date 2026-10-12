"""Tables for the README and the deck, computed from one results directory's JSON files.

    python3 results/summarize.py results/2026-10-12 > results/2026-10-12/comparison.txt

Reads demo1-engine-<b>.json, demo1-rows-<b>.json and bench-<b>.json for every backend present.
Every number printed is counted, sorted or taken as a median here.
"""

import glob
import json
import os
import statistics
import sys

ORDER = ["gemini", "maas", "vertex", "cloudrun", "gke", "gce", "tpu"]


def load(d, kind):
    out = {}
    for path in glob.glob(os.path.join(d, f"{kind}-*.json")):
        name = os.path.basename(path)[len(kind) + 1:-5]
        out[name] = json.load(open(path))
    return dict(sorted(out.items(), key=lambda kv: ORDER.index(kv[0]) if kv[0] in ORDER else 99))


def main():
    d = sys.argv[1]
    engine, rows, bench = load(d, "demo1-engine"), load(d, "demo1-rows"), load(d, "bench")

    print("Demo 1: how many of 11 ids are 10 or more? (answer 8)\n")
    print(f"{'backend':10s} {'engine correct':>15s} {'right filter':>13s} {'errors':>7s} {'median s':>9s}"
          f" {'rows correct':>13s} {'median s':>9s}  filters sent")
    for name, e in engine.items():
        s = e["summary"][0]
        r = rows.get(name, {}).get("summary", [{}])[0]
        print(f"{name:10s} {s['correct']:>8d}/{s['runs']:<6d} {s['filter_ok']:>6d}/{s['runs']:<6d} {s['errors']:>7d}"
              f" {s['median_seconds']:>9}"
              f" {r.get('correct', '-'):>6}/{r.get('runs', '-'):<6} {r.get('median_seconds', '-'):>9}"
              f"  {', '.join(s['filters_sent']) or '-'}")

    print("\nSame prompt, 100-token limit, one request at a time (bench_chat.py)\n")
    print(f"{'backend':10s} {'median s':>9s} {'median tok/s':>13s} {'tok/s range':>14s} {'tokens':>7s}")
    ranked = sorted(((n, b["summary"][0]) for n, b in bench.items()),
                    key=lambda kv: -kv[1]["median_tokens_per_second"])
    for name, s in ranked:
        print(f"{name:10s} {s['median_seconds']:>9} {s['median_tokens_per_second']:>13} "
              f"{s['min_tps']:>6}-{s['max_tps']:<7} {s['tokens']:>7}")

    # Agent-run latency across both tool sets, all runs pooled per backend
    print("\nDemo 1 seconds per agent run, both tool sets pooled\n")
    print(f"{'backend':10s} {'runs':>5s} {'median s':>9s} {'max s':>7s}")
    for name in engine:
        secs = [r["seconds"] for r in engine[name]["runs"]] + [r["seconds"] for r in rows.get(name, {}).get("runs", [])]
        print(f"{name:10s} {len(secs):>5d} {statistics.median(secs):>9.2f} {max(secs):>7.2f}")


if __name__ == "__main__":
    main()
