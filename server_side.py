"""Server-side decode speed, read from each vLLM server's own /metrics, so the network is left out.

bench_chat.py times requests from the client: that number includes the path to the backend (an IAP
tunnel, a kubectl port-forward, Cloud Run's front end). This script sends the same prompt, reads
vllm:inter_token_latency_seconds before and after, and reports tokens per second of decode as the
server measured it: (change in count) / (change in sum).

    python3 server_side.py --repeats 5 --json results/2026-10-12/server-side.json

Covers the self-hosted vLLM backends whose /metrics is reachable: gce, gke, cloudrun, tpu. The Vertex
AI endpoint exposes only its predict route, so it is skipped.
"""

import argparse
import json
import re
import sys
import urllib.request

import backends
from bench_chat import PROMPT

METRIC = "vllm:inter_token_latency_seconds"


def scrape(b):
    url = b.url.rstrip("/").removesuffix("/v1") + "/metrics"
    headers = {"authorization": f"Bearer {backends.token(b.auth)}"} if b.auth != "none" else {}
    text = urllib.request.urlopen(urllib.request.Request(url, headers=headers), timeout=60).read().decode()
    vals = {}
    for kind in ("sum", "count"):
        m = re.search(rf"^{re.escape(METRIC)}_{kind}\{{[^}}]*\}} ([0-9.e+]+)$", text, re.M)
        vals[kind] = float(m.group(1))
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repeats", type=int, default=5)
    ap.add_argument("--max-tokens", type=int, default=100)
    ap.add_argument("--json")
    a = ap.parse_args()
    rows = []
    for b in backends.configured():
        if b.kind != "openai" or b.name in ("maas", "vertex"):
            continue
        backends.chat_once(b, "Hi", max_tokens=4)  # warm the path; not counted
        before = scrape(b)
        client = [backends.chat_once(b, PROMPT, max_tokens=a.max_tokens) for _ in range(a.repeats)]
        after = scrape(b)
        tokens = after["count"] - before["count"]
        seconds = after["sum"] - before["sum"]
        client_tps = sorted(r["tokens_per_second"] for r in client)[len(client) // 2]
        rows.append({"backend": b.name, "inter_token_intervals": tokens, "decode_seconds": round(seconds, 4),
                     "server_decode_tok_s": round(tokens / seconds, 1), "client_median_tok_s": client_tps})
    print(f"Prompt: {PROMPT!r}, max_tokens={a.max_tokens}, {a.repeats} requests per backend, one at a time\n")
    print(f"{'backend':10s} {'server decode tok/s':>20s} {'client median tok/s':>20s}")
    for r in sorted(rows, key=lambda r: -r["server_decode_tok_s"]):
        print(f"{r['backend']:10s} {r['server_decode_tok_s']:>20} {r['client_median_tok_s']:>20}")
    if a.json:
        json.dump({"prompt": PROMPT, "metric": METRIC, "rows": rows}, open(a.json, "w"), indent=2)
    return 0


if __name__ == "__main__":
    sys.exit(main())
