#!/usr/bin/env bash
# One backend, the full measurement set, as recorded in this directory:
#   results/2026-10-12/run.sh <backend>
# demo 1 with the engine tool and with raw rows (20 runs each), then bench_chat (5 repeats).
set -uo pipefail
b=$1; d=$(cd "$(dirname "$0")" && pwd); cd "$d/../.."
source stage/demo.env
export BACKENDS=$b
python3 demo1_counting.py --runs 20 --json "$d/demo1-engine-$b.json" > "$d/demo1-engine-$b.txt" 2>&1
python3 demo1_counting.py --runs 20 --tools rows --json "$d/demo1-rows-$b.json" > "$d/demo1-rows-$b.txt" 2>&1
python3 bench_chat.py --repeats 5 --json "$d/bench-$b.json" > "$d/bench-$b.txt" 2>&1
echo "$(date -u +%H:%M:%SZ) $b done"
