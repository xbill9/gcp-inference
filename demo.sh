#!/usr/bin/env bash
# The whole demo in one command: a TPU v5e-1, Gemma 4 E2B on vLLM, the MCP server, demo 1.
#
#   ./demo.sh          # start the TPU (billing starts), wait for vLLM, run the demo over MCP
#   ./demo.sh again    # run the demo again on the TPU that is already up
#   ./demo.sh down     # delete the TPU and everything else labelled managed-by=gcp-inference
#
# The TPU is a flex-start request, so it also stops by itself after 4 hours.
set -uo pipefail
cd "$(dirname "$0")"
TPU_PROJECT=${TPU_PROJECT:-aisprint-491218}
say() { echo; echo "== [$(elapsed)] $*"; }
elapsed() { printf '%dm%02ds' $((SECONDS / 60)) $((SECONDS % 60)); }
die() { echo "!! $*" >&2; exit 1; }

node() {   # name and zone of the READY TPU VM, if any
  gcloud compute tpus tpu-vm list --project "$TPU_PROJECT" --zone - \
    --filter="name~gcp-inference-tpu AND state=READY" --format='value(name.basename(),name.segment(3))' 2>/dev/null | head -1
}

case "${1:-up}" in
  down) exec stage/down.sh ;;
  again) ;;
  up)
    say "1/5  Asking Google Cloud for a TPU v5e-1 (flex-start, us-west4-a)"
    stage/up.sh tpu | grep -v -e '^Next:' -e '^$' | sed 's/^/   /' || die "the TPU request failed; see the line above"

    say "2/5  Waiting for the TPU to be handed over"
    for _ in $(seq 120); do
      [ -n "$(node)" ] && break
      st=$(gcloud alpha compute tpus queued-resources list --project "$TPU_PROJECT" --zone - \
        --filter="name~gcp-inference-tpu" --format='value(state.state)' 2>/dev/null | head -1)
      case "$st" in FAILED|SUSPENDED) die "the TPU request ended in state $st; run ./demo.sh down and try again" ;; esac
      case "$st" in
        WAITING_FOR_RESOURCES) st="queued, waiting for a free TPU" ;;
        PROVISIONING) st="TPU found, Google is setting it up" ;;
        ""|ACCEPTED|CREATING) st="request accepted" ;;
      esac
      echo "   [$(elapsed)] $st"; sleep 15
    done
    [ -n "$(node)" ] || die "no TPU after 30 minutes; run ./demo.sh down"
    ;;
  *) die "usage: ./demo.sh [up|again|down]" ;;
esac

read -r tname tzone < <(node)
[ -n "${tzone:-}" ] || die "no TPU is up; run ./demo.sh first"
echo "   TPU $tname in $tzone"

say "3/5  Connecting to it over SSH (the demo reaches vLLM on the TPU's port 8000 through a tunnel)"
for i in $(seq 10); do
  gcloud compute tpus tpu-vm ssh "$tname" --project "$TPU_PROJECT" --zone "$tzone" --quiet --command true \
    >/dev/null 2>&1 && { echo "   [$(elapsed)] SSH works"; break; }
  echo "   [$(elapsed)] SSH not ready yet (try $i of 10)"; sleep 10
done
stage/forward.sh >/dev/null || die "stage/forward.sh failed"
echo "   [$(elapsed)] tunnel open: localhost:8005 -> TPU port 8000"
source stage/demo.env
export BACKENDS=tpu
[ -n "${TPU_URL:-}" ] || die "no SSH forward to the TPU; see stage/.fwd-tpu.log"

say "4/5  Starting Gemma on vLLM (first start downloads and compiles: several minutes)"
echo "   each line is the newest line of vLLM's own log on the TPU"
t0=$SECONDS
until curl -sf -m 5 "$TPU_URL/models" >/dev/null; do
  (( SECONDS - t0 > 1800 )) && die "vLLM did not answer within 30 minutes; ask Claude for backend_logs tpu"
  line=$(gcloud compute tpus tpu-vm ssh "$tname" --project "$TPU_PROJECT" --zone "$tzone" --quiet \
    --command 'sudo docker logs --tail 1 vllm 2>&1 || echo "downloading the vLLM container image"' 2>/dev/null |
    tr '\r' '\n' | grep -v '^\s*$' | tail -1 | sed 's/^.*\] //' | cut -c1-100)
  echo "   [$(elapsed)] ${line:-waiting for the TPU to start the container}"
  sleep 15
done
echo "   [$(elapsed)] vLLM is answering"
python3 stage/check.py || die "vLLM answers but the check failed; see the line above"

say "5/5  The demo: Gemma on the TPU counts ids with the count_ids tool from the MCP server"
python3 demo1_counting.py --tools mcp --runs 3

cat <<EOF

Done in $(elapsed). The TPU is still running and billing.
  Again:              ./demo.sh again
  Talk to it:         run 'claude' in $(pwd) and approve the tpu-demo MCP server,
                      then ask "list my backends" or "ask the tpu backend for a haiku"
  Shut it all down:   ./demo.sh down
EOF
