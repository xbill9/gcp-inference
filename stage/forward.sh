#!/usr/bin/env bash
# Reach every backend that is up and write stage/demo.env for the demos.
#
#   gce   IAP TCP tunnel to port 8000 (the VM takes no inbound traffic from the internet)
#   gke   kubectl port-forward to the vLLM Deployment
#   tpu   SSH local forward through the TPU VM
#   vertex, cloudrun, maas, gemini   Google APIs and HTTPS; nothing to forward
#
# Local ports: gce 8001, gke 8003, tpu 8005.
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PROJECT=${PROJECT:-comglitn}
TPU_PROJECT=${TPU_PROJECT:-aisprint-491218}
RUN_REGION=${RUN_REGION:-us-east4}
VERTEX_REGION=${VERTEX_REGION:-us-central1}
GKE_REGION=${GKE_REGION:-us-central1}
pidfile=$here/.forwards.pid
env=$here/demo.env

if [ -f "$pidfile" ]; then while read -r p; do kill "$p" 2>/dev/null; done < "$pidfile"; fi
: > "$pidfile"
{
  echo "export GOOGLE_CLOUD_PROJECT=$PROJECT"
  echo "export GEMINI_MODEL=\${GEMINI_MODEL:-gemini-3.5-flash-lite}"
  echo "export MAAS_MODEL=\${MAAS_MODEL:-google/gemma-4-26b-a4b-it-maas}"
} > "$env"

zone=$(gcloud compute instances list --project "$PROJECT" --filter="name=gcp-inference-gce" --format='value(zone.basename())')
if [ -n "$zone" ]; then
  gcloud compute start-iap-tunnel gcp-inference-gce 8000 --local-host-port=localhost:8001 \
    --project "$PROJECT" --zone "$zone" > "$here/.fwd-gce.log" 2>&1 &
  echo $! >> "$pidfile"; echo "export GCE_URL=http://localhost:8001/v1" >> "$env"; echo "gce: tunnel from $zone"
fi

if gcloud container clusters describe gcp-inference-gke --project "$PROJECT" --region "$GKE_REGION" >/dev/null 2>&1; then
  gcloud container clusters get-credentials gcp-inference-gke --project "$PROJECT" --region "$GKE_REGION" >/dev/null 2>&1
  kubectl port-forward deploy/vllm 8003:8000 > "$here/.fwd-gke.log" 2>&1 &
  echo $! >> "$pidfile"; echo "export GKE_URL=http://localhost:8003/v1" >> "$env"; echo "gke: port-forward"
fi

# An on-demand node is gcp-inference-tpu; a flex-start node is gcp-inference-tpu-node.
read -r tname tzone < <(gcloud compute tpus tpu-vm list --project "$TPU_PROJECT" --zone - \
  --filter="name~gcp-inference-tpu AND state=READY" --format='value(name.basename(),name.segment(3))' 2>/dev/null | head -1)
if [ -n "${tzone:-}" ]; then
  gcloud compute tpus tpu-vm ssh "$tname" --project "$TPU_PROJECT" --zone "$tzone" \
    -- -N -L 8005:localhost:8000 -o ExitOnForwardFailure=yes -o ServerAliveInterval=30 > "$here/.fwd-tpu.log" 2>&1 &
  echo $! >> "$pidfile"; echo "export TPU_URL=http://localhost:8005/v1" >> "$env"; echo "tpu: ssh forward from $tzone"
fi

ep=$(gcloud ai endpoints list --project "$PROJECT" --region "$VERTEX_REGION" \
     --filter="displayName=gcp-inference-vertex" --format='value(name)' 2>/dev/null | head -1)
if [ -n "$ep" ]; then
  echo "export VERTEX_URL=https://$VERTEX_REGION-aiplatform.googleapis.com/v1/projects/$PROJECT/locations/$VERTEX_REGION/endpoints/${ep##*/}" >> "$env"
  echo "vertex: endpoint ${ep##*/}"
fi

url=$(gcloud run services describe gcp-inference-run --project "$PROJECT" --region "$RUN_REGION" --format='value(status.url)' 2>/dev/null)
if [ -n "$url" ]; then echo "export RUN_URL=$url/v1" >> "$env"; echo "cloudrun: $url"; fi

echo "wrote $env"
