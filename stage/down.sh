#!/usr/bin/env bash
# Stop everything the demo bills for, then sweep for anything still running.
# Keeps the Artifact Registry remote repository (no charge beyond cached image storage).
set -uo pipefail
here=$(cd "$(dirname "$0")" && pwd)
PROJECT=${PROJECT:-comglitn}
TPU_PROJECT=${TPU_PROJECT:-aisprint-491218}
RUN_REGION=${RUN_REGION:-us-east4}
VERTEX_REGION=${VERTEX_REGION:-us-central1}
GKE_REGION=${GKE_REGION:-us-central1}
say() { echo "$(date -u +%H:%M:%SZ) $*"; }

if [ -f "$here/.forwards.pid" ]; then
  while read -r p; do kill "$p" 2>/dev/null; done < "$here/.forwards.pid"; rm -f "$here/.forwards.pid"
fi

gcloud compute instances list --project "$PROJECT" --filter="labels.managed-by=strands-demo" --format='value(name,zone.basename())' |
while read -r n z; do say "gce: deleting $n ($z)"; gcloud compute instances delete "$n" --project "$PROJECT" --zone "$z" --quiet; done &

# A flex-start request owns its node: deleting the request with --force deletes the node.
gcloud alpha compute tpus queued-resources list --project "$TPU_PROJECT" --zone - --filter="name~strands-demo-tpu" \
  --format='value(name.basename(),name.segment(3))' 2>/dev/null |
while read -r n z; do
  say "tpu: deleting queued resource $n ($z)"
  gcloud alpha compute tpus queued-resources delete "$n" --project "$TPU_PROJECT" --zone "$z" --force --quiet
done &
gcloud compute tpus tpu-vm list --project "$TPU_PROJECT" --zone - --filter="name=strands-demo-tpu" --format='value(name.basename(),name.segment(3))' |
while read -r n z; do say "tpu: deleting $n ($z)"; gcloud compute tpus tpu-vm delete "$n" --project "$TPU_PROJECT" --zone "$z" --quiet; done &

# strands-demo-run, plus the -eager and -cache variants from the cold-start test
for svc in $(gcloud run services list --project "$PROJECT" --region "$RUN_REGION" --filter="metadata.labels.managed-by=strands-demo" --format='value(metadata.name)'); do
  say "run: deleting $svc"; gcloud run services delete "$svc" --project "$PROJECT" --region "$RUN_REGION" --quiet &
done
if gcloud storage buckets describe "gs://$PROJECT-strands-demo-cache" >/dev/null 2>&1; then
  say "run: deleting gs://$PROJECT-strands-demo-cache"; gcloud storage rm -r "gs://$PROJECT-strands-demo-cache" --quiet >/dev/null 2>&1 &
fi

if gcloud container clusters describe strands-demo-gke --project "$PROJECT" --region "$GKE_REGION" >/dev/null 2>&1; then
  say "gke: deleting cluster strands-demo-gke"; gcloud container clusters delete strands-demo-gke --project "$PROJECT" --region "$GKE_REGION" --quiet &
fi

(
  for ep in $(gcloud ai endpoints list --project "$PROJECT" --region "$VERTEX_REGION" --filter="displayName=strands-demo-vertex" --format='value(name.basename())' 2>/dev/null); do
    for dm in $(gcloud ai endpoints describe "$ep" --project "$PROJECT" --region "$VERTEX_REGION" --format='value(deployedModels[].id)' 2>/dev/null | tr ';' ' '); do
      say "vertex: undeploying $dm from $ep"
      gcloud ai endpoints undeploy-model "$ep" --project "$PROJECT" --region "$VERTEX_REGION" --deployed-model-id "$dm" --quiet
    done
    say "vertex: deleting endpoint $ep"; gcloud ai endpoints delete "$ep" --project "$PROJECT" --region "$VERTEX_REGION" --quiet
  done
  for m in $(gcloud ai models list --project "$PROJECT" --region "$VERTEX_REGION" --filter="displayName=strands-demo-vertex" --format='value(name.basename())' 2>/dev/null); do
    say "vertex: deleting model $m"; gcloud ai models delete "$m" --project "$PROJECT" --region "$VERTEX_REGION" --quiet
  done
) &
wait

echo; echo "Sweep (anything listed here is still billing):"
gcloud compute instances list --project "$PROJECT" --filter="labels.managed-by=strands-demo OR guestAccelerators:* OR machineType~g2-" --format='value(name,zone.basename(),status)' | sed 's/^/  gce /'
gcloud compute tpus tpu-vm list --project "$TPU_PROJECT" --zone - --format='value(name.basename(),state)' 2>/dev/null | sed 's/^/  tpu /'
gcloud alpha compute tpus queued-resources list --project "$TPU_PROJECT" --zone - --format='value(name.basename(),state.state)' 2>/dev/null | sed 's/^/  tpu-queued /'
gcloud run services list --project "$PROJECT" --filter="metadata.labels.managed-by=strands-demo" --format='value(metadata.name,region)' | sed 's/^/  run /'
gcloud storage buckets list --project "$PROJECT" --filter="name~strands-demo" --format='value(name)' | sed 's/^/  bucket /'
gcloud container clusters list --project "$PROJECT" --format='value(name,location,status)' | sed 's/^/  gke /'
gcloud ai endpoints list --project "$PROJECT" --region "$VERTEX_REGION" --format='value(displayName,name.basename())' 2>/dev/null | sed 's/^/  vertex /'
echo "(sweep done)"
