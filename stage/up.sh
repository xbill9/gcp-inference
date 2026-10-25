#!/usr/bin/env bash
# Launch the self-hosted backends. Every one starts billing; stage/down.sh stops all of them.
#
#   stage/up.sh                    # all five: gce vertex gke run tpu
#   stage/up.sh gce run            # only some
#
# Gemini and Gemma MaaS on Vertex AI need nothing launched: they are billed per token.
#
# Every GPU backend serves the same checkpoint with the same vLLM image and flags as
# the AWS run (~/aws-inference-strands): the E2B QAT w4a16 repack, max-model-len 8192,
# tool calls on. The TPU serves google/gemma-4-E2B-it in bf16 with vLLM's TPU image, at the
# model's full 131072-token context: its KV cache holds 321,280 tokens, 2.45 requests that long.
#
# Resources carry the label managed-by=gcp-inference. One that already exists is left alone.
# Each launch line is stamped with UTC time; stage/check.py stamps the first pass.
set -uo pipefail

PROJECT=${PROJECT:-comglitn}
TPU_PROJECT=${TPU_PROJECT:-aisprint-491218}    # holds the TPU API quota
# v5e-1 ("v5litepod-1" to the TPU API), straight to a flex-start request in us-west4-a: on
# 2026-10-12 v6e-1 was refused in every zone and a v5e-1 flex-start request got a node in 22 s.
# Flex-start stops by itself after 4 hours. For v6e on-demand first:
#   TPU_ACCEL=v6e-1 TPU_RUNTIME=v2-alpha-tpuv6e TPU_ZONES="us-east5-b us-east5-a" TPU_FLEX_ZONE=us-east5-b
TPU_ACCEL=${TPU_ACCEL:-v5litepod-1}
TPU_RUNTIME=${TPU_RUNTIME:-v2-alpha-tpuv5-lite}
TPU_ZONES=${TPU_ZONES:-}                       # on-demand zones to try before flex-start
TPU_FLEX_ZONE=${TPU_FLEX_ZONE:-us-west4-a}
GCE_ZONES=${GCE_ZONES:-us-east4-a us-east4-c us-east4-b us-central1-a us-central1-b us-central1-c}
RUN_REGION=${RUN_REGION:-us-east4}
VERTEX_REGION=${VERTEX_REGION:-us-central1}
GKE_REGION=${GKE_REGION:-us-central1}
GKE_NETWORK=${GKE_NETWORK:-vpc-glitnir}   # the default network in this project is legacy, which GKE cannot use

REPACK=${REPACK:-xbill9/gemma-4-E2B-it-qat-q4_0-w4a16-ct-text-emb4}
TPU_MODEL=${TPU_MODEL_ID:-google/gemma-4-E2B-it}
VLLM_IMAGE=${VLLM_IMAGE:-vllm/vllm-openai:v0.30.0}   # the image the AWS run used
# vllm/vllm-openai:v0.30.0-cu129 dies at import on every platform here ("operator torchvision::nms
# does not exist"); the default CUDA 13 build serves. Its forward-compat libraries stay off: with them
# on, Cloud Run and GKE Autopilot fail at cudaGetDeviceCount with error 803. GKE Autopilot's default
# driver measured 580.173.02 and the Deep Learning VM's 580.178.04, both new enough for CUDA 13.
COMPAT=${COMPAT:-VLLM_ENABLE_CUDA_COMPATIBILITY=0}
TPU_IMAGE=${TPU_IMAGE:-vllm/vllm-tpu:v0.30.0}
VLLM_ARGS="--model $REPACK --served-model-name gemma --max-model-len 8192 --gpu-memory-utilization 0.90 --enable-auto-tool-choice --tool-call-parser gemma4"
LABEL=managed-by=gcp-inference
here=$(cd "$(dirname "$0")" && pwd)

say() { echo "$(date -u +%H:%M:%SZ) $*"; }
fail() { say "$*"; return 1; }

gcloud auth print-access-token >/dev/null 2>&1 || { echo "gcloud is not logged in: run gcloud auth login"; exit 1; }

gce() {
  local name=${GCE_NAME:-gcp-inference-gce} zone out ud
  if gcloud compute instances list --project "$PROJECT" --filter="name=$name" --format='value(zone)' | grep -q .; then
    say "gce: already up"; return 0
  fi
  ud=$(mktemp)
  cat > "$ud" <<UD
#!/usr/bin/env bash
set -x
until nvidia-smi >/dev/null 2>&1; do sleep 5; done
# The Deep Learning VM image has the driver and the NVIDIA container toolkit, but no Docker.
# First boot races the image's own package setup for the dpkg lock: wait for it.
until command -v docker >/dev/null; do
  apt-get -o DPkg::Lock::Timeout=600 update && apt-get -o DPkg::Lock::Timeout=600 install -y docker.io || sleep 15
done
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
docker inspect vllm >/dev/null 2>&1 && exit 0
docker run -d --name vllm --restart unless-stopped --gpus all --ipc=host -p 8000:8000 $VLLM_IMAGE $VLLM_ARGS
UD
  for zone in $GCE_ZONES; do
    out=$(gcloud compute instances create "$name" --project "$PROJECT" --zone "$zone" \
      --machine-type g2-standard-4 --maintenance-policy TERMINATE \
      --image-project deeplearning-platform-release --image-family common-cu129-ubuntu-2204-nvidia-580 \
      --boot-disk-size 100GB --boot-disk-type pd-balanced \
      --metadata install-nvidia-driver=True --metadata-from-file startup-script="$ud" \
      --labels "$LABEL" --shielded-secure-boot --format='value(name,zone)' 2>&1)
    case "$out" in
      *"$name"*) say "gce: launched in $zone"; rm -f "$ud"; return 0 ;;
      *ZONE_RESOURCE_POOL_EXHAUSTED*|*"does not have enough resources"*|*stockout*|*"does not exist in zone"*) say "gce: no capacity in $zone, trying the next zone" ;;
      *) fail "gce: $zone: $(echo "$out" | grep -v '^$' | tail -2)"; rm -f "$ud"; return 1 ;;
    esac
  done
  rm -f "$ud"; say "gce: no capacity in any of: $GCE_ZONES"; return 1
}

cloudrun() {
  local name=gcp-inference-run
  # A service whose last deploy failed is redeployed; one that is Ready is left alone.
  if [ "$(gcloud run services describe "$name" --project "$PROJECT" --region "$RUN_REGION" \
          --format='value(status.conditions[0].status)' 2>/dev/null)" = True ]; then
    say "run: already deployed"; return 0
  fi
  say "run: deploying $name in $RUN_REGION"
  gcloud beta run deploy "$name" --project "$PROJECT" --region "$RUN_REGION" --image "$VLLM_IMAGE" \
    --gpu 1 --gpu-type nvidia-l4 --no-gpu-zonal-redundancy --no-cpu-throttling \
    --cpu 4 --memory 16Gi --port 8000 --concurrency 4 --timeout 3600 \
    --min-instances 0 --max-instances 1 --execution-environment gen2 \
    --startup-probe=timeoutSeconds=60,periodSeconds=60,failureThreshold=10,initialDelaySeconds=60,httpGet.port=8000,httpGet.path=/health \
    --args="$(echo "$VLLM_ARGS" | tr ' ' ',')" \
    --set-env-vars "$COMPAT" \
    --no-allow-unauthenticated --labels "$LABEL" 2>&1 | tail -3
  [ "${PIPESTATUS[0]}" = 0 ] || { fail "run: deploy failed"; return 1; }
  say "run: deployed"
}

vertex() {
  local ep model repo=dockerhub
  ep=$(gcloud ai endpoints list --project "$PROJECT" --region "$VERTEX_REGION" \
       --filter="displayName=gcp-inference-vertex" --format='value(name)' 2>/dev/null | head -1)
  if [ -n "$ep" ]; then say "vertex: endpoint exists ($ep)"; return 0; fi
  # Vertex AI pulls only from Artifact Registry; a remote repository proxies Docker Hub. Kept between runs.
  gcloud artifacts repositories describe "$repo" --project "$PROJECT" --location "$VERTEX_REGION" >/dev/null 2>&1 ||
    gcloud artifacts repositories create "$repo" --project "$PROJECT" --location "$VERTEX_REGION" \
      --repository-format docker --mode remote-repository --remote-docker-repo DOCKER-HUB \
      --description "Docker Hub proxy for Vertex AI custom containers" || { fail "vertex: repository create failed"; return 1; }
  # The proxy caches on first pull, and models upload checks that the image exists: fetch the manifest once.
  local path=${VLLM_IMAGE%:*} tag=${VLLM_IMAGE##*:}
  curl -sf -o /dev/null -H "Authorization: Bearer $(gcloud auth print-access-token)" \
    -H "Accept: application/vnd.oci.image.index.v1+json,application/vnd.docker.distribution.manifest.list.v2+json,application/vnd.docker.distribution.manifest.v2+json" \
    "https://$VERTEX_REGION-docker.pkg.dev/v2/$PROJECT/$repo/$path/manifests/$tag" || { fail "vertex: cannot prime $VLLM_IMAGE through $repo"; return 1; }
  gcloud ai models upload --project "$PROJECT" --region "$VERTEX_REGION" --display-name gcp-inference-vertex \
    --container-image-uri "$VERTEX_REGION-docker.pkg.dev/$PROJECT/$repo/$VLLM_IMAGE" \
    --container-args="$(echo "$VLLM_ARGS --port 8000" | tr ' ' ',')" --container-ports 8000 \
    --container-health-route /health --container-predict-route /v1/chat/completions \
    --container-shared-memory-size-mb 8192 --container-env-vars "$COMPAT" --labels "$LABEL" >/dev/null 2>&1 || { fail "vertex: model upload failed"; return 1; }
  # upload prints no id: take the newest model with this name.
  model=$(gcloud ai models list --project "$PROJECT" --region "$VERTEX_REGION" --filter="displayName=gcp-inference-vertex" \
          --sort-by=~createTime --limit 1 --format='value(name)' 2>/dev/null)
  say "vertex: model uploaded (${model##*/})"
  ep=$(gcloud ai endpoints create --project "$PROJECT" --region "$VERTEX_REGION" --display-name gcp-inference-vertex \
       --labels "$LABEL" --format='value(name)')
  case "$ep" in projects/*) ;; *) fail "vertex: endpoint create failed"; return 1 ;; esac
  say "vertex: endpoint created ($ep); deploying on g2-standard-4 + L4"
  gcloud ai endpoints deploy-model "${ep##*/}" --project "$PROJECT" --region "$VERTEX_REGION" \
    --model "${model##*/}" --display-name gcp-inference-vertex --machine-type g2-standard-4 \
    --accelerator type=nvidia-l4,count=1 --min-replica-count 1 --max-replica-count 1 2>&1 | tail -2
  [ "${PIPESTATUS[0]}" = 0 ] || { fail "vertex: deploy-model failed"; return 1; }
  say "vertex: deployed"
}

gke() {
  local name=gcp-inference-gke
  if ! gcloud container clusters describe "$name" --project "$PROJECT" --region "$GKE_REGION" >/dev/null 2>&1; then
    say "gke: creating Autopilot cluster $name in $GKE_REGION"
    gcloud container clusters create-auto "$name" --project "$PROJECT" --region "$GKE_REGION" --network "$GKE_NETWORK" --subnetwork "$GKE_NETWORK" \
      --labels "$LABEL" 2>&1 | tail -2
    [ "${PIPESTATUS[0]}" = 0 ] || { fail "gke: cluster create failed"; return 1; }
    say "gke: cluster created"
  fi
  gcloud container clusters get-credentials "$name" --project "$PROJECT" --region "$GKE_REGION" 2>&1 | tail -1
  sed -e "s|IMAGE|$VLLM_IMAGE|" -e "s|REPACK|$REPACK|" "$here/k8s/vllm.yaml" | kubectl apply -f - ||
    { fail "gke: kubectl apply failed"; return 1; }
  say "gke: deployment applied"
}

tpu() {
  local name=gcp-inference-tpu zone out ud rc
  if gcloud compute tpus tpu-vm list --project "$TPU_PROJECT" --zone - --filter="name~$name" --format='value(name)' 2>/dev/null | grep -q .; then
    say "tpu: already up"; return 0
  fi
  if gcloud alpha compute tpus queued-resources list --project "$TPU_PROJECT" --zone - --filter="name~$name" --format='value(name)' 2>/dev/null | grep -q .; then
    say "tpu: flex-start request already queued"; return 0
  fi
  ud=$(mktemp)
  cat > "$ud" <<UD
#!/usr/bin/env bash
set -x
docker inspect vllm >/dev/null 2>&1 && exit 0
# The script runs a second after boot, before Docker Hub is reachable: retry the pull.
until docker pull $TPU_IMAGE; do sleep 10; done
docker run -d --name vllm --restart unless-stopped --privileged --net=host \
  -v /dev/shm:/dev/shm --shm-size 10gb -e HF_HOME=/dev/shm $TPU_IMAGE \
  vllm serve $TPU_MODEL --served-model-name gemma --max-model-len 131072 --port 8000 \
  --max-num-batched-tokens 4096 --enable-auto-tool-choice --tool-call-parser gemma4
UD
  for zone in $TPU_ZONES; do
    out=$(gcloud compute tpus tpu-vm create "$name" --project "$TPU_PROJECT" --zone "$zone" \
      --accelerator-type "$TPU_ACCEL" --version "$TPU_RUNTIME" \
      --metadata-from-file startup-script="$ud" --labels "$LABEL" 2>&1)
    rc=$?
    if [ $rc = 0 ]; then say "tpu: launched in $zone"; rm -f "$ud"; return 0; fi
    case "$out" in
      *"no more capacity"*|*RESOURCE_EXHAUSTED*|*[Qq]uota*|*"insufficient capacity"*|*"Insufficient capacity"*|*stockout*|*"not available"*)
        say "tpu: no capacity in $zone, trying the next zone ($(echo "$out" | grep -i error | tail -1 | grep -o "\"message\": \"[^\"]*" | head -1 | cut -c13-150))" ;;
      *) fail "tpu: $zone: $(echo "$out" | grep -i error | tail -1 | cut -c1-200)"; rm -f "$ud"; return 1 ;;
    esac
  done
  # No on-demand capacity anywhere: queue a flex-start (DWS) request, which draws on a separate
  # pool, stops by itself after 4 hours, and names its node <queued resource>-node.
  zone=$TPU_FLEX_ZONE
  say "tpu: queueing a $TPU_ACCEL flex-start request in $zone${TPU_ZONES:+ (no on-demand capacity in: $TPU_ZONES)}"
  out=$(gcloud alpha compute tpus queued-resources create "$name" --project "$TPU_PROJECT" --zone "$zone" \
    --node-id "$name-node" --accelerator-type "$TPU_ACCEL" --runtime-version "$TPU_RUNTIME" \
    --provisioning-model flex-start --max-run-duration 4h --valid-until-duration 4h \
    --metadata-from-file startup-script="$ud" --labels "$LABEL" 2>&1)
  rc=$?; rm -f "$ud"
  [ $rc = 0 ] || { fail "tpu: flex-start request failed: $(echo "$out" | tail -1)"; return 1; }
  say "tpu: $TPU_ACCEL flex-start request queued in $zone"
}

want=${*:-gce vertex gke run tpu}
pids=()
for b in $want; do
  case $b in
    gce) gce & ;;
    vertex) vertex & ;;
    gke) gke & ;;
    run) cloudrun & ;;
    tpu) tpu & ;;
    *) echo "unknown backend: $b (use gce vertex gke run tpu)"; continue ;;
  esac
  pids+=($!)
done
rc=0; for p in "${pids[@]}"; do wait "$p" || rc=1; done
echo; echo "Next: stage/forward.sh && source stage/demo.env && python3 stage/check.py --wait 1800"
exit $rc
