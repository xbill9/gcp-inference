#!/usr/bin/env bash
# Cold start from zero for three Cloud Run L4 variants, one request each, sent at the same moment
# after every service has been idle long enough to scale to zero.
#   default  stage/up.sh settings; startup probe period 60 s
#   eager    --enforce-eager (no torch.compile, no CUDA graphs); probe period 10 s
#   cache    Hugging Face and vLLM compile caches on a Cloud Storage volume (warm); probe period 10 s
set -uo pipefail
T=$(gcloud auth print-identity-token)
body='{"model":"gemma","messages":[{"role":"user","content":"What is the capital of France? Answer in one word."}],"max_tokens":8}'
for s in strands-demo-run strands-demo-run-eager strands-demo-run-cache; do
  u=$(gcloud run services describe $s --project comglitn --region us-east4 --format='value(status.url)')
  ( curl -s -m 1200 -o /dev/null -w "$s sent=$(date -u +%T) http=%{http_code} total_s=%{time_total}\n" \
      -H "Authorization: Bearer $T" -H content-type:application/json "$u/v1/chat/completions" -d "$body" ) &
done
wait
