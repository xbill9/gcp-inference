# The TPU backend

What the `tpu` backend is, how it behaves, and how to operate it. Every figure here was read from a
command, a vLLM log or a test run on 2026-10-25 unless the line says otherwise. The raw logs and test
output are archived in [xbill9/tpu-survey `evidence/`](https://github.com/xbill9/tpu-survey/tree/main/evidence).

## What runs

| | |
|---|---|
| Accelerator | one TPU v5e chip: `v5litepod-1`, type `V5LITE_POD`, topology 1x1 |
| Project / zone | `aisprint-491218` (holds the TPU quota) / `us-west4-a` |
| Provisioning | flex-start queued resource `gcp-inference-tpu`, node `gcp-inference-tpu-node` |
| Runtime | `v2-alpha-tpuv5-lite`, TPU API `V2_ALPHA1` |
| Host | Ubuntu 22.04.3 LTS, kernel 6.5.0-1013-gcp, 24 vCPU, 47 GiB RAM, 97G boot disk |
| Image | `vllm/vllm-tpu:v0.30.0` |
| Inside the image | vllm 0.30.1rc1.dev56+g153f2ba1b.tpu, tpu_inference 0.0.0 (source build), jax/jaxlib 0.11.0, libtpu 0.0.44, torch 2.10.0+cpu |
| Model | `google/gemma-4-E2B-it`, bf16, served as `gemma` |
| Flags | `--max-model-len 131072 --max-num-batched-tokens 4096 --enable-auto-tool-choice --tool-call-parser gemma4` |
| Label | `managed-by=gcp-inference` |

Weights download to `/dev/shm` (`HF_HOME=/dev/shm`), 9.54 GiB of checkpoint; no Hugging Face token is
needed. The VM also runs seven Google TPU agent containers beside `vllm`; leave them alone.

## Memory on the chip

From vLLM's start-up log:

| HBM, one v5e chip | GiB |
|---|---:|
| Total | 15.75 |
| Weights, bf16 | 8.97 |
| vLLM's ceiling (92% of total) | 14.49 |
| Left for the KV cache | 5.52 |
| In use after the KV cache is allocated | 14.51 |

KV cache: 15 layers x 1,255 blocks x 256 tokens, shape `(256, 1, 2, 256)` in bf16 = 1,024 bytes per
token per layer, 15,360 bytes per token, **321,280 tokens**. The config has 35 layers and
`num_kv_shared_layers: 20`, so only 15 hold their own KV. One 131,072-token request needs 1.875 GiB of
KV; the cache holds 2.45 of them at once (39.22 at `--max-model-len 8192`). The cache size does not
depend on `--max-model-len`: vLLM sizes it from the memory left after the weights.

## Measured behaviour

| | |
|---|---|
| Engine start (profile, KV cache, warm-up) | 502.84 s at 8,192; 503.54 s at 131,072 |
| of which XLA compile | 408.59 s; 413.41 s (81 to 82%) |
| Health check (`backend_health tpu`) | "Paris" in 0.21 s |
| Generation, `ask_backend` | 115.5 to 125.3 tok/s, one stream, timed from the client |
| Prefill, 124,949-token prompt | 9.41 to 9.54 s, about 13,100 tok/s from the client; vLLM logs 12,495 tok/s |
| Long-context recall | passphrase found at 10%, 50% and 90% depth of 124,949 tokens (`tpu-survey/scripts/longctx.py`) |
| Demo 1 over MCP | 3/3 correct, 3/3 right filter (`id >= 10`), median 0.9 s |
| Prompt over the limit | refused with HTTP 400 "maximum context length is 131072 tokens" before reaching the chip |

## Operating it

```bash
./demo.sh            # flex-start request, wait for the node, SSH tunnel, wait for vLLM, run demo 1
./demo.sh again      # the same on a node that is already up
./demo.sh down       # stage/down.sh: deletes everything labelled managed-by=gcp-inference
stage/forward.sh && source stage/demo.env   # TPU_URL=http://localhost:8005/v1 over SSH
```

MCP tools (`tpu-demo` server, `.mcp.json`): `list_backends`, `backend_health tpu`, `backend_logs tpu`
(docker logs over SSH; `tail` up to 2000, and 2000 lines overflows a tool result, so ask for less),
`ask_backend tpu` (default 1,024 tokens, cap 4,096), `launch_backends`, `teardown`.

Direct inspection:

```bash
gcloud compute tpus tpu-vm describe gcp-inference-tpu-node --project aisprint-491218 --zone us-west4-a
gcloud alpha compute tpus queued-resources describe gcp-inference-tpu --project aisprint-491218 --zone us-west4-a
gcloud compute tpus tpu-vm ssh gcp-inference-tpu-node --project aisprint-491218 --zone us-west4-a \
  --command 'sudo docker logs --tail 50 vllm'
```

**Change a vLLM flag without losing the node.** Replace the container over SSH (the command is in the
README under "Quick demo"). It keeps the node and its flex-start clock and costs one engine start,
about 8.4 minutes. Deleting the node to relaunch means a new flex-start request and a new wait for
capacity. Update `stage/up.sh` too, so the next launch matches.

## Quirks

- **Flex-start ends by itself.** `--max-run-duration 4h`: the node goes 4 hours after it was granted,
  whatever is running. `queueingPolicy.validUntilTime` on the queued resource shows the deadline.
- **Two names.** The queued resource is `gcp-inference-tpu`; the node it owns is
  `gcp-inference-tpu-node`. Deleting the queued resource with `--force` deletes the node
  (`stage/down.sh`). An on-demand launch would name the node `gcp-inference-tpu`.
- **The startup script runs once per container.** It exits when a `vllm` container exists, and the
  container restarts with its own flags (`--restart unless-stopped`). So the node's `startup-script`
  metadata can show older flags than the running server; `docker inspect vllm` is the truth.
- **Compile dominates start.** 81 to 82% of engine start is XLA compile; any flag change pays it again.
- **Allow about 20 tokens per line of English filler** when sizing long prompts for this tokenizer; the server's `usage.prompt_tokens` is the number to trust.
- **vLLM's throughput log averages over 10 s windows.** Its idle-period "Avg prompt throughput" of
  122 to 161 tok/s says nothing about prefill speed; time a real request.
- **`TPU KV cache usage` in the log** read 38 to 39% during one 124,949-token request (124,949 / 321,280 = 38.9%).

## Public port, by design

The TPU's vLLM is open to the internet, deliberately: this is a demo. The node has an external IP
(34.50.188.133 on 2026-10-25), vLLM listens on `0.0.0.0:8000`, and the `default` network in
`aisprint-491218` has firewall rules `allow-vllm-8000` and `allow-vllm-port-8000` allowing tcp:8000
from `0.0.0.0/0`. `http://<external-ip>:8000/v1/models` answered HTTP 200 from outside, with no
authentication. The scripts still use the SSH tunnel, so nothing depends on the open port. Anyone
with the IP can send prompts while the node is up; the 4-hour flex-start limit bounds it. To close it
for a non-demo run: `--host 127.0.0.1` on `vllm serve`, or narrow those firewall rules.

## Cost

v5e list prices (gemma4-dev, Cloud TPU pricing read 2026-10-12, not re-read here): $1.20 per chip-hour
on demand, $0.60 flex-start. The flex-start node bills from grant until deletion or the 4-hour limit.

## Further reading

- `~/gemma4-dev/TPU.md` and `HARDWARE.md`: zones, quota against capacity, stockouts, per-generation specs
- [xbill9/tpu-survey](https://github.com/xbill9/tpu-survey): the companion article and its evidence
