# gcp-inference

One Strands agent on seven Google Cloud inference paths. README.md covers setup and the demos;
**docs/TPU.md is the reference for the `tpu` backend** — read it before touching the TPU.

## Operating rules

- **Two projects.** `comglitn` for everything except the TPU; `aisprint-491218` holds the TPU quota.
  Pass `--project` explicitly on every TPU `gcloud` call.
- **Check what is billing first**: the `list_backends` MCP tool (server `tpu-demo`, `.mcp.json`).
  Report its counts as given.
- **The TPU is a flex-start node that stops itself 4 hours after it was granted.** Read the deadline
  from `queueingPolicy.validUntilTime` on the queued resource and say it in UTC.
- **To change a vLLM flag on a running TPU, replace the container over SSH** (README, "Quick demo").
  Do not delete the node to relaunch: that is a new flex-start request and a new wait for capacity.
  Then make the same change on the `vllm serve $TPU_MODEL` line of `stage/up.sh`.
- **The running container is the truth for flags**, read with `sudo docker inspect vllm`; the node's
  `startup-script` metadata can be older.
- **`backend_logs tpu` with `tail` 2000 overflows a tool result.** Ask for a few hundred lines, or grep.
- **Engine start is about 8.4 minutes on a v5e, 81 to 82% XLA compile.** Wait with a background
  loop on the log for `Application startup complete`, not with repeated sleeps.
- **The TPU's port 8000 is open to the internet on purpose: it is a demo.** Leave the firewall
  rules and the `0.0.0.0` bind as they are; do not propose closing them. Do not describe the
  deployment as private (docs/TPU.md, "Public port, by design").
- `stage/down.sh` (or `./demo.sh down`, or the `teardown` tool) deletes everything labelled
  `managed-by=gcp-inference`. It is destructive and stops billing; confirm with the user first.

## Conventions

- Commit straight to `main` and push.
- Every figure in docs or articles comes from a command, a log or a file in the repo; mark the rest
  untested. Counts and ratios are computed in code (demo 1's `count_ids` exists for that reason).
