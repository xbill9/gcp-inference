"""MCP server for the GCP inference demo: the seven backends, and the engine that counts.

One server over every backend. It runs gcloud and kubectl, so it needs nothing beyond what
stage/ uses.

    read-only    list_backends      every demo resource: Compute Engine, TPU, Cloud Run, GKE, Vertex AI
                 backend_health     ask one backend for the capital of France, on the box where there is one
                 backend_logs       the serving log: SSH, kubectl or Cloud Logging, by backend
                 launch_log         tail the last launch or teardown log
                 ask_backend        one chat request through backends.chat_once, timed here
                 count_ids          the demo 1 engine tool: exact count, min and max for a predicate
    write        launch_backends    stage/up.sh in the background; starts billing
    destructive  teardown           stage/down.sh in the background; deletes everything

How each health check and log reaches its backend, with no forward open:

    gce       gcloud compute ssh through IAP; curl-equivalent and docker logs on the VM
    tpu       gcloud compute tpus tpu-vm ssh; the same on the TPU VM
    gke       kubectl exec / kubectl logs on deploy/vllm
    cloudrun  HTTPS with an identity token; logs from Cloud Logging
    vertex    :rawPredict with an access token; logs from Cloud Logging
    gemini, maas  the Vertex AI API; no logs

Launch and teardown call the stage scripts rather than reimplementing them, so the MCP
path and the documented path launch exactly the same thing.

ask_backend reaches the backends through the variables in stage/demo.env, which is read at
start-up for any variable the environment does not already set; gce, gke and tpu need the
forwards from stage/forward.sh.

    python3 mcp_server.py                  # stdio; .mcp.json registers it for Claude Code
    python3 demo1_counting.py --tools mcp  # demo 1 with count_ids served from here
"""

import asyncio
import json
import logging
import os
import re
import shlex
import subprocess
import sys

from mcp.server.mcpserver import MCPServer
from mcp.types import ToolAnnotations

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import backends
import demo1_counting

NAME = "tpu-demo"
logging.basicConfig(level=logging.INFO, stream=sys.stderr)
logger = logging.getLogger(NAME)

mcp = MCPServer(os.getenv("MCP_SERVER_NAME", NAME))
READ_ONLY = ToolAnnotations(readOnlyHint=True, idempotentHint=True)
WRITE = ToolAnnotations(destructiveHint=False)
DESTRUCTIVE = ToolAnnotations(destructiveHint=True)

# The same defaults as stage/up.sh, forward.sh and down.sh.
PROJECT = os.getenv("PROJECT", "comglitn")
TPU_PROJECT = os.getenv("TPU_PROJECT", "aisprint-491218")
RUN_REGION = os.getenv("RUN_REGION", "us-east4")
VERTEX_REGION = os.getenv("VERTEX_REGION", "us-central1")
GKE_REGION = os.getenv("GKE_REGION", "us-central1")
LABEL = "managed-by=gcp-inference"  # the label stage/up.sh puts on every resource
GCE_NAME = os.getenv("GCE_NAME", "gcp-inference-gce")
TPU_NAME = "gcp-inference-tpu"      # a flex-start node is gcp-inference-tpu-node
RUN_NAME = "gcp-inference-run"
GKE_NAME = "gcp-inference-gke"
VERTEX_NAME = "gcp-inference-vertex"
KUBE_CONTEXT = f"gke_{PROJECT}_{GKE_REGION}_{GKE_NAME}"

BOX_BACKENDS = ["gce", "tpu", "gke"]            # a machine or Pod to run a command on
SERVED_BACKENDS = ["cloudrun", "vertex"]        # launched here, reached over HTTPS
HOSTED_BACKENDS = ["gemini", "maas"]            # per token, nothing to launch
LAUNCHED = BOX_BACKENDS + SERVED_BACKENDS
ALL_BACKENDS = HOSTED_BACKENDS + LAUNCHED
HEALTH_QUESTION = "What is the capital of France? Answer in one word."

# Runs on the box (VM host or the vLLM container): POST one chat request to vLLM on port 8000
# and print "<http status> <seconds>" then the body. python3 is on the Deep Learning VM, the TPU
# VM and in the vLLM image; curl is not guaranteed in the image.
PROBE = r"""
import json, sys, time, urllib.error, urllib.request
body = json.dumps({"model": "gemma", "max_tokens": 16, "temperature": 0.0,
                   "messages": [{"role": "user", "content": sys.argv[1]}]}).encode()
req = urllib.request.Request("http://localhost:8000/v1/chat/completions", body, {"content-type": "application/json"})
t0 = time.perf_counter()
try:
    with urllib.request.urlopen(req, timeout=90) as r:
        code, out = r.status, r.read().decode()
except urllib.error.HTTPError as e:
    code, out = e.code, e.read().decode()
except OSError as e:
    code, out = 0, str(e)
print(code, round(time.perf_counter() - t0, 3))
print(out)
"""


def _load_demo_env() -> None:
    """Take stage/demo.env's values for anything the environment leaves unset."""
    path = os.path.join(HERE, "stage", "demo.env")
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            for key, value in re.findall(r"(\w+)=(\S+)", line.removeprefix("export ")):
                default = re.fullmatch(r"\$\{\w+:-(.*)\}", value)
                os.environ.setdefault(key, default.group(1) if default else value)


_load_demo_env()


class CommandError(RuntimeError):
    """A gcloud or kubectl command that exited non-zero."""


def _run(*args: str, timeout: int = 120) -> str:
    # stdin is the MCP pipe; a child must not inherit it (az blocks reading it; kubectl exec forwards it)
    proc = subprocess.run(args, capture_output=True, text=True, timeout=timeout, check=False,
                          stdin=subprocess.DEVNULL)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip().splitlines()
        raise CommandError(f"{args[0]} {args[1]} exited {proc.returncode}: {' '.join(detail[-3:])[:600]}")
    return proc.stdout


async def _cmd(*args: str, timeout: int = 120) -> str:
    return await asyncio.to_thread(_run, *args, timeout=timeout)


async def _json(*args: str) -> list | dict:
    out = await _cmd(*args, "--format=json")
    return json.loads(out) if out.strip() else []


def _error(exc: Exception) -> str:
    if isinstance(exc, subprocess.TimeoutExpired):
        return f"❌ timed out after {exc.timeout}s: {' '.join(exc.cmd[:4])}"
    return f"❌ {type(exc).__name__}: {exc}"


def _check_backend(backend: str, allowed: list[str]) -> None:
    if backend not in allowed:
        raise ValueError(f"backend must be one of {', '.join(allowed)}")


# --- discovery: one function per backend, each returning rows for list_backends ---

async def _gce() -> list[dict]:
    rows = await _json("gcloud", "compute", "instances", "list", "--project", PROJECT,
                       f"--filter=labels.{LABEL}")
    return [{"backend": "gce", "name": i["name"], "where": i["zone"].rsplit("/", 1)[-1],
             "type": i["machineType"].rsplit("/", 1)[-1], "state": i["status"],
             "created": i["creationTimestamp"], "billing": i["status"] in {"PROVISIONING", "STAGING", "RUNNING"}}
            for i in rows]


async def _tpu() -> list[dict]:
    nodes, queued = await asyncio.gather(
        _json("gcloud", "compute", "tpus", "tpu-vm", "list", "--project", TPU_PROJECT, "--zone", "-",
              f"--filter=name~{TPU_NAME}"),
        _json("gcloud", "alpha", "compute", "tpus", "queued-resources", "list", "--project", TPU_PROJECT,
              "--zone", "-", f"--filter=name~{TPU_NAME}"))
    rows = [{"backend": "tpu", "name": n["name"].rsplit("/", 1)[-1], "where": n["name"].split("/")[3],
             "type": n.get("acceleratorType", "?"), "state": n.get("state", "?"),
             "created": n.get("createTime", "?"), "billing": n.get("state") not in {"STOPPED", "DELETING"}}
            for n in nodes]
    rows += [{"backend": "tpu", "name": q["name"].rsplit("/", 1)[-1],
              "where": q["name"].split("/")[3], "type": "queued resource",
              "state": q.get("state", {}).get("state", "?"), "created": q.get("createTime", "?"),
              "billing": False}  # the node it provisions is listed, and billed, above
             for q in queued]
    return rows


async def _cloudrun() -> list[dict]:
    rows = await _json("gcloud", "run", "services", "list", "--project", PROJECT, "--region", RUN_REGION,
                       f"--filter=metadata.labels.{LABEL}")
    out = []
    for s in rows:
        ready = {c["type"]: c["status"] for c in s.get("status", {}).get("conditions", [])}.get("Ready", "?")
        out.append({"backend": "cloudrun", "name": s["metadata"]["name"], "where": RUN_REGION,
                    "type": "L4, scales to zero", "state": "Ready" if ready == "True" else f"Ready={ready}",
                    "created": s["metadata"].get("creationTimestamp", "?"), "url": s.get("status", {}).get("url"),
                    "billing": "only while an instance is up"})
    return out


async def _gke() -> list[dict]:
    rows = await _json("gcloud", "container", "clusters", "list", "--project", PROJECT,
                       f"--filter=name={GKE_NAME}")
    out = []
    for c in rows:
        row = {"backend": "gke", "name": c["name"], "where": c["location"], "type": "Autopilot",
               "state": c["status"], "created": c.get("createTime", "?"), "billing": True}
        if c["status"] == "RUNNING":
            try:
                await _kube_credentials()
                d = await _json_kube("get", "deployment", "vllm", "-o", "json")
                row["deployment"] = (f"vllm {d.get('status', {}).get('readyReplicas', 0)}/"
                                     f"{d['spec']['replicas']} ready")
            except CommandError as exc:
                row["deployment"] = f"not found ({str(exc)[-120:]})"
        out.append(row)
    return out


async def _vertex() -> list[dict]:
    rows = await _json("gcloud", "ai", "endpoints", "list", "--project", PROJECT, "--region", VERTEX_REGION,
                       f"--filter=displayName={VERTEX_NAME}")
    out = []
    for e in rows:
        endpoint_id = e["name"].rsplit("/", 1)[-1]
        deployed = e.get("deployedModels", [])
        machine = (deployed[0].get("dedicatedResources", {}).get("machineSpec", {}).get("machineType", "?")
                   if deployed else "-")
        out.append({"backend": "vertex", "name": endpoint_id, "where": VERTEX_REGION, "type": machine,
                    "state": f"{len(deployed)} model(s) deployed", "created": e.get("createTime", "?"),
                    "url": (f"https://{VERTEX_REGION}-aiplatform.googleapis.com/v1/projects/{PROJECT}"
                            f"/locations/{VERTEX_REGION}/endpoints/{endpoint_id}"),
                    "billing": bool(deployed)})
    return out


DISCOVER = {"gce": _gce, "tpu": _tpu, "cloudrun": _cloudrun, "gke": _gke, "vertex": _vertex}


async def _kube_credentials() -> None:
    await _cmd("gcloud", "container", "clusters", "get-credentials", GKE_NAME, "--project", PROJECT,
               "--region", GKE_REGION)


async def _kube(*args: str, timeout: int = 120) -> str:
    return await _cmd("kubectl", "--context", KUBE_CONTEXT, *args, timeout=timeout)


async def _json_kube(*args: str) -> dict:
    return json.loads(await _kube(*args))


async def _find(backend: str) -> dict:
    """The one live resource behind a backend, or an error naming where it looked."""
    rows = await DISCOVER[backend]()
    live = {"gce": "RUNNING", "tpu": "READY", "gke": "RUNNING"}
    if backend in live:
        rows = [r for r in rows if r["state"] == live[backend] and r["type"] != "queued resource"]
    elif backend == "cloudrun":
        rows = [r for r in rows if r["name"] == RUN_NAME]
    elif backend == "vertex":
        rows = [r for r in rows if r["billing"]]
    if not rows:
        where = {"gce": PROJECT, "tpu": TPU_PROJECT}.get(backend, PROJECT)
        raise RuntimeError(f"no live {backend} resource labelled {LABEL} in project {where}")
    return rows[0]


async def _on_box(backend: str, row: dict, command: str, timeout: int = 180) -> str:
    """Run a shell command on the VM (gce, tpu) or in the vLLM Pod (gke)."""
    if backend == "gce":
        return await _cmd("gcloud", "compute", "ssh", row["name"], "--project", PROJECT, "--zone", row["where"],
                          "--tunnel-through-iap", "--quiet", "--command", command, timeout=timeout)
    if backend == "tpu":
        return await _cmd("gcloud", "compute", "tpus", "tpu-vm", "ssh", row["name"], "--project", TPU_PROJECT,
                          "--zone", row["where"], "--quiet", "--command", command, timeout=timeout)
    await _kube_credentials()
    return await _kube("exec", "deploy/vllm", "--", "sh", "-c", command, timeout=timeout)


def _hosted(backend: str, row: dict | None = None) -> backends.Backend:
    """A Backend for chat_once: from the environment, else built from what discovery found."""
    found = {b.name: b for b in backends.configured()}
    if backend in found:
        return found[backend]
    if backend == "maas":
        return backends.Backend("maas", "openai", "Gemma 4 26B, Vertex AI MaaS", "google/gemma-4-26b-a4b-it-maas",
                                url=f"https://aiplatform.googleapis.com/v1/projects/{PROJECT}/locations/global"
                                    "/endpoints/openapi", auth="access")
    if backend == "cloudrun":
        return backends.Backend("cloudrun", "openai", "vLLM on Cloud Run (L4)", "gemma",
                                url=row["url"] + "/v1", auth="identity")
    return backends.Backend("vertex", "openai", "Vertex AI endpoint (vLLM, L4)", "gemma", url=row["url"],
                            auth="access")


@mcp.tool(title="List demo backends", annotations=READ_ONLY)
async def list_backends() -> str:
    """Every backend the demo uses and its state right now.

    Lists Compute Engine VMs, TPU VMs and queued resources, Cloud Run services, the GKE cluster
    and its vllm Deployment, and Vertex AI endpoints, all found by the label managed-by=gcp-inference
    or the gcp-inference names stage/up.sh gives them. Reports how many are billing now and which
    backends ask_backend is configured for (from stage/demo.env, so stale once the forwards close).
    Counts are computed here.
    """
    try:
        found = await asyncio.gather(*[f() for f in DISCOVER.values()], return_exceptions=True)
        rows = [{"backend": "gemini", "where": "Vertex AI, global", "state": "always on", "billing": "per token"},
                {"backend": "maas", "where": "Vertex AI, global", "state": "always on", "billing": "per token"}]
        errors = {}
        for name, result in zip(DISCOVER, found):
            if isinstance(result, Exception):
                errors[name] = _error(result)
            else:
                rows += result
        seen = {r["backend"] for r in rows}
        rows += [{"backend": b, "state": "not launched", "billing": False}
                 for b in LAUNCHED if b not in seen and b not in errors]
        return json.dumps({"backends": rows,
                           "billing_now": sum(r["billing"] is True for r in rows),
                           "cloudrun_services": sum(r["backend"] == "cloudrun" and "name" in r for r in rows),
                           "lookup_errors": errors,
                           # from the environment and stage/demo.env; stale once the forwards close
                           "ask_backend_configured": [b.name for b in backends.configured()]}, indent=2)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Check one backend's health", annotations=READ_ONLY)
async def backend_health(backend: str) -> str:
    """Ask one backend for the capital of France and pass it only if the answer says Paris.

    For gce and tpu the request runs on the VM over SSH (IAP for gce); for gke it runs in the
    vLLM Pod with kubectl exec; so no forward is needed. Cloud Run and the Vertex AI endpoint are
    found by name and called over HTTPS; Gemini and MaaS through the Vertex AI API. A non-empty
    answer is not a pass: a broken deploy on this lineage has answered with degenerate repetition.

    Args:
        backend: one of gemini, maas, gce, tpu, gke, cloudrun, vertex
    """
    try:
        _check_backend(backend, ALL_BACKENDS)
        if backend in BOX_BACKENDS:
            row = await _find(backend)
            out = await _on_box(backend, row, f"python3 -c {shlex.quote(PROBE)} {shlex.quote(HEALTH_QUESTION)}")
            status_line, _, payload = out.strip().partition("\n")
            code, seconds = status_line.split()[0], float(status_line.split()[1])
            if code != "200":
                return f"❌ {backend}: HTTP {code} from port 8000 on {row['name']}: {payload[:300]}"
            text = json.loads(payload)["choices"][0]["message"]["content"] or ""
            how = {"gce": "over IAP SSH", "tpu": "over SSH", "gke": "via kubectl exec"}[backend]
            where = f"on {row['name']} ({row['where']}) {how}"
        else:
            row = await _find(backend) if backend in SERVED_BACKENDS else None
            r = await asyncio.to_thread(backends.chat_once, _hosted(backend, row), HEALTH_QUESTION, 16)
            text, seconds = r["text"], r["seconds"]
            where = "direct" if row is None else f"{row['name']} ({row['where']}), direct over HTTPS"
        ok = "paris" in text.lower()
        return f"{'✅' if ok else '❌'} {backend}: answered {text.strip()[:80]!r} in {seconds:.2f}s, {where}"
    except Exception as exc:
        return _error(exc)


def _logging_filter(backend: str, row: dict) -> str:
    """The serving log for a Cloud Run service or Vertex AI endpoint, without the audit log."""
    if backend == "cloudrun":
        resource = f'resource.type="cloud_run_revision" AND resource.labels.service_name="{row["name"]}"'
    else:
        resource = f'resource.type="aiplatform.googleapis.com/Endpoint" AND resource.labels.endpoint_id="{row["name"]}"'
    return resource + ' AND NOT logName:"cloudaudit.googleapis.com"'


def _log_line(e: dict) -> str:
    """vLLM's stdout is textPayload on Cloud Run and jsonPayload.message on Vertex AI;
    Cloud Run's request log is an httpRequest."""
    if e.get("textPayload"):
        text = e["textPayload"]
    elif e.get("jsonPayload"):
        text = e["jsonPayload"].get("message") or json.dumps(e["jsonPayload"])
    elif e.get("httpRequest"):
        h = e["httpRequest"]
        text = f"{h.get('requestMethod', '?')} {h.get('requestUrl', '?')} {h.get('status', '?')} {h.get('latency', '')}"
    else:
        text = "(no payload)"
    return f"{e.get('timestamp', '?')[:19]}  {text.rstrip()}"


@mcp.tool(title="Tail a backend's serving log", annotations=READ_ONLY)
async def backend_logs(backend: str, tail: int = 60) -> str:
    """Tail one backend's vLLM log.

    gce and tpu: docker logs on the VM over SSH, falling back to the startup-script journal
    while the container has not started. gke: kubectl logs on deploy/vllm, falling back to the
    Pod's events. cloudrun and vertex: the last day of entries from Cloud Logging, oldest first.

    Args:
        backend: one of gce, tpu, gke, cloudrun, vertex
        tail: lines to return, 1 to 2000
    """
    try:
        _check_backend(backend, LAUNCHED)
        n = max(1, min(int(tail), 2000))
        row = await _find(backend)
        head = f"{backend} {row['name']} ({row['where']})"
        if backend in ("gce", "tpu"):
            cmd = (f"sudo docker logs --tail {n} vllm 2>&1 || (echo '--- vllm container not running yet; "
                   f"startup script:'; sudo journalctl -u google-startup-scripts.service --no-pager -n {n})")
            return f"{head}\n```\n{(await _on_box(backend, row, cmd)).rstrip()}\n```"
        if backend == "gke":
            await _kube_credentials()
            try:
                out = await _kube("logs", "deploy/vllm", f"--tail={n}")
            except CommandError as exc:
                events = await _kube("get", "events", "--field-selector", "involvedObject.kind=Pod",
                                     "--sort-by=.lastTimestamp")
                out = f"--- no log yet ({exc}); Pod events:\n" + "\n".join(events.splitlines()[-n:])
            return f"{head}\n```\n{out.rstrip()}\n```"
        entries = await _json("gcloud", "logging", "read", _logging_filter(backend, row), "--project", PROJECT,
                              "--freshness", "1d", "--limit", str(n))
        lines = [_log_line(e) for e in reversed(entries)]
        return f"{head}, {len(lines)} entries from Cloud Logging\n```\n" + "\n".join(lines) + "\n```"
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Ask a backend", annotations=READ_ONLY)
async def ask_backend(backend: str, prompt: str, max_tokens: int = 1024) -> str:
    """One chat request to one backend, timed here, as stage/ask.py does.

    Tokens per second uses the backend's own completion token count over this server's wall
    clock, so it includes network and prefill. gce, gke and tpu need the forwards
    (stage/forward.sh); the others are reached directly.

    Args:
        backend: one of gemini, maas, gce, tpu, gke, cloudrun, vertex
        prompt: the question
        max_tokens: completion limit, 1 to 4096 (the GPU backends hold 8192 tokens in all)
    """
    try:
        found = {b.name: b for b in backends.configured()}
        if backend not in found:
            return f"❌ {backend} is not configured here; configured: {', '.join(found)}. Run stage/forward.sh."
        r = await asyncio.to_thread(backends.chat_once, found[backend], prompt, max(1, min(int(max_tokens), 4096)))
        return json.dumps(r, indent=2)
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Count ids", annotations=READ_ONLY)
def count_ids(op: str, value: int) -> dict:
    """Count the ids matching `id <op> value`, computed by the engine.

    Args:
        op: one of >=, >, <=, <, ==
        value: the integer to compare each id against
    """
    return demo1_counting.count(op, value)


def _background(script: str, args: list[str], log: str) -> str:
    path = os.path.join(HERE, "stage", log)
    with open(path, "w") as out:
        proc = subprocess.Popen([os.path.join(HERE, "stage", script), *args], cwd=HERE, stdin=subprocess.DEVNULL,
                                stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    return f"started stage/{script} {' '.join(args)} (pid {proc.pid}); output in stage/{log}, read it with launch_log"


@mcp.tool(title="Launch demo backends", annotations=WRITE)
async def launch_backends(names: list[str] | None = None) -> str:
    """Run stage/up.sh in the background. Billing starts now and runs until teardown.

    A backend already up is left alone. Ready in 7 to 25 minutes depending on the backend;
    poll list_backends and backend_health. Gemini and MaaS need no launch.

    Args:
        names: any of gce, vertex, gke, run, tpu; empty launches all five
    """
    try:
        names = names or []
        bad = [n for n in names if n not in {"gce", "vertex", "gke", "run", "tpu"}]
        if bad:
            return f"❌ unknown backend(s) {bad}; use gce vertex gke run tpu"
        return _background("up.sh", names, "up.log")
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Tear down every demo backend", annotations=DESTRUCTIVE)
async def teardown() -> str:
    """Run stage/down.sh in the background: delete every VM, TPU, queued resource and Cloud Run
    service labelled managed-by=gcp-inference, the GKE cluster and the Vertex AI endpoint and model,
    close the forwards, then sweep for anything still billing. Read the sweep with launch_log("down")."""
    try:
        return _background("down.sh", [], "down.log")
    except Exception as exc:
        return _error(exc)


@mcp.tool(title="Read the launch or teardown log", annotations=READ_ONLY)
def launch_log(which: str = "up", tail: int = 60) -> str:
    """Tail stage/up.log or stage/down.log.

    Args:
        which: up or down
        tail: lines to return
    """
    if which not in ("up", "down"):
        return "❌ which must be up or down"
    path = os.path.join(HERE, "stage", f"{which}.log")
    if not os.path.exists(path):
        return f"stage/{which}.log does not exist yet"
    with open(path) as f:
        lines = f.read().splitlines()
    return "\n".join(lines[-max(1, min(int(tail), 2000)):])


if __name__ == "__main__":
    mcp.run()
