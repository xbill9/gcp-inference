"""The inference backends the two demos talk to, and how to reach each one.

Every backend is configured from the environment, so the same code runs against
whatever is up on the day. A backend whose variable is unset is skipped, not
faked: the demos print which ones they could reach.

    kind    the Google ADK model                  auth
    gemini  google.adk.models.Gemini              ADC, Vertex AI
    openai  google.adk.models.lite_llm.LiteLlm    none (a local tunnel), a Google access token
                                                  (Vertex AI), or an identity token (Cloud Run)

Every self-hosted backend runs vLLM with --enable-auto-tool-choice, so all of them
can drive an agent, the TPU included.

The Vertex AI endpoint serves vLLM behind :rawPredict, whose path does not end in
/chat/completions; _vertex_raw_predict rewrites the OpenAI client's request onto it.
"""

import json
import os
import subprocess
import time
import urllib.request
import warnings
from dataclasses import dataclass
from typing import Optional


# ADK announces each experimental feature it turns on; the demos' output has no room for it.
warnings.filterwarnings("ignore", message=r"\[EXPERIMENTAL\]")


@dataclass
class Backend:
    name: str
    kind: str
    label: str
    model: str
    url: Optional[str] = None
    auth: str = "none"         # none | access | identity
    project: Optional[str] = None
    location: Optional[str] = None

    tool_capable = True


def configured() -> list[Backend]:
    """Every backend whose settings are present in the environment."""
    env = os.environ.get
    project = env("GOOGLE_CLOUD_PROJECT", "comglitn")
    found = [
        Backend("gemini", "gemini", "Gemini on Vertex AI", env("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                project=project, location=env("GEMINI_LOCATION", "global")),
    ]
    if env("MAAS_MODEL"):
        # Serverless open model: the Vertex AI OpenAI-compatible endpoint, billed per token.
        found.append(Backend("maas", "openai", "Gemma 4 26B, Vertex AI MaaS", env("MAAS_MODEL"),
                             url=f"https://aiplatform.googleapis.com/v1/projects/{project}/locations/global/endpoints/openapi",
                             auth="access"))
    if env("VERTEX_URL"):
        found.append(Backend("vertex", "openai", "Vertex AI endpoint (vLLM, L4)",
                             env("VERTEX_MODEL", "gemma"), url=env("VERTEX_URL"), auth="access"))
    if env("GCE_URL"):
        found.append(Backend("gce", "openai", "vLLM on Compute Engine g2 (L4)",
                             env("GCE_MODEL", "gemma"), url=env("GCE_URL")))
    if env("GKE_URL"):
        found.append(Backend("gke", "openai", "vLLM on GKE Autopilot (L4)",
                             env("GKE_MODEL", "gemma"), url=env("GKE_URL")))
    if env("RUN_URL"):
        found.append(Backend("cloudrun", "openai", "vLLM on Cloud Run (L4)",
                             env("RUN_MODEL", "gemma"), url=env("RUN_URL"), auth="identity"))
    if env("TPU_URL"):
        found.append(Backend("tpu", "openai", "vLLM on Cloud TPU " + env("TPU_TYPE", "v5e-1"),
                             env("TPU_MODEL", "gemma"), url=env("TPU_URL")))
    only = env("BACKENDS")
    if only:
        keep = {b.strip() for b in only.split(",")}
        found = [b for b in found if b.name in keep]
    return found


_tokens: dict = {}


def token(kind: str) -> str:
    """A Google access or identity token from gcloud, reused for 45 minutes."""
    if kind == "none":
        return "unused"
    t, at = _tokens.get(kind, ("", 0.0))
    if time.time() - at > 45 * 60:
        cmd = ["gcloud", "auth", "print-access-token" if kind == "access" else "print-identity-token"]
        t = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout.strip()
        _tokens[kind] = (t, time.time())
    return t


def _vertex_raw_predict(request) -> None:
    """httpx hook: .../endpoints/ID/chat/completions -> .../endpoints/ID:rawPredict."""
    path = request.url.path
    if "/endpoints/" in path and path.endswith("/chat/completions") and "/endpoints/openapi/" not in path:
        request.url = request.url.copy_with(path=path[: -len("/chat/completions")] + ":rawPredict")


def _url(b: Backend) -> str:
    base = b.url.rstrip("/") + "/chat/completions"
    if b.name == "vertex":
        base = base[: -len("/chat/completions")] + ":rawPredict"
    return base


def adk_model(b: Backend):
    """The Google ADK model object for a backend."""
    if b.kind == "gemini":
        from google.adk.models import Gemini
        os.environ.setdefault("GOOGLE_GENAI_USE_VERTEXAI", "TRUE")
        os.environ.setdefault("GOOGLE_CLOUD_PROJECT", b.project)
        os.environ["GOOGLE_CLOUD_LOCATION"] = b.location
        return Gemini(model=b.model)
    if b.kind == "openai":
        from google.adk.models.lite_llm import LiteLlm
        extra = {}
        if b.name == "vertex":
            import httpx
            from openai import AsyncOpenAI
            hook = httpx.AsyncClient(event_hooks={"request": [_async(_vertex_raw_predict)]}, timeout=300)
            extra["client"] = AsyncOpenAI(base_url=b.url, api_key=token(b.auth), http_client=hook)
        return LiteLlm(model=f"openai/{b.model}", api_base=b.url, api_key=token(b.auth), **extra)
    raise ValueError(f"{b.name}: unknown kind {b.kind}")


def adk_agent(b: Backend, instruction: str, tools: list, temperature: float = 0.0, max_tokens: int = 512):
    """An ADK LlmAgent on one backend, greedy unless told otherwise."""
    from google.adk.agents import LlmAgent
    from google.genai import types
    return LlmAgent(name=f"agent_{b.name}", model=adk_model(b), instruction=instruction, tools=tools,
                    generate_content_config=types.GenerateContentConfig(temperature=temperature,
                                                                        max_output_tokens=max_tokens))


async def run_agent(agent, text: str) -> tuple[str, list]:
    """One user turn through an in-memory ADK runner. Returns the final text and the tool calls
    the agent made, read from its events as {"tool": name, **args}."""
    from google.adk.runners import InMemoryRunner
    from google.genai import types
    runner = InMemoryRunner(agent=agent, app_name="gcp-inference")
    session = await runner.session_service.create_session(app_name="gcp-inference", user_id="demo")
    answer, calls = "", []
    async for event in runner.run_async(user_id="demo", session_id=session.id,
                                        new_message=types.Content(role="user", parts=[types.Part(text=text)])):
        calls += [{"tool": c.name, **(c.args or {})} for c in event.get_function_calls()]
        if event.is_final_response() and event.content and event.content.parts:
            answer = "".join(p.text or "" for p in event.content.parts)
    return answer, calls


def _async(fn):
    async def hook(request):
        fn(request)
    return hook


_clients: dict = {}


def _genai(b: Backend):
    """One google-genai client per project and location, kept alive: a client that goes out of
    scope closes its HTTP session, and a request still using it fails."""
    key = (b.project, b.location)
    if key not in _clients:
        from google import genai
        _clients[key] = genai.Client(vertexai=True, project=b.project, location=b.location)
    return _clients[key]


def chat_once(b: Backend, prompt: str, max_tokens: int = 64) -> dict:
    """One plain chat request, timed here. Returns text, tokens and seconds.

    Tokens per second is computed from the backend's own completion token count
    and this function's wall clock, so it includes network and prefill.
    """
    t0 = time.perf_counter()
    if b.kind == "gemini":
        r = _genai(b).models.generate_content(
            model=b.model, contents=prompt,
            config={"max_output_tokens": max_tokens, "temperature": 0.0})
        text = r.text or ""
        completion = r.usage_metadata.candidates_token_count or 0
    else:
        body = {"model": b.model, "messages": [{"role": "user", "content": prompt}],
                "max_tokens": max_tokens, "temperature": 0.0}
        headers = {"content-type": "application/json"}
        if b.auth != "none":
            headers["authorization"] = f"Bearer {token(b.auth)}"
        req = urllib.request.Request(_url(b), json.dumps(body).encode(), headers)
        with urllib.request.urlopen(req, timeout=300) as resp:
            out = json.loads(resp.read())
        if isinstance(out, list):  # Vertex wraps some responses in a list
            out = out[0]
        text = out["choices"][0]["message"]["content"] or ""
        completion = out["usage"]["completion_tokens"]
    seconds = time.perf_counter() - t0
    return {"backend": b.name, "label": b.label, "text": text.strip(), "completion_tokens": completion,
            "seconds": round(seconds, 3), "tokens_per_second": round(completion / seconds, 1) if seconds else 0.0}
