"""Step through the talk: one cue per slide, with the live command for that slide.

Run it in the terminal you will show, next to the Google Slides deck:

    source stage/demo.env
    python3 stage/present.py              # from slide 1
    python3 stage/present.py --start 30   # pick up at the agent section
    python3 stage/present.py --list       # print every cue and command, run nothing

Keys (type the letter, then Enter):
    Enter   run this slide's next command; with none left, go to the next slide
    n / p   next / previous slide          g N   go to slide N
    r       show the recorded run for the current command instead of running it
    s       skip the current command       l     show the slide's link again
    q       quit

Each slide header shows its link (#slide=id.<object id>) so the deck can be jumped to
from the terminal, the minutes planned so far, and the clock since slide 1. A
command whose backend is not configured, or that exits non-zero, offers its
recorded run from results/2026-10-12/.
"""

import argparse
import os
import subprocess
import sys
import time

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

DECK = "https://docs.google.com/presentation/d/11qC6IwhqEEtOFqmQH3l4f0XGUbokoQi9NkeMN5F4K_Q/edit"
# Google Slides object id of each slide, in deck order (slide N is SLIDE_IDS[N - 1]), is listed after
# CUES. Ids were fixed when each slide was made, so they no longer follow the slide numbers.


def link(n: int) -> str:
    return f"{DECK}#slide=id.{SLIDE_IDS[n - 1]}"


RECORDED = "results/2026-10-12"
BOLD, DIM, GREEN, YELLOW, RED, RESET = "\033[1m", "\033[2m", "\033[32m", "\033[33m", "\033[31m", "\033[0m"


def ask(backend: str) -> dict:
    return {"cmd": f"python3 stage/ask.py {backend}", "needs": backend,
            "recorded": f"{RECORDED}/bench-{backend}.txt"}


# (slide, title, minutes, notes, commands)
CUES = [
    (1, "Inference on GCP", 1.0,
     ["Seven ways to serve a model on Google Cloud, one ADK agent over all of them."], []),
    (2, "About me", 1.0, ["GDE Cloud and AI/ML, AWS Community Builder."], []),
    (3, "Where to find me", 0.5, ["Everything today is on GitHub and dev.to."], []),
    (4, "One Question, Seven Ways to Answer It", 2.0,
     ["Hosted API, managed endpoint, serverless GPU, your GPU, Google's chip.",
      "Grading and timing happen in code, never in the model."], []),
    (5, "The Seven Backends", 2.0,
     ["Four L4s with the checkpoint and vLLM image of the AWS run.",
      "The TPU serves the bf16 checkpoint with vLLM TPU, tool calls included."], []),
    (6, "Precision: What You Trade", 2.0,
     ["Memory and quality belong to the model; the right panel is the chip.",
      "TPU v6e runs int8 at twice its bf16 rate and fp8 at the bf16 rate. No TPU page lists int4."], []),
    (7, "Memory and Bandwidth Across Google Cloud", 1.5,
     ["v5e reads bf16 at 800 GiB/s; the L4s read 4-bit at 300 GB/s. Both decode at 140-144 tok/s on the server."], []),
    (8, "Strategies", 0.5, ["Four habits that decided whether a run happened at all."], []),
    (9, "Quota Does Not Mean Availability", 2.5,
     ["13 zones in 5 US regions said STOCKOUT for one L4, with 16 L4s of quota in each.",
      "TPU v6e: refused everywhere. v5e flex-start: a node in 22 seconds."],
     [{"cmd": "grep -E 'no capacity|launched|flex-start' stage/up.log", "recorded": f"{RECORDED}/capacity.txt"}]),
    (10, "Pin the Image, Read the Driver", 2.0,
     ["The -cu129 build of vLLM 0.30.0 dies at import everywhere; the default build serves.",
      "Driver 580 on Cloud Run and GKE runs CUDA 13 natively: leave forward-compat off."], []),
    (11, "Check Your Numbers, Verify Results", 2.0,
     ["The grader read '8 ids are 10 or more' as 10 until it dropped restated thresholds.",
      "One GPU type, 90 to 119 tok/s from the client; 140 to 143 on the server. The difference is the network."], []),
    (12, "No Public Ports, Still Reachable", 1.5,
     ["IAP tunnel, kubectl port-forward, SSH forward, Cloud Run with an identity token.",
      "This project's default network is legacy; GKE needed the VPC."], []),
    (13, "Hosted: Gemini and Gemma MaaS", 0.5, [], []),
    (14, "Gemini: Nothing to Start", 2.0,
     ["A model id and a role. 20 of 20 on both tool sets, 2.95 s per agent run."],
     [ask("gemini")]),
    (15, "Gemma 4 MaaS: Open, Serverless", 2.5,
     ["Experimental, global only, USD 0.60 per million output tokens.",
      "Counting the rows itself, it said 9 in 19 of 20 runs. With the engine, 8."],
     [ask("maas")]),
    (16, "Managed: Vertex AI and Cloud Run", 0.5, [], []),
    (17, "Vertex AI: You Bring the Container", 1.5,
     ["Upload a model, create an endpoint, deploy onto g2-standard-4 with an L4."], []),
    (18, "Vertex AI: Your vLLM, Managed", 2.0,
     ["Docker Hub through an Artifact Registry proxy; 24.4 minutes to deploy.",
      "Shared endpoint: rawPredict, 60-second timeout, no streaming."],
     [ask("vertex")]),
    (19, "Cloud Run: An L4 That Scales to Zero", 3.0,
     ["6.6 minutes from deploy to serving. Billed per second while an instance exists.",
      "Server-side decode equals the VM's: the GPU is the same."],
     [ask("cloudrun")]),
    (20, "From Zero, the First Request Waits Minutes", 3.0,
     ["The request that wakes the service is held, then answered; others get a 429.",
      "torch.compile is two of those minutes; a warm compile cache cuts it to 6 seconds."],
     [{"cmd": "cat results/2026-10-12/coldstart-all.txt", "recorded": f"{RECORDED}/coldstart-all.txt"}]),
    (21, "GPUs: Compute Engine and GKE", 0.5, [], []),
    (22, "Compute Engine: vLLM on an L4", 2.5,
     ["9.6 minutes launch to ready, in Toronto after Montreal filled up.",
      "143.1 tok/s on the server; 94.3 through the IAP tunnel."],
     [ask("gce")]),
    (23, "NVIDIA L4: The GPU in Four of Seven Backends", 1.5,
     ["Ada Lovelace, compute capability 8.9, 24 GB at 300 GB/s."], []),
    (24, "GKE: A Pod Asks, a Node Appears", 2.5,
     ["The Pod asked for 4 vCPU and 16 GiB; Autopilot made a g2-standard-8.",
      "Cheapest L4 per token at 16 in flight: USD 0.203 per million."],
     [ask("gke")]),
    (25, "Google Silicon: Cloud TPU", 0.5, [], []),
    (26, "TPU v5e: vLLM, Tool Calls and All", 3.0,
     ["vllm/vllm-tpu, same release as the GPUs; 6.5 minutes of XLA compilation at start.",
      "20 of 20 with tool calls. 144.1 tok/s on the server, bf16."],
     [ask("tpu")]),
    (27, "The Cloud TPU API Is Frozen", 1.5,
     ["TPU7x and later: Compute Engine or GKE only. v5e stays on the TPU API."], []),
    (28, "vLLM on TPU Ships With the GPU Releases", 1.0,
     ["v0.31.0 on Docker Hub for both, two days apart."], []),
    (29, "TPU's Hard Part: Getting One", 2.0,
     ["Serving was easy. Capacity was the work: v6e nowhere, v5e flex-start in 22 seconds."], []),
    (30, "One ADK Agent", 0.5, [], []),
    (31, "One Agent, Two Model Classes", 2.0,
     ["Gemini for Gemini; LiteLlm with an api_base for the other six.",
      "Vertex needs rawPredict; Cloud Run an identity token."], []),
    (32, "Demo 1: Counting Itself, Gemma MaaS Said 9", 4.0,
     ["Eleven ids; how many are 10 or more? The answer is 8.",
      "Then the contrast: --tools rows hands the model the ids to count itself."],
     [{"cmd": "python3 demo1_counting.py --runs 3", "recorded": f"{RECORDED}/demo1-engine-gemini.txt"},
      {"cmd": "python3 demo1_counting.py --runs 3 --tools rows", "recorded": f"{RECORDED}/demo1-rows-maas.txt"}]),
    (33, "Demo 2: TPU and GKE Answered in About 0.2 s", 3.0,
     ["A Gemini agent asks every backend; scoreboard() sorts in code.",
      "The [harness] line lists any backend the agent skipped."],
     [{"cmd": "python3 demo2_orchestrator.py", "recorded": f"{RECORDED}/demo2-run1.txt"}]),
    (34, "Same Silicon Speed, Different Paths", 3.0,
     ["Client tok/s includes the network; server tok/s is vLLM's own clock."],
     [{"cmd": "python3 bench_chat.py --repeats 3", "recorded": f"{RECORDED}/bench-cloudrun.txt"}]),
    (35, "At 16 in Flight, Self-Hosting Beats Per-Token", 2.0,
     ["TPU v5e flex-start USD 0.138 per million at 16 in flight; Gemma MaaS 0.60, Gemini 2.50.",
      "One request at a time, every L4 costs about what Gemini costs per token."], []),
    (36, "So, Which One?", 2.5,
     ["Gemini first; Cloud Run for your weights at low traffic; GKE or a VM when it is busy; TPU when you can get one."], []),
    (37, "Clean Up Two Projects", 1.5,
     ["stage/down.sh deletes everything and sweeps for leftovers. Run it after the talk."], []),
    (38, "Links", 1.0, ["Repo, rigs, write-up."], []),
    (39, "Questions", 5.0, [], []),
]
SLIDE_IDS = ["sl_01", "sl_02", "sl_03", "sl_04", "sl_05", "sl_prec", "sl_mem", "sl_06", "sl_07", "sl_08",
             "sl_09", "sl_10", "sl_11", "sl_12", "sl_maas", "sl_mgd", "sl_smi", "sl_13", "sl_run", "sl_cold",
             "sl_14", "sl_15", "sl_l4", "sl_16", "sl_17", "sl_18", "sl_19", "sl_20", "sl_why", "sl_21",
             "sl_22", "sl_23", "sl_24", "sl_25", "sl_26", "sl_27", "sl_28", "sl_29", "sl_30"]
assert len(SLIDE_IDS) == len(CUES)


def configured() -> set:
    try:
        import backends
        return {b.name for b in backends.configured()}
    except Exception:
        return set()


def show_file(path: str) -> None:
    full = os.path.join(REPO, path)
    print(f"{YELLOW}[recorded run: {path}]{RESET}")
    try:
        with open(full) as f:
            sys.stdout.write(f.read())
    except OSError as exc:
        print(f"{RED}cannot read it: {exc}{RESET}")


def run(c: dict, have: set) -> bool:
    need = c.get("needs")
    if need and need not in have:
        print(f"{YELLOW}{need} is not configured (source stage/demo.env?). r shows the recorded run.{RESET}")
        return False
    print(f"{BOLD}$ {c['cmd']}{RESET}", flush=True)
    t0 = time.perf_counter()
    rc = subprocess.call(c["cmd"], shell=True, cwd=REPO)
    took = time.perf_counter() - t0
    if rc != 0:
        print(f"{RED}exit {rc} after {took:.1f} s. r shows the recorded run.{RESET}")
        return False
    print(f"{DIM}({took:.1f} s){RESET}")
    return True


def header(i: int, start: float) -> None:
    n, title, minutes, notes, cmds = CUES[i]
    planned = sum(c[2] for c in CUES[:i])
    elapsed = (time.time() - start) / 60
    drift = elapsed - planned
    pace = (f"{GREEN}on time{RESET}" if abs(drift) < 1 else
            f"{YELLOW}{drift:+.1f} min{RESET}" if abs(drift) < 3 else f"{RED}{drift:+.1f} min{RESET}")
    print("\n" + "─" * 72)
    print(f"{BOLD}Slide {n}/{len(CUES)}  {title}{RESET}   {DIM}{minutes:g} min{RESET}")
    print(f"{DIM}planned {planned:.1f} min · clock {elapsed:.1f} min · {RESET}{pace}")
    print(f"{DIM}{link(n)}{RESET}")
    for line in notes:
        print(f"  • {line}")
    for c in cmds:
        print(f"  {BOLD}▶ {c['cmd']}{RESET}")


def list_cues() -> None:
    total = sum(c[2] for c in CUES)
    for n, title, minutes, notes, cmds in CUES:
        print(f"{n:>2}  {minutes:>4g} min  {title}")
        for c in cmds:
            print(f"              $ {c['cmd']}")
    print(f"\n{len(CUES)} slides, {total:g} minutes planned, "
          f"{sum(len(c[4]) for c in CUES)} live commands")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--start", type=int, default=1, help="slide to start on")
    ap.add_argument("--list", action="store_true", help="print the cues and commands, run nothing")
    a = ap.parse_args()
    if a.list:
        list_cues()
        return 0

    have = configured()
    print(f"Backends configured: {', '.join(sorted(have)) or 'none (source stage/demo.env)'}")
    i = max(0, min(len(CUES) - 1, a.start - 1))
    # Starting mid-deck counts as being on schedule at that slide.
    start = time.time() - 60 * sum(c[2] for c in CUES[:i])
    step = 0
    header(i, start)
    while True:
        cmds = CUES[i][4]
        cur = cmds[step] if step < len(cmds) else None
        prompt = f"[Enter: run {cur['cmd'].removeprefix('python3 ')}]" if cur else "[Enter: next slide]"
        try:
            key = input(f"{DIM}{prompt} n p g N r s l q > {RESET}").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            return 0
        if key == "q":
            return 0
        if key == "" and cur:
            run(cur, have)
            step += 1
            continue
        if key == "r" and cur:
            show_file(cur["recorded"])
            step += 1
            continue
        if key == "r" and step > 0 and cmds:
            show_file(cmds[step - 1]["recorded"])
            continue
        if key == "s" and cur:
            step += 1
            continue
        if key == "l":
            print(link(CUES[i][0]))
            continue
        if key in ("", "n"):
            if i == len(CUES) - 1:
                print("Last slide. q to quit.")
                continue
            i, step = i + 1, 0
        elif key == "p":
            i, step = max(0, i - 1), 0
        elif key.startswith("g"):
            try:
                i, step = max(0, min(len(CUES) - 1, int(key[1:].strip()) - 1)), 0
            except ValueError:
                print("g needs a slide number, e.g. g 23")
                continue
        else:
            print("Keys: Enter n p g N r s l q")
            continue
        header(i, start)


if __name__ == "__main__":
    sys.exit(main())
