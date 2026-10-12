"""Cost of each backend, from GCP list prices and measured throughput. All arithmetic is here.

    python3 cost.py results/<run> --out results/<run>

Prices come from the Cloud Billing Catalog API at run time (on-demand list prices) and are written to
OUT/prices.json, each with the SKU description it came from. Throughput is the median tok/s of every
backend in a bench_chat.py table: one request at a time.

USD per million output tokens = USD per hour / (tok/s * 3600) * 1e6, so it assumes the backend is
busy for the whole hour. Gemini and Gemma MaaS are billed per token and have no hourly price. Cloud
Run with min-instances 0 bills only while an instance exists; its hourly figure is for an instance
that is up, and its idle cost is zero.

Machine shapes are the ones stage/up.sh launches: g2-standard-4 (4 vCPU, 16 GiB, one L4) on Compute
Engine and Vertex AI, a 4 vCPU / 16 GiB / one-L4 container on Cloud Run and GKE Autopilot, and one
TPU chip (v5e by default, flex-start price) on Cloud TPU. GKE Autopilot GPU Pods are billed as the node Autopilot provisions plus an
accelerator premium; pass the node machine type with --gke-node (kubectl get nodes -L
node.kubernetes.io/instance-type).
"""

import argparse
import datetime
import glob
import json
import os
import re
import subprocess
import urllib.request

HOURS_PER_MONTH = 730
SERVICES = {"gce": "6F81-5844-456A", "vertex": "C7E2-9256-1C43", "run": "152E-C115-5142",
            "gke": "CCD8-9BF1-090E"}
# Region names as Compute Engine SKUs spell them.
LOCATION = {"us-east4": "Virginia", "us-central1": "Americas", "us-east1": "Americas", "us-west1": "Americas",
            "us-east5": "Columbus", "us-west4": "Las Vegas", "northamerica-northeast1": "Montreal",
            "northamerica-northeast2": "Toronto"}
TPU_SKU = {"v5e": "TpuV5e", "v6e": "TpuV6e"}  # Compute Engine SKU spelling
TPU_FLEX = {"v5e": "V5e", "v6e": "V6e"}       # "DWS Defined Duration <x>": flex-start
G2_SHAPES = {"g2-standard-4": (4, 16), "g2-standard-8": (8, 32), "g2-standard-12": (12, 48),
             "g2-standard-16": (16, 64)}


def skus(service):
    token = subprocess.check_output(["gcloud", "auth", "print-access-token"], text=True).strip()
    out, page = [], ""
    while True:
        req = urllib.request.Request(
            f"https://cloudbilling.googleapis.com/v1/services/{service}/skus?pageSize=5000&pageToken={page}",
            headers={"Authorization": f"Bearer {token}"})
        d = json.load(urllib.request.urlopen(req))
        out += d.get("skus", [])
        page = d.get("nextPageToken", "")
        if not page:
            return out


def rate(catalog, desc, region):
    """USD per unit of the one on-demand SKU with exactly this description in this region."""
    m = [s for s in catalog if s["description"] == desc and region in s.get("serviceRegions", [])
         and s["category"]["usageType"] == "OnDemand"]
    if len(m) != 1:
        raise SystemExit(f"expected one SKU '{desc}' in {region}, found {len(m)}")
    e = m[0]["pricingInfo"][0]["pricingExpression"]
    t = e["tieredRates"][-1]["unitPrice"]
    return {"sku": desc, "region": region, "unit": e["usageUnit"],
            "usd": int(t.get("units", "0")) + t.get("nanos", 0) / 1e9}


def fetch_prices(a):
    cat = {k: skus(v) for k, v in SERVICES.items()}
    r = {}
    loc = LOCATION[a.gce_region]
    r["gce_core"] = rate(cat["gce"], f"G2 Instance Core running in {loc}", a.gce_region)
    r["gce_ram"] = rate(cat["gce"], f"G2 Instance Ram running in {loc}", a.gce_region)
    r["gce_l4"] = rate(cat["gce"], f"Nvidia L4 GPU running in {loc}", a.gce_region)
    v = "Vertex AI: Online/Batch Prediction"
    r["vertex_core"] = rate(cat["vertex"], f"{v} G2 Predefined Instance Core running in Iowa", "us-central1")
    r["vertex_ram"] = rate(cat["vertex"], f"{v} G2 Predefined Instance Ram running in Iowa", "us-central1")
    r["vertex_l4"] = rate(cat["vertex"], f"{v} Nvidia L4 GPU running in Iowa", "us-central1")
    r["vertex_fee_core"] = rate(cat["vertex"], f"{v} Management fee on G2 Predefined Instance Core in Iowa", "us-central1")
    r["vertex_fee_ram"] = rate(cat["vertex"], f"{v} Management fee on G2 Predefined Instance RAM in Iowa", "us-central1")
    r["vertex_fee_l4"] = rate(cat["vertex"], f"{v} Management fee on NVIDIA L4 in Iowa", "us-central1")
    rr = a.run_region
    r["run_cpu"] = rate(cat["run"], f"Services CPU (Instance-based billing) in {rr}", rr)
    r["run_mem"] = rate(cat["run"], f"Services Memory (Instance-based billing) in {rr}", rr)
    r["run_l4"] = rate(cat["run"], f"NVIDIA L4 GPU with no zonal redundancy in {rr}", rr)
    gl = LOCATION[a.gke_region]
    r["gke_node_core"] = rate(cat["gce"], f"G2 Instance Core running in {gl}", a.gke_region)
    r["gke_node_ram"] = rate(cat["gce"], f"G2 Instance Ram running in {gl}", a.gke_region)
    r["gke_node_l4"] = rate(cat["gce"], f"Nvidia L4 GPU running in {gl}", a.gke_region)
    r["gke_prem_cpu"] = rate(cat["gke"], f"Autopilot Accelerator CPU Premium ({a.gke_region})", a.gke_region)
    r["gke_prem_mem"] = rate(cat["gke"], f"Autopilot Accelerator Memory Premium ({a.gke_region})", a.gke_region)
    r["gke_prem_l4"] = rate(cat["gke"], f"Autopilot L4 Premium ({a.gke_region})", a.gke_region)
    r["gke_cluster"] = rate(cat["gke"], "Autopilot Kubernetes Clusters", "global")
    tl = LOCATION[a.tpu_region]
    r["tpu_ondemand"] = rate(cat["gce"], f"{TPU_SKU[a.tpu]} running in {tl}", a.tpu_region)
    r["tpu_flex"] = rate(cat["gce"], f"DWS Defined Duration {TPU_FLEX[a.tpu]} running in {tl}", a.tpu_region)
    r["gemini_in"] = rate(cat["vertex"], "Gemini 3.5 Flash Lite Global Text Input - Predictions", "global")
    r["gemini_out"] = rate(cat["vertex"], "Gemini 3.5 Flash Lite Global Text Output - Predictions", "global")
    r["maas_in"] = rate(cat["vertex"], "Cloud Vertex AI Model Garden Model as a Service Gemma-4 Input Token", "global")
    r["maas_out"] = rate(cat["vertex"], "Cloud Vertex AI Model Garden Model as a Service Gemma-4 Output Token", "global")
    return {"fetched": datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds"), "rates": r}


def hourly(p, gke_node):
    u = {k: v["usd"] for k, v in p["rates"].items()}
    cpu, ram = 4, 16  # every GPU backend asks for 4 vCPU, 16 GiB and one L4
    node_cpu, node_ram = G2_SHAPES[gke_node]
    return {
        "gce": cpu * u["gce_core"] + ram * u["gce_ram"] + u["gce_l4"],
        "vertex": cpu * (u["vertex_core"] + u["vertex_fee_core"]) + ram * (u["vertex_ram"] + u["vertex_fee_ram"])
                  + u["vertex_l4"] + u["vertex_fee_l4"],
        "cloudrun": 3600 * (cpu * u["run_cpu"] + ram * u["run_mem"] + u["run_l4"]),
        "gke": node_cpu * u["gke_node_core"] + node_ram * u["gke_node_ram"] + u["gke_node_l4"]
               + cpu * u["gke_prem_cpu"] + ram * u["gke_prem_mem"] + u["gke_prem_l4"] + u["gke_cluster"],
        "tpu": u["tpu_flex"],  # the flex-start request is what got capacity; on-demand is listed beside it
    }


def read_bench(path):
    """bench_chat.py output -> {backend: median tok/s}: a results directory of bench-<b>.json, or one table."""
    if os.path.isdir(path):
        rates = {}
        for f in sorted(glob.glob(os.path.join(path, "bench-*.json"))):
            for srow in json.load(open(f))["summary"]:
                rates[srow["backend"]] = srow["median_tokens_per_second"]
        return rates
    rates = {}
    for line in open(path):
        m = re.match(r"(\w+)\s+[\d.]+\s+([\d.]+)\s+[\d.]+-[\d.]+\s+\d+-\d+", line)
        if m:
            rates[m.group(1)] = float(m.group(2))
    return rates


def per_million(usd_hour, tok_s):
    return usd_hour / (tok_s * 3600) * 1e6


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("bench")
    ap.add_argument("--out", required=True)
    ap.add_argument("--gce-region", default="northamerica-northeast1")
    ap.add_argument("--run-region", default="us-east4")
    ap.add_argument("--gke-region", default="us-central1")
    ap.add_argument("--gke-node", default="g2-standard-8", choices=sorted(G2_SHAPES))
    ap.add_argument("--tpu", default="v5e", choices=sorted(TPU_SKU))
    ap.add_argument("--tpu-region", default="us-west4")
    a = ap.parse_args()
    prices = fetch_prices(a)
    json.dump(prices, open(os.path.join(a.out, "prices.json"), "w"), indent=2)
    hours, rates = hourly(prices, a.gke_node), read_bench(a.bench)
    u = {k: v["usd"] * 1e6 for k, v in prices["rates"].items()}  # per million, for the token-billed two
    per_token = {"gemini": (u["gemini_in"], u["gemini_out"]), "maas": (u["maas_in"], u["maas_out"])}

    def cost(kv):
        name, tok_s = kv
        return per_token[name][1] if name in per_token else per_million(hours[name], tok_s)

    lines = [f"prices fetched {prices['fetched']}; bench {a.bench}; GKE node {a.gke_node}", "",
             "One request at a time (bench_chat.py, 100-token answer)", "",
             f"{'backend':10s} {'tok/s':>7s} {'USD/hr':>7s} {'USD/month 24x7':>15s} {'USD / 1M output tokens':>23s}"]
    for name, tok_s in sorted(rates.items(), key=cost):
        if name in per_token:
            i, o = per_token[name]
            lines.append(f"{name:10s} {tok_s:7.1f} {'-':>7s} {'0 idle':>15s} {o:23.2f}  (per-token list price, "
                         f"input {i:.3f})")
        else:
            h = hours[name]
            idle = {"cloudrun": "  (0 when scaled to zero)",
                    "tpu": f"  (flex-start; on-demand USD {prices['rates']['tpu_ondemand']['usd']:.2f}/hr = "
                           f"{per_million(prices['rates']['tpu_ondemand']['usd'], tok_s):.2f})"}.get(name, "")
            lines.append(f"{name:10s} {tok_s:7.1f} {h:7.4f} {h * HOURS_PER_MONTH:15.0f} "
                         f"{per_million(h, tok_s):23.2f}{idle}")

    par = os.path.join(a.bench, "parallel.json") if os.path.isdir(a.bench) else ""
    if par and os.path.exists(par):
        lines += ["", "Requests in parallel (parallel.py, timed from the client, 3 rounds per level)", "",
                  f"{'backend':10s} {'parallel':>8s} {'tok/s':>8s} {'USD / 1M output tokens':>23s}"]
        for r in json.load(open(par))["rows"]:
            lines.append(f"{r['backend']:10s} {r['parallel']:>8d} {r['aggregate_tok_s']:8.1f} "
                         f"{per_million(hours[r['backend']], r['aggregate_tok_s']):23.3f}")

    gce = hours["gce"] * HOURS_PER_MONTH
    lines += ["", f"gce 24x7 = USD {gce:.0f}/month. Per-token spend matches it at:"]
    for name, (i, o) in per_token.items():
        for ratio in (0, 5):  # input tokens per output token in the request mix
            blended = o + ratio * i
            m_tokens = gce / blended
            lines.append(f"  {name:6s} {ratio} input tokens per output token: USD {blended:.3f} per 1M output tokens, "
                         f"{m_tokens:.0f}M output tokens/month = {m_tokens * 1e6 / (HOURS_PER_MONTH * 3600):.0f} "
                         f"tok/s around the clock")
    text = "\n".join(lines)
    open(os.path.join(a.out, "cost.txt"), "w").write(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
