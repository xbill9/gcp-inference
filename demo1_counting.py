"""Demo 1: the same Google ADK agent, the same question, on every tool-capable backend.

The question is "how many of these ids are 10 or more?" over eleven ids. Two tool
sets are offered, never both at once:

    engine  count_ids(op, value) -> the exact count, min and max, computed here
    mcp     the same count_ids, served by mcp_server.py over MCP (stdio)
    rows    list_ids()           -> the raw ids; the model has to count them itself

A run is graded on what happened, not on what the model says:
    correct   the final answer contains the right count
    filter    the agent called count_ids with a predicate that selects the same ids
              (an "engine" run that quotes 8 after sending id > 10 got lucky, and is
              marked as a wrong filter even though the number is right)

    python3 demo1_counting.py                 # engine tools, 5 runs per backend
    python3 demo1_counting.py --tools rows    # the model counts (diagnostic)
    python3 demo1_counting.py --tools mcp     # count_ids from the MCP server
    python3 demo1_counting.py --runs 20 --json results/demo1.json
"""

import argparse
import asyncio
import json
import operator
import os
import re
import sys
import time

import backends

IDS = [0, 2, 3, 20, 21, 22, 23, 10, 11, 12, 13]
QUESTION = "How many of the ids are 10 or more? Answer with the number."
OPS = {">=": operator.ge, ">": operator.gt, "<=": operator.le, "<": operator.lt, "==": operator.eq}
TRUTH = sorted(i for i in IDS if i >= 10)
SYSTEM = ("You answer questions about a table of integer ids. Use the tools to look at the "
          "table. End with a sentence that states the number.")


def clean_op(op) -> str:
    """Gemma 4 on vLLM TPU sends the operator inside its own quotes ('">="'); take the operator."""
    return str(op).strip().strip("\"'").strip()


def count(op: str, value: int) -> dict:
    """The engine: exact count, min and max of the ids matching `id <op> value`."""
    op = clean_op(op)
    if op not in OPS:
        return {"error": f"op must be one of {sorted(OPS)}"}
    hit = [i for i in IDS if OPS[op](i, value)]
    return {"where": f"id {op} {value}", "count": len(hit),
            "min": min(hit) if hit else None, "max": max(hit) if hit else None}


def mcp_toolset():
    """An ADK toolset for mcp_server.py over stdio, offering count_ids only."""
    from google.adk.tools.mcp_tool import McpToolset
    from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
    from mcp import StdioServerParameters
    server = os.path.join(os.path.dirname(os.path.abspath(__file__)), "mcp_server.py")
    return McpToolset(connection_params=StdioConnectionParams(
        server_params=StdioServerParameters(command=sys.executable, args=[server]), timeout=30),
        tool_filter=["count_ids"])


def count_ids(op: str, value: int) -> dict:
    """Count the ids matching `id <op> value`, computed by the engine.

    Args:
        op: one of >=, >, <=, <, ==
        value: the integer to compare each id against
    """
    return count(op, value)


def list_ids() -> list:
    """Return every id in the table."""
    return IDS


# Restatements of the question's threshold ("8 ids are 10 or more"), dropped before
# the count is read, so the last number left is the model's answer.
THRESHOLD = re.compile(r"(?:\b10\s+or\s+(?:more|greater|higher|above|larger)\b|(?:>=|≥)\s*10\b|"
                       r"\b(?:at least|greater than or equal to|equal to or greater than)\s+10\b)", re.I)


def grade(answer: str, calls: list, kind: str) -> dict:
    numbers = [int(n) for n in re.findall(r"\b\d+\b", THRESHOLD.sub(" ", answer))]
    correct = bool(numbers) and numbers[-1] == len(TRUTH)
    counted = [{**c, "op": clean_op(c.get("op"))} for c in calls if c["tool"] == "count_ids"]
    counted = [c for c in counted if c["op"] in OPS]
    if kind == "rows":
        filter_ok = None
    else:
        filter_ok = any(sorted(i for i in IDS if OPS[c["op"]](i, c["value"])) == TRUTH for c in counted)
    return {"correct": correct, "filter_ok": filter_ok, "quoted": numbers[-1] if numbers else None,
            "filters": [f"id {c['op']} {c['value']}" for c in counted], "tool_calls": len(calls)}


async def run_once(b: backends.Backend, kind: str, toolset=None) -> dict:
    """One agent run. The tool calls are read back from the agent's events, so a tool served
    by the MCP server in its own process is graded the same way as a local one."""
    tools = [toolset] if kind == "mcp" else [count_ids] if kind == "engine" else [list_ids]
    agent = backends.adk_agent(b, SYSTEM, tools)
    t0 = time.perf_counter()
    calls: list = []
    try:
        answer, calls = await backends.run_agent(agent, QUESTION)
        error = None
    except Exception as exc:  # a backend failure is a result, not a crash
        answer, error = "", f"{type(exc).__name__}: {exc}"[:300]
    result = grade(answer, calls, kind)
    result.update(backend=b.name, seconds=round(time.perf_counter() - t0, 2), error=error,
                  answer=answer.strip()[-200:])
    return result


def summarize(rows: list, kind: str) -> list:
    out = []
    for name in dict.fromkeys(r["backend"] for r in rows):
        mine = [r for r in rows if r["backend"] == name]
        secs = sorted(r["seconds"] for r in mine)
        out.append({
            "backend": name,
            "runs": len(mine),
            "correct": sum(r["correct"] for r in mine),
            "filter_ok": None if kind == "rows" else sum(bool(r["filter_ok"]) for r in mine),
            "errors": sum(r["error"] is not None for r in mine),
            "median_seconds": secs[len(secs) // 2],
            "filters_sent": sorted({f for r in mine for f in r["filters"]}),
        })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tools", choices=["engine", "rows", "mcp"], default="engine")
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--json", help="write every run and the summary here")
    a = ap.parse_args()

    usable = [b for b in backends.configured() if b.tool_capable]
    print(f"Question: {QUESTION}\nIds: {IDS}\nTrue answer: {len(TRUTH)}  tools: {a.tools}\n")
    rows = asyncio.run(run_all(usable, a))
    summary = summarize(rows, a.tools)
    report(summary, rows, a)
    return 0


async def run_all(usable: list, a) -> list:
    toolset = mcp_toolset() if a.tools == "mcp" else None
    rows = []
    try:
        for b in usable:
            for n in range(a.runs):
                r = await run_once(b, a.tools, toolset)
                rows.append(r)
                mark = "ERR" if r["error"] else ("ok " if r["correct"] else "BAD")
                print(f"  {b.name:<10} run {n + 1}/{a.runs}  {mark}  quoted={r['quoted']}  "
                      f"filters={r['filters']}  {r['seconds']}s" + (f"  {r['error']}" if r["error"] else ""))
    finally:
        if toolset:
            await toolset.close()
    return rows


def report(summary: list, rows: list, a) -> None:
    print("\nbackend     runs  correct  right-filter  errors  median s  filters sent")
    for s in summary:
        rf = "-" if s["filter_ok"] is None else f"{s['filter_ok']}/{s['runs']}"
        print(f"{s['backend']:<10} {s['runs']:>5}  {s['correct']:>3}/{s['runs']:<3}  {rf:>12}  "
              f"{s['errors']:>6}  {s['median_seconds']:>8}  {', '.join(s['filters_sent']) or '-'}")
    skipped = [b.name for b in backends.configured() if not b.tool_capable]
    if skipped:
        print(f"\nNot run (no tool calls on these backends): {', '.join(skipped)}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump({"question": QUESTION, "ids": IDS, "truth": len(TRUTH), "tools": a.tools,
                       "summary": summary, "runs": rows}, f, indent=2)


if __name__ == "__main__":
    sys.exit(main())
