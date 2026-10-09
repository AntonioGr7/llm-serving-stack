"""Combine results/*.jsonl runs into one table: mean and min–max per (variant, workload).

  uv run bench/summarize.py results/run1.jsonl results/run2.jsonl [--json]
"""

import argparse
import json
import statistics
from collections import defaultdict

METRICS = ["req_per_s", "ttft_ms_p50", "ttft_ms_p90", "latency_ms_p50", "prefix_hit_rate"]
VARIANTS = ["llmd-load-only", "llmd-prefix-aware", "ray-pow2", "ray-prefix"]
WORKLOADS = ["unique", "shared-prefix", "multiturn"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("files", nargs="+")
    p.add_argument("--json", action="store_true", help="emit JSON rows instead of markdown")
    args = p.parse_args()

    cells = defaultdict(list)
    for path in args.files:
        for line in open(path):
            r = json.loads(line)
            if r["errors"] == 0:
                cells[(r["label"], r["workload"])].append(r)

    rows = []
    for w in WORKLOADS:
        for v in VARIANTS:
            runs = cells.get((v, w))
            if not runs:
                continue
            row = {"variant": v, "workload": w, "runs": len(runs)}
            for m in METRICS:
                xs = [r[m] for r in runs]
                row[m] = {"mean": statistics.fmean(xs), "min": min(xs), "max": max(xs)}
            rows.append(row)

    if args.json:
        print(json.dumps(rows, indent=1))
        return
    print("| Variant | Workload | Runs | req/s | TTFT p50 (ms) | TTFT p90 (ms) | Latency p50 (ms) | Prefix hit |")
    print("|---|---|--:|--:|--:|--:|--:|--:|")
    for r in rows:
        def f(m, fmt):
            s = r[m]
            spread = f" ({fmt(s['min'])}–{fmt(s['max'])})" if r["runs"] > 1 else ""
            return fmt(s["mean"]) + spread
        print(
            f"| {r['variant']} | {r['workload']} | {r['runs']} | {f('req_per_s', lambda x: f'{x:.2f}')} "
            f"| {f('ttft_ms_p50', lambda x: f'{x:.0f}')} | {f('ttft_ms_p90', lambda x: f'{x:.0f}')} "
            f"| {f('latency_ms_p50', lambda x: f'{x:.0f}')} | {f('prefix_hit_rate', lambda x: f'{x:.0%}')} |"
        )


if __name__ == "__main__":
    main()
