"""Convert llama-benchy JSON results to readable markdown tables.

Usage:
    python -m mirror_sd.fmt_bench benchmarks/spec.json
    python -m mirror_sd.fmt_bench benchmarks/spec.json benchmarks/baseline.json
"""

import json
import sys


def parse_bench(path):
    with open(path) as f:
        data = json.load(f)
    rows = []
    for b in data["benchmarks"]:
        rows.append({
            "depth": b["context_size"],
            "is_ctx": b["is_context_prefill_phase"],
            "pp": b["pp_throughput"]["mean"],
            "pp_std": b["pp_throughput"]["std"],
            "tg": b["tg_throughput"]["mean"],
            "tg_std": b["tg_throughput"]["std"],
            "peak": b["peak_throughput"]["mean"],
            "ttft": b["e2e_ttft"]["mean"],
        })
    meta = {
        "model": data.get("model", ""),
        "latency_mode": data.get("latency_mode", ""),
        "prefix_caching": data.get("prefix_caching_enabled", ""),
    }
    if "metadata" in data:
        meta.update(data["metadata"])
    return rows, meta


def fmt_table(rows, label=""):
    lines = []
    if label:
        lines.append(f"## {label}")
    lines.append(f'| {"Depth":>6} | {"Phase":>10} | {"PP t/s":>12} | {"TG t/s":>12} | {"Peak t/s":>10} | {"TTFT (ms)":>10} |')
    lines.append(f'|{"-"*8}|{"-"*12}|{"-"*14}|{"-"*14}|{"-"*12}|{"-"*12}|')
    for r in rows:
        d = r["depth"]
        if d == 0 and not r["is_ctx"]:
            phase = "tg"
        elif r["is_ctx"]:
            phase = "ctx_tg"
        else:
            phase = "pp+tg"
        pp_str = f'{r["pp"]:.0f}' if r["pp"] > 1000 else f'{r["pp"]:.1f}'
        tg_str = f'{r["tg"]:.1f} ± {r["tg_std"]:.1f}'
        lines.append(f'| {d:>6} | {phase:>10} | {pp_str:>12} | {tg_str:>12} | {r["peak"]:>10.1f} | {r["ttft"]:>10.0f} |')
    return "\n".join(lines)


def fmt_summary(rows):
    lines = []
    lines.append("| Depth | TG t/s | Peak t/s |")
    lines.append("|------:|-------:|---------:|")
    for r in rows:
        if r["is_ctx"] or (r["depth"] == 0 and not r["is_ctx"]):
            lines.append(f'| {r["depth"]:>5} | {r["tg"]:.1f} | {r["peak"]:.1f} |')
    return "\n".join(lines)


def fmt_compare(rows_a, rows_b, label_a, label_b):
    lines = []
    lines.append(f"| Depth | {label_a} TG | {label_b} TG | Speedup | {label_a} Peak | {label_b} Peak |")
    lines.append(f"|------:|----------:|----------:|--------:|------------:|------------:|")
    ctx_a = {r["depth"]: r for r in rows_a if r["is_ctx"] or (r["depth"] == 0 and not r["is_ctx"])}
    ctx_b = {r["depth"]: r for r in rows_b if r["is_ctx"] or (r["depth"] == 0 and not r["is_ctx"])}
    for d in sorted(set(ctx_a.keys()) & set(ctx_b.keys())):
        a, b = ctx_a[d], ctx_b[d]
        speedup = a["tg"] / b["tg"] if b["tg"] > 0 else 0
        lines.append(f'| {d:>5} | {a["tg"]:.1f} | {b["tg"]:.1f} | {speedup:.2f}x | {a["peak"]:.1f} | {b["peak"]:.1f} |')
    return "\n".join(lines)


def main():
    paths = sys.argv[1:]
    if not paths:
        print("Usage: python -m mirror_sd.fmt_bench <spec.json> [baseline.json]")
        sys.exit(1)

    rows_a, meta_a = parse_bench(paths[0])
    label_a = meta_a.get("label", meta_a.get("mode", "spec"))

    print(fmt_table(rows_a, label_a))
    print()
    print(fmt_summary(rows_a))

    if len(paths) > 1:
        rows_b, meta_b = parse_bench(paths[1])
        label_b = meta_b.get("label", "baseline")
        print()
        print(fmt_table(rows_b, label_b))
        print()
        print(fmt_compare(rows_a, rows_b, label_a, label_b))


if __name__ == "__main__":
    main()
