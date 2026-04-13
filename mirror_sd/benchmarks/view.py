"""View and compare llama-benchy JSON results.

Usage:
    # Show a single result as markdown table
    python -m mirror_sd.benchmarks.view spec.json

    # Compare two results with speedup table and chart
    python -m mirror_sd.benchmarks.view spec.json baseline.json

    # Save comparison chart
    python -m mirror_sd.benchmarks.view spec.json baseline.json --chart comparison.png
"""

import argparse
import json
import os
import sys


def parse_bench(path):
    with open(path) as f:
        data = json.load(f)
    rows = {}
    raw_rows = []
    for b in data["benchmarks"]:
        d = b["context_size"]
        is_ctx = b["is_context_prefill_phase"]
        if d == 0 and not is_ctx:
            key = ("tg", 0)
        elif is_ctx:
            key = ("ctx_tg", d)
        else:
            key = ("pp_tg", d)
        row = {
            "depth": d,
            "is_ctx": is_ctx,
            "tg": b["tg_throughput"]["mean"],
            "tg_std": b["tg_throughput"]["std"],
            "peak": b["peak_throughput"]["mean"],
            "peak_std": b["peak_throughput"]["std"],
            "pp": b["pp_throughput"]["mean"] if b["pp_throughput"] else 0,
            "pp_std": b["pp_throughput"]["std"] if b["pp_throughput"] else 0,
            "ttft": b["e2e_ttft"]["mean"] if b["e2e_ttft"] else 0,
        }
        rows[key] = row
        raw_rows.append(row)
    meta = {}
    if "metadata" in data:
        meta = data["metadata"]
    meta["model"] = data.get("model", "")
    return rows, raw_rows, meta


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
    ctx_a = {r["depth"]: r for r in rows_a if r["is_ctx"] or (r["depth"] == 0 and not r["is_ctx"])}
    ctx_b = {r["depth"]: r for r in rows_b if r["is_ctx"] or (r["depth"] == 0 and not r["is_ctx"])}
    depths = sorted(set(ctx_a.keys()) & set(ctx_b.keys()))

    lines = []
    lines.append(f"| Depth | {label_a} TG | {label_b} TG | Speedup | {label_a} Peak | {label_b} Peak |")
    lines.append(f"|------:|----------:|----------:|--------:|------------:|------------:|")
    for d in depths:
        a, b = ctx_a[d], ctx_b[d]
        speedup = a["tg"] / b["tg"] if b["tg"] > 0 else 0
        lines.append(f'| {d:>5} | {a["tg"]:.1f} | {b["tg"]:.1f} | {speedup:.2f}x | {a["peak"]:.1f} | {b["peak"]:.1f} |')
    return "\n".join(lines)


def plot_chart(rows_a, rows_b, meta_a, meta_b, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    label_a = meta_a.get("label", meta_a.get("mode", "spec"))
    label_b = meta_b.get("label", "Baseline")

    ctx_a = {r["depth"]: r for r in rows_a if r["is_ctx"] or (r["depth"] == 0 and not r["is_ctx"])}
    ctx_b = {r["depth"]: r for r in rows_b if r["is_ctx"] or (r["depth"] == 0 and not r["is_ctx"])}
    depths = sorted(set(ctx_a.keys()) & set(ctx_b.keys()))

    a_tg = [ctx_a[d]["tg"] for d in depths]
    a_std = [ctx_a[d]["tg_std"] for d in depths]
    b_tg = [ctx_b[d]["tg"] for d in depths]
    b_std = [ctx_b[d]["tg_std"] for d in depths]

    x = np.arange(len(depths))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    ax1.bar(x - width/2, b_tg, width, label=label_b, color="#6c757d", yerr=b_std, capsize=3)
    bars2 = ax1.bar(x + width/2, a_tg, width, label=label_a, color="#0d6efd", yerr=a_std, capsize=3)
    ax1.set_xlabel("Context Depth")
    ax1.set_ylabel("TG Throughput (tok/s)")
    ax1.set_title("Decode Speed Comparison")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in depths])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars2, a_tg):
        ax1.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    speedups = [s/b if b > 0 else 0 for s, b in zip(a_tg, b_tg)]
    colors = ["#198754" if s >= 1.0 else "#dc3545" for s in speedups]
    bars3 = ax2.bar(x, speedups, width*1.5, color=colors, alpha=0.85)
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Context Depth")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title(f"{label_a} Speedup over {label_b}")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in depths])
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars3, speedups):
        ax2.annotate(f"{val:.2f}x", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    model_name = meta_a.get("model", "Qwen3.5-27B")
    fig.suptitle(f"{model_name} — Speculative Decoding Benchmark", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="View and compare llama-benchy results")
    parser.add_argument("results", nargs="+", help="JSON result files (1=show, 2=compare)")
    parser.add_argument("--chart", "-c", default=None, help="Save comparison chart as PNG")
    args = parser.parse_args()

    if len(args.results) == 1:
        _, raw_rows, meta = parse_bench(args.results[0])
        label = meta.get("label", meta.get("mode", "result"))
        print(fmt_table(raw_rows, label))
        print()
        print(fmt_summary(raw_rows))

    elif len(args.results) == 2:
        _, raw_a, meta_a = parse_bench(args.results[0])
        _, raw_b, meta_b = parse_bench(args.results[1])
        label_a = meta_a.get("label", meta_a.get("mode", "spec"))
        label_b = meta_b.get("label", meta_b.get("mode", "baseline"))

        print(fmt_compare(raw_a, raw_b, label_a, label_b))

        chart_path = args.chart
        if chart_path is None:
            out_dir = os.path.dirname(args.results[0]) or "."
            chart_path = os.path.join(out_dir, "comparison.png")
        plot_chart(raw_a, raw_b, meta_a, meta_b, chart_path)

    else:
        print("ERROR: provide 1 file (view) or 2 files (compare)")
        sys.exit(1)


if __name__ == "__main__":
    main()
