"""Compare DFlash spec vs baseline benchmark results and generate a bar chart.

Usage:
    python -m mirror_sd.compare_bench benchmarks/spec.json benchmarks/baseline.json
    python -m mirror_sd.compare_bench benchmarks/spec.json benchmarks/baseline.json --output benchmarks/comparison.png
"""

import argparse
import json
import sys


def parse_bench(path):
    with open(path) as f:
        data = json.load(f)
    rows = {}
    for b in data["benchmarks"]:
        d = b["context_size"]
        is_ctx = b["is_context_prefill_phase"]
        if d == 0 and not is_ctx:
            key = ("tg", 0)
        elif is_ctx:
            key = ("ctx_tg", d)
        else:
            key = ("pp_tg", d)
        rows[key] = {
            "tg": b["tg_throughput"]["mean"],
            "tg_std": b["tg_throughput"]["std"],
            "peak": b["peak_throughput"]["mean"],
            "peak_std": b["peak_throughput"]["std"],
            "pp": b["pp_throughput"]["mean"] if b["pp_throughput"] else 0,
            "ttft": b["e2e_ttft"]["mean"] if b["e2e_ttft"] else 0,
        }
    meta = {}
    if "metadata" in data:
        meta = data["metadata"]
    meta["model"] = data.get("model", "")
    return rows, meta


def print_table(spec_rows, baseline_rows, spec_meta, baseline_meta):
    spec_label = spec_meta.get("label", spec_meta.get("mode", "DFlash+KOD"))
    bl_label = baseline_meta.get("label", "baseline")

    depths = sorted(set(
        d for (phase, d) in spec_rows.keys()
        if phase in ("tg", "ctx_tg")
    ))

    print(f"\n| {'Depth':>6} | {bl_label+' TG':>12} | {spec_label+' TG':>14} | {'Speedup':>8} | {bl_label+' Peak':>12} | {spec_label+' Peak':>14} |")
    print(f"|{'-'*8}|{'-'*14}|{'-'*16}|{'-'*10}|{'-'*14}|{'-'*16}|")
    for d in depths:
        s = spec_rows.get(("ctx_tg", d)) or spec_rows.get(("tg", d), {})
        b = baseline_rows.get(("ctx_tg", d)) or baseline_rows.get(("tg", d), {})
        s_tg = s.get("tg", 0)
        b_tg = b.get("tg", 0)
        s_pk = s.get("peak", 0)
        b_pk = b.get("peak", 0)
        speedup = s_tg / b_tg if b_tg > 0 else 0
        print(f"| {d:>6} | {b_tg:>12.1f} | {s_tg:>14.1f} | {speedup:>7.2f}x | {b_pk:>12.1f} | {s_pk:>14.1f} |")


def plot_chart(spec_rows, baseline_rows, spec_meta, baseline_meta, output_path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("ERROR: matplotlib not installed. Run: pip install matplotlib")
        sys.exit(1)

    spec_label = spec_meta.get("label", spec_meta.get("mode", "DFlash+KOD"))
    bl_label = baseline_meta.get("label", "Baseline")

    depths = sorted(set(
        d for (phase, d) in spec_rows.keys()
        if phase in ("tg", "ctx_tg")
    ))

    spec_tg = []
    spec_tg_std = []
    bl_tg = []
    bl_tg_std = []
    for d in depths:
        s = spec_rows.get(("ctx_tg", d)) or spec_rows.get(("tg", d), {})
        b = baseline_rows.get(("ctx_tg", d)) or baseline_rows.get(("tg", d), {})
        spec_tg.append(s.get("tg", 0))
        spec_tg_std.append(s.get("tg_std", 0))
        bl_tg.append(b.get("tg", 0))
        bl_tg_std.append(b.get("tg_std", 0))

    x = np.arange(len(depths))
    width = 0.35

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Left: TG throughput comparison
    bars1 = ax1.bar(x - width/2, bl_tg, width, label=bl_label, color="#6c757d", yerr=bl_tg_std, capsize=3)
    bars2 = ax1.bar(x + width/2, spec_tg, width, label=spec_label, color="#0d6efd", yerr=spec_tg_std, capsize=3)
    ax1.set_xlabel("Context Depth")
    ax1.set_ylabel("TG Throughput (tok/s)")
    ax1.set_title("Decode Speed: Baseline vs DFlash+KOD")
    ax1.set_xticks(x)
    ax1.set_xticklabels([str(d) for d in depths])
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars2, spec_tg):
        ax1.annotate(f"{val:.1f}", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=8)

    # Right: Speedup ratio
    speedups = [s/b if b > 0 else 0 for s, b in zip(spec_tg, bl_tg)]
    colors = ["#198754" if s >= 1.0 else "#dc3545" for s in speedups]
    bars3 = ax2.bar(x, speedups, width*1.5, color=colors, alpha=0.85)
    ax2.axhline(y=1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.5)
    ax2.set_xlabel("Context Depth")
    ax2.set_ylabel("Speedup (x)")
    ax2.set_title("DFlash+KOD Speedup over Baseline")
    ax2.set_xticks(x)
    ax2.set_xticklabels([str(d) for d in depths])
    ax2.grid(axis="y", alpha=0.3)

    for bar, val in zip(bars3, speedups):
        ax2.annotate(f"{val:.2f}x", xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                     xytext=(0, 3), textcoords="offset points", ha="center", fontsize=9, fontweight="bold")

    model_name = spec_meta.get("model", "Qwen3.5-27B")
    fig.suptitle(f"{model_name} — DFlash Speculative Decoding Benchmark", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Saved chart: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Compare spec vs baseline benchmarks")
    parser.add_argument("spec", help="Path to spec benchmark JSON")
    parser.add_argument("baseline", help="Path to baseline benchmark JSON")
    parser.add_argument("--output", "-o", default=None, help="Output image path (default: same dir as spec)")
    args = parser.parse_args()

    spec_rows, spec_meta = parse_bench(args.spec)
    baseline_rows, baseline_meta = parse_bench(args.baseline)

    print_table(spec_rows, baseline_rows, spec_meta, baseline_meta)

    output = args.output
    if output is None:
        import os
        out_dir = os.path.dirname(args.spec) or "."
        output = os.path.join(out_dir, "comparison.png")

    plot_chart(spec_rows, baseline_rows, spec_meta, baseline_meta, output)


if __name__ == "__main__":
    main()
