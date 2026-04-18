#!/usr/bin/env python3
"""
bench_ane_power.py — ANE kernel timing + power profiler.

Runs the ANE draft model forward pass while sampling ANE power via macmon,
producing a per-kernel breakdown of both latency (ms) and energy (mJ).

Install: brew install macmon

Usage:
    uv run python bench_ane_power.py
    uv run python bench_ane_power.py --ctx-len 128 --seq-q 16 --runs 20
    uv run python bench_ane_power.py --no-power   # timing only (no macmon needed)
"""

import argparse
import statistics
import time
from collections import defaultdict

import mlx.core as mx

import ane_meter

DRAFT_MODEL_NAME = "z-lab/Qwen3-8B-DFlash-b16"
DEFAULT_CTX_LEN = 64
DEFAULT_SEQ_Q = 16
N_WARMUP = 5
N_RUNS = 20

KERNEL_LABELS = {
    "mega_qkv":        "mega_qkv  (QKV + norms + RoPE)",
    "gqa_tile":        "gqa_tile  (GQA KV expand)",
    "attn_out":        "attn_out  (SDPA)",
    "o_proj_residual": "o_proj_residual",
    "ffn_residual":    "ffn_residual (MLP)",
    "final_norm":      "final_norm",
}


def _sep(c="─", w=72): print(c * w)
def _header(t): _sep("═"); print(f"  {t}"); _sep("═")
def _section(t): print(); _sep(); print(f"  {t}"); _sep()


def _run_timed(ane, n_layers):
    """Run a single full forward pass, timing every kernel dispatch.
    Returns dict[kernel_name -> ms]."""
    k = ane.kernels
    times: dict[str, float] = {}

    t0 = time.perf_counter()
    for i in range(n_layers):
        p = f"l{i}_"

        t = time.perf_counter()
        k["mega_qkv"].run_uncached(
            [ane.b_hidden, getattr(ane, f"w_{p}in_norm"),
             ane.b_context,
             getattr(ane, f"w_{p}k_proj"),
             getattr(ane, f"w_{p}k_norm_4d"),
             ane.b_cos_k, ane.b_sin_k,
             getattr(ane, f"w_{p}v_proj"),
             getattr(ane, f"w_{p}q_proj"),
             getattr(ane, f"w_{p}q_norm_4d"),
             ane.b_cos_q, ane.b_sin_q],
            [ane.b_k_rope_4d, ane.b_v_4d_t, ane.b_q_rope_4d],
        )
        times.setdefault("mega_qkv", 0.0)
        times["mega_qkv"] += (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        k["gqa_tile"].run_uncached(
            [ane.b_k_rope_4d, ane.b_v_4d_t],
            [ane.b_kv_tiled],
        )
        times.setdefault("gqa_tile", 0.0)
        times["gqa_tile"] += (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        k["attn_out"].run_uncached(
            [ane.b_q_rope_4d, ane.b_kv_tiled, ane.b_attn_mask],
            [ane.b_attn_flat],
        )
        times.setdefault("attn_out", 0.0)
        times["attn_out"] += (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        k["o_proj_residual"].run_uncached(
            [ane.b_attn_flat, getattr(ane, f"w_{p}o_proj"), ane.b_hidden],
            [ane.b_attn_res],
        )
        times.setdefault("o_proj_residual", 0.0)
        times["o_proj_residual"] += (time.perf_counter() - t) * 1e3

        t = time.perf_counter()
        k["ffn_residual"].run_uncached(
            [ane.b_attn_res,
             getattr(ane, f"w_{p}post_norm"),
             getattr(ane, f"w_{p}gate"),
             getattr(ane, f"w_{p}up"),
             getattr(ane, f"w_{p}down")],
            [ane.b_hidden],
        )
        times.setdefault("ffn_residual", 0.0)
        times["ffn_residual"] += (time.perf_counter() - t) * 1e3

    t = time.perf_counter()
    k["final_norm"].run_uncached(
        [ane.b_hidden, ane.w_final_norm],
        [ane.b_output],
    )
    times["final_norm"] = (time.perf_counter() - t) * 1e3
    times["__total__"] = (time.perf_counter() - t0) * 1e3

    return times


def main():
    parser = argparse.ArgumentParser(description="ANE kernel timing + power profiler")
    parser.add_argument("--model",     default=DRAFT_MODEL_NAME)
    parser.add_argument("--ctx-len",   type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--seq-q",     type=int, default=DEFAULT_SEQ_Q)
    parser.add_argument("--warmup",    type=int, default=N_WARMUP)
    parser.add_argument("--runs",      type=int, default=N_RUNS)
    parser.add_argument("--no-power",  action="store_true",
                        help="Disable macmon power sampling (timing only)")
    args = parser.parse_args()

    _header("ANE Kernel Timing + Power Profiler")
    print(f"  Model    : {args.model}")
    print(f"  ctx_len  : {args.ctx_len}   seq_q : {args.seq_q}")
    print(f"  Warmup   : {args.warmup}    Runs  : {args.runs}")
    print(f"  Power    : {'disabled (--no-power)' if args.no_power else 'macmon 500ms sampling'}")

    _section("Loading model")
    from mirror_sd.loader import load_dflash_model
    from mirror_sd.ane_model import ANEDraftModel

    t0 = time.perf_counter()
    draft_model, config = load_dflash_model(args.model)
    print(f"  GPU draft loaded in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    ane = ANEDraftModel(seq_q=args.seq_q, ctx_len=args.ctx_len, config=config)
    ane.load_weights(draft_model)
    print(f"  ANE compiled+loaded in {time.perf_counter()-t0:.1f}s")
    print(f"  Layers: {ane.n_layers}   hidden: {ane.hidden}   heads: {ane.n_heads}/{ane.n_kv_heads}")

    # Prepare static inputs
    mx.random.seed(42)
    H = config.hidden_size
    noise_emb  = mx.random.normal((1, args.seq_q, H),        dtype=mx.float32) * 0.02
    target_hid = mx.random.normal((1, args.ctx_len, 5 * H),  dtype=mx.float32) * 0.02
    mx.eval(noise_emb, target_hid)

    def _prepare():
        ane._rope_cache_key = None
        ane._attn_mask_ctx_len = None
        ane.prepare_forward(noise_emb, None, None, target_hid)

    # Warmup
    _section(f"Warmup ({args.warmup} runs)")
    for i in range(args.warmup):
        _prepare()
        _run_timed(ane, ane.n_layers)
        print(f"  warmup {i+1}/{args.warmup}", end="\r", flush=True)
    print()

    # Timed runs, optionally with power sampling
    _section(f"Timed runs ({args.runs} runs)")
    all_times: dict[str, list[float]] = defaultdict(list)

    if not args.no_power:
        ane_meter.start()

    t_wall_start = time.perf_counter()
    for i in range(args.runs):
        _prepare()
        run_t = _run_timed(ane, ane.n_layers)
        for k, v in run_t.items():
            all_times[k].append(v)
        total = run_t.get("__total__", 0)
        print(f"  run {i+1:3d}/{args.runs}: total={total:.1f}ms", end="\r", flush=True)
    print()

    t_wall = time.perf_counter() - t_wall_start
    power_stats = ane_meter.stop() if not args.no_power else {}

    # ── Report ──────────────────────────────────────────────────────────────
    _header("REPORT")

    # Timing breakdown
    _section("Per-kernel latency  (summed over all layers, median ms)")
    kernel_order = ["mega_qkv", "gqa_tile", "attn_out", "o_proj_residual",
                    "ffn_residual", "final_norm"]
    total_layer_med = 0.0

    print(f"  {'Kernel':<36}  {'median':>8}  {'min':>8}  {'max':>8}  {'%total':>7}")
    _sep("-", 72)

    total_vals = all_times.get("__total__", [1])
    total_med  = statistics.median(total_vals)

    for kn in kernel_order:
        vals = all_times.get(kn, [])
        if not vals:
            continue
        med = statistics.median(vals)
        mn  = min(vals)
        mx_ = max(vals)
        pct = med / max(total_med, 1e-9) * 100
        label = KERNEL_LABELS.get(kn, kn)
        print(f"  {label:<36}  {med:>8.2f}  {mn:>8.2f}  {mx_:>8.2f}  {pct:>6.1f}%")
        if kn != "final_norm":
            total_layer_med += med

    _sep("-", 72)
    print(f"  {'TOTAL forward pass':<36}  {total_med:>8.2f}  "
          f"{min(total_vals):>8.2f}  {max(total_vals):>8.2f}  {'100.0':>6}%")

    # Power / energy
    if not args.no_power and power_stats.get("samples", 0) > 0:
        _section("ANE Power (macmon)")
        print(f"  Samples collected : {power_stats['samples']}")
        print(f"  Peak power        : {power_stats['peak_mw']:.1f} mW")
        print(f"  Average power     : {power_stats['avg_mw']:.1f} mW")
        print(f"  Total energy      : {power_stats['energy_mj']:.1f} mJ  "
              f"(over {power_stats['duration_s']:.1f}s)")

        # Energy per forward pass
        n_passes = args.runs
        energy_per_pass_mj = power_stats["energy_mj"] / max(n_passes, 1)
        print(f"  Energy / forward  : {energy_per_pass_mj:.3f} mJ")

        # Efficiency: mW / (ops/s)  — proxy via mW * ms = uJ per pass
        uj_per_pass = power_stats["avg_mw"] * total_med  # mW * ms = uJ
        print(f"  Avg power * lat   : {uj_per_pass:.1f} uJ  (proxy energy/pass)")

        # Utilization hint: ANE peak draw ~8000mW on M-series
        ANE_PEAK_MW = 8000.0
        util_pct = power_stats["avg_mw"] / ANE_PEAK_MW * 100
        print(f"  Utilization est.  : {util_pct:.1f}%  (vs {ANE_PEAK_MW:.0f}mW peak)")

    elif not args.no_power:
        print("\n  [No power samples collected — is macmon installed?]")
        print("  Install: brew install macmon")

    # Config summary for logs
    _section("Config summary (paste into optimization-attempts.md)")
    print(f"  model={args.model}  ctx_len={args.ctx_len}  seq_q={args.seq_q}")
    print(f"  n_layers={ane.n_layers}  hidden={ane.hidden}")
    print(f"  forward_median={total_med:.2f}ms  forward_min={min(total_vals):.2f}ms")
    if not args.no_power and power_stats.get("avg_mw"):
        print(f"  ane_avg_mw={power_stats['avg_mw']:.1f}  ane_peak_mw={power_stats['peak_mw']:.1f}")
    print()
    _sep("═")


if __name__ == "__main__":
    main()
