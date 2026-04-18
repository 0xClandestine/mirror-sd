#!/usr/bin/env python3
"""Per-dispatch ANE profiling — find where the 68ms goes.

Usage:
    .venv/bin/python bench_ane_dispatch.py
"""

import statistics
import time

import mlx.core as mx
from mirror_sd.loader import load_dflash_model
from mirror_sd.ane_model import ANEDraftModel

DRAFT_MODEL_NAME = "z-lab/Qwen3-8B-DFlash-b16"
N_WARMUP = 3
N_RUNS = 10


def profile_dispatches():
    draft_model, config = load_dflash_model(DRAFT_MODEL_NAME)
    ane = ANEDraftModel(seq_q=16, ctx_len=64, config=config)
    ane.load_weights(draft_model)

    mx.random.seed(42)
    H = config.hidden_size
    noise_emb = mx.random.normal((1, 16, H), dtype=mx.float32) * 0.02
    target_hid = mx.random.normal((1, 64, 5*H), dtype=mx.float32) * 0.02
    mx.eval(noise_emb, target_hid)

    # warmup
    for _ in range(N_WARMUP):
        ane._rope_cache_key = None
        ane._attn_mask_ctx_len = None
        out = ane(noise_emb, target_hid, rope_offset=0, ctx_len=64)
        mx.eval(out)

    # Now profile each kernel dispatch individually
    # We need to call prepare_forward then time each _run_layer piece
    k = ane.kernels
    all_times = {}  # kernel_name -> [times]

    for run in range(N_RUNS):
        ane._rope_cache_key = None
        ane._attn_mask_ctx_len = None
        ane.prepare_forward(noise_emb, None, None, target_hid)

        # Run through the full forward, timing each dispatch
        # Follow the exact sequence from run_kernels / _run_layer
        
        # 1. fc_norm
        t0 = time.perf_counter()
        # The actual fc_norm is run inside __call__ — we'll time _run_layer per kernel
        # Actually let me time the full _run_layer vs individual kernels

        for li in range(ane.n_layers):
            p = f"l{li}_"
            
            # mega_qkv (biggest: 3 matmuls + 2 norms + RoPE)
            t0 = time.perf_counter()
            k['mega_qkv'].run_uncached(
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
            dt = (time.perf_counter() - t0) * 1e3
            all_times.setdefault(f'L{li}_mega_qkv', []).append(dt)

            t0 = time.perf_counter()
            k['gqa_tile'].run_uncached(
                [ane.b_k_rope_4d, ane.b_v_4d_t],
                [ane.b_kv_tiled],
            )
            all_times.setdefault(f'L{li}_gqa_tile', []).append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            k['attn_out'].run_uncached(
                [ane.b_q_rope_4d, ane.b_kv_tiled, ane.b_attn_mask],
                [ane.b_attn_flat],
            )
            all_times.setdefault(f'L{li}_attn_out', []).append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            k['o_proj_residual'].run_uncached(
                [ane.b_attn_flat, getattr(ane, f"w_{p}o_proj"), ane.b_hidden],
                [ane.b_attn_res],
            )
            all_times.setdefault(f'L{li}_o_proj', []).append((time.perf_counter() - t0) * 1e3)

            t0 = time.perf_counter()
            k['ffn_residual'].run_uncached(
                [ane.b_attn_res, getattr(ane, f"w_{p}post_norm"),
                 getattr(ane, f"w_{p}gate"), getattr(ane, f"w_{p}up"), getattr(ane, f"w_{p}down")],
                [ane.b_hidden],
            )
            all_times.setdefault(f'L{li}_ffn', []).append((time.perf_counter() - t0) * 1e3)

    # Report
    print(f"\n{'='*65}")
    print(f"Per-dispatch ANE profile ({N_RUNS} runs, median times)")
    print(f"{'='*65}")
    
    type_totals = {}
    grand_total = 0
    for name, times in all_times.items():
        med = statistics.median(times)
        ktype = name.split('_', 1)[1] if '_' in name else name
        type_totals[ktype] = type_totals.get(ktype, 0) + med
        grand_total += med

    # Print per-layer
    for name in sorted(all_times.keys()):
        med = statistics.median(all_times[name])
        pct = med / grand_total * 100
        print(f"  {name:20s}: {med:6.1f}ms  ({pct:4.1f}%)")

    print(f"\n  By kernel type:")
    for ktype, total in sorted(type_totals.items(), key=lambda x: -x[1]):
        pct = total / grand_total * 100
        print(f"  {ktype:20s}: {total:6.1f}ms  ({pct:4.1f}%)")

    print(f"\n  Total dispatch time: {grand_total:.1f}ms")


if __name__ == "__main__":
    profile_dispatches()
