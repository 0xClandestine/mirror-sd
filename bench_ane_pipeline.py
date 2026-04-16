#!/usr/bin/env python3
"""
bench_ane_pipeline.py — Full ANE draft step pipeline profiler.

Times every phase of one ANE draft step in isolation, replicating the exact
calling sequence from _spec_generate_parallel._start_draft / join / post-join:

  [main thread, pre-fork]
    embed          — embed_fn(last_token + mask * (block_size-1))
    soft_anchor    — weighted embedding sum (skipped on first step; ~same cost)
    compute_context— _compute_context(target_hidden)  GPU matmul + mx.eval
    prepare_forward— write all ANE input buffers (IOSurface copies)
    thread_launch  — threading.Thread.start() overhead

  [ANE thread]
    run_kernels    — CoreML ANE dispatch for all layers (this is the ~57ms)

  [main thread, post-join]
    join_overhead  — wall time from thread.start() to thread.join() returning,
                     minus run_kernels time (scheduling / GIL overhead)
    read_output    — _read_mlx_2d: IOSurface → mx.array
    lm_head_eval   — lm_head_fn(draft_hidden) + mx.argmax + mx.eval
                     (uses synthetic weight unless --target is given)

Usage:
    uv run python bench_ane_pipeline.py
    uv run python bench_ane_pipeline.py --ctx-len 256 --runs 30
    uv run python bench_ane_pipeline.py --target ~/.omlx/models/Qwen3.5-27B-4bit
"""

import argparse
import statistics
import threading
import time
from collections import defaultdict

import mlx.core as mx
import mlx.nn as nn

DRAFT_MODEL_NAME = "z-lab/Qwen3-8B-DFlash-b16"
DEFAULT_CTX_LEN = 64
DEFAULT_SEQ_Q = 16
N_WARMUP = 5
N_RUNS = 30

PHASE_ORDER = [
    "embed",
    "soft_anchor",
    "compute_context",
    "prepare_forward",
    "thread_launch",
    "run_kernels",
    "join_overhead",
    "read_output",
    "lm_head_eval",
]

PHASE_LABELS = {
    "embed":           "embed tokens                 [GPU]",
    "soft_anchor":     "soft anchor (weighted emb)   [GPU]",
    "compute_context": "_compute_context (FC matmul) [GPU]",
    "prepare_forward": "prepare_forward (buf writes) [CPU→ANE]",
    "thread_launch":   "Thread.start() overhead      [CPU]",
    "run_kernels":     "run_kernels (ANE dispatch)   [ANE]",
    "join_overhead":   "join overhead (sched/GIL)    [CPU]",
    "read_output":     "read_output (buf→mx.array)   [ANE→CPU]",
    "lm_head_eval":    "lm_head + argmax + eval      [GPU]",
}


def _sep(c="─", w=74): print(c * w)
def _header(t): _sep("═"); print(f"  {t}"); _sep("═")
def _section(t): print(); _sep(); print(f"  {t}"); _sep()


def _make_soft_anchor(noise_emb, correction_logits, embed_fn, top_k=8):
    """Replicate the soft-anchor computation from _start_draft."""
    top_ids = mx.argsort(-correction_logits)[:top_k]
    top_logits = correction_logits[top_ids]
    weights = mx.softmax(top_logits.astype(mx.float32))
    top_embeds = embed_fn(top_ids.reshape(1, top_k))
    soft = mx.sum(weights.reshape(1, top_k, 1) * top_embeds, axis=1, keepdims=True)
    soft = soft.astype(noise_emb.dtype)
    return mx.concatenate([soft, noise_emb[:, 1:, :]], axis=1)


_deferred_stream = None  # initialized in main()

def run_one(ane, embed_fn, lm_head_fn, target_hidden, draft_cache,
            last_token, mask_token_id, block_size, correction_logits,
            use_ane_lm_head: bool = False, deferred_lm_head: bool = False):
    """Run one full draft step with fine-grained phase timing.

    Replicates _start_draft + thread run + post-join sequence exactly.
    Returns dict[phase_name -> ms].
    """
    phases: dict[str, float] = {}

    # ── Pre-fork: main thread ──────────────────────────────────────────────

    # 1. Embed tokens
    t0 = time.perf_counter()
    block_tokens = [last_token] + [mask_token_id] * (block_size - 1)
    noise_emb = embed_fn(mx.array([block_tokens], dtype=mx.int32))
    mx.eval(noise_emb)
    phases["embed"] = (time.perf_counter() - t0) * 1e3

    # 2. Soft anchor (uses correction_logits from a simulated prior verify)
    t0 = time.perf_counter()
    noise_emb = _make_soft_anchor(noise_emb, correction_logits, embed_fn)
    mx.eval(noise_emb)
    phases["soft_anchor"] = (time.perf_counter() - t0) * 1e3

    # 3. _compute_context — GPU matmul hidden→context
    t0 = time.perf_counter()
    precomputed_context = ane._compute_context(target_hidden)
    mx.eval(precomputed_context)
    phases["compute_context"] = (time.perf_counter() - t0) * 1e3

    # 4. prepare_forward — write all ANE input buffers on main thread
    t0 = time.perf_counter()
    ane.prepare_forward(noise_emb, precomputed_context, draft_cache, target_hidden)
    phases["prepare_forward"] = (time.perf_counter() - t0) * 1e3

    # ── Fork: launch pure-ANE thread ──────────────────────────────────────

    kernel_ms: list[float] = []

    def _thread_fn():
        t = time.perf_counter()
        ane.run_kernels()
        kernel_ms.append((time.perf_counter() - t) * 1e3)

    t0 = time.perf_counter()
    thread = threading.Thread(target=_thread_fn)
    thread.start()
    phases["thread_launch"] = (time.perf_counter() - t0) * 1e3

    # ── Join ──────────────────────────────────────────────────────────────

    t_join_start = time.perf_counter()
    thread.join()
    t_join_end = time.perf_counter()
    wall_join = (t_join_end - t_join_start) * 1e3

    phases["run_kernels"]   = kernel_ms[0] if kernel_ms else 0.0
    phases["join_overhead"] = max(0.0, wall_join - phases["run_kernels"])

    # ── Post-join: main thread ────────────────────────────────────────────

    # 5. read_output — IOSurface buffer → mx.array (skipped for ANE lm_head path)
    if use_ane_lm_head and hasattr(ane, 'b_logits') and ane.b_logits is not None:
        phases["read_output"] = 0.0   # ANE lm_head path: b_logits holds logits, not hidden
        draft_hidden = None
    else:
        t0 = time.perf_counter()
        draft_hidden = ane.read_output()
        phases["read_output"] = (time.perf_counter() - t0) * 1e3

    # 6. lm_head + sample: either ANE (read_draft_tokens) or GPU
    t0 = time.perf_counter()
    if use_ane_lm_head and hasattr(ane, 'read_draft_tokens') and ane.b_logits is not None:
        # NEON argmax directly on the ANE IOSurface — no GPU, no mx.eval.
        raw = ane.read_draft_tokens()   # (1, seq_q) int32, already on CPU
        _ = raw[:, 1:block_size]        # slice to draft positions (lazy but cheap)
        phases["lm_head_eval"] = (time.perf_counter() - t0) * 1e3
        phases["lm_head_path"] = 0.0    # sentinel: 0 = ANE path
    elif deferred_lm_head:
        # Simulate the _draft_stream deferred path from generate.py:
        # submit to a separate GPU stream without eval.  Only measures
        # submission overhead — the actual 7ms matmul is "free" during verify.
        q_len = block_size
        with mx.stream(_deferred_stream):
            draft_logits = lm_head_fn(draft_hidden[:, -(q_len - 1):, :])
            sampled = mx.argmax(draft_logits, axis=-1)
        # No mx.eval here — matches the new generate.py behavior.
        phases["lm_head_eval"] = (time.perf_counter() - t0) * 1e3
        phases["lm_head_path"] = 2.0    # sentinel: 2 = deferred GPU path
        # Force a barrier so the next run starts clean.
        mx.eval(sampled)
    else:
        q_len = block_size
        draft_logits = lm_head_fn(draft_hidden[:, -(q_len - 1):, :])
        sampled = mx.argmax(draft_logits, axis=-1)
        mx.eval(sampled)
        phases["lm_head_eval"] = (time.perf_counter() - t0) * 1e3
        phases["lm_head_path"] = 1.0    # sentinel: 1 = GPU path (immediate)

    return phases


def main():
    parser = argparse.ArgumentParser(description="ANE draft step pipeline profiler")
    parser.add_argument("--model",    default=DRAFT_MODEL_NAME)
    parser.add_argument("--ctx-len",  type=int, default=DEFAULT_CTX_LEN)
    parser.add_argument("--seq-q",    type=int, default=DEFAULT_SEQ_Q)
    parser.add_argument("--warmup",   type=int, default=N_WARMUP)
    parser.add_argument("--runs",     type=int, default=N_RUNS)
    parser.add_argument("--target",   default=None,
                        help="Path to target model for real lm_head. "
                             "Omit to use a synthetic weight (timing proxy).")
    parser.add_argument("--ane-lm-head", action="store_true",
                        help="Compile fused final_norm+lm_head on ANE and use "
                             "read_draft_tokens() instead of GPU lm_head. "
                             "NOTE: fails for vocab>=~16K (ANE channel limit). "
                             "Requires --target for real lm_head weights.")
    parser.add_argument("--deferred-lm-head", action="store_true",
                        help="Simulate deferred lm_head: submit to _draft_stream "
                             "without eval, measure only submission overhead "
                             "(the 7ms matmul runs concurrently with verify).")
    args = parser.parse_args()

    _header("ANE Draft Step Pipeline Profiler")
    print(f"  Draft model : {args.model}")
    print(f"  ctx_len     : {args.ctx_len}   seq_q : {args.seq_q}")
    print(f"  Warmup      : {args.warmup}    Runs  : {args.runs}")
    print(f"  lm_head     : {'real (target model)' if args.target else 'synthetic (proxy timing)'}")
    print(f"  ANE lm_head : {'enabled' if args.ane_lm_head else 'disabled (GPU path)'}")
    print(f"  Deferred lh : {'yes (_draft_stream, no eval)' if args.deferred_lm_head else 'no (immediate mx.eval)'}")

    global _deferred_stream
    _deferred_stream = mx.new_stream(mx.gpu)

    _section("Loading models")
    import os
    from mirror_sd.loader import load_dflash_model
    from mirror_sd.ane_model import ANEDraftModel
    from mirror_sd.target import get_embed_tokens, get_lm_head

    t0 = time.perf_counter()
    gpu_draft, config = load_dflash_model(args.model)
    print(f"  GPU draft loaded in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    vocab_size = getattr(config, 'vocab_size', None) if args.ane_lm_head else None
    ane = ANEDraftModel(seq_q=args.seq_q, ctx_len=args.ctx_len, config=config,
                        vocab_size=vocab_size)
    ane.load_weights(gpu_draft)
    print(f"  ANE compiled+loaded in {time.perf_counter()-t0:.1f}s")

    if args.target:
        from mlx_lm import load as mlx_load
        t0 = time.perf_counter()
        target_model, _ = mlx_load(os.path.expanduser(args.target))
        print(f"  Target loaded in {time.perf_counter()-t0:.1f}s")
        embed_fn   = get_embed_tokens(target_model)
        lm_head_fn = get_lm_head(target_model)
    else:
        # Synthetic embed + lm_head sized to match the 8B draft model
        vocab_size = getattr(config, "vocab_size", 151936)
        H = config.hidden_size
        _embed_w = mx.random.normal((vocab_size, H), dtype=mx.float16) * 0.02
        _lm_w    = mx.random.normal((vocab_size, H), dtype=mx.float16) * 0.02
        mx.eval(_embed_w, _lm_w)
        def embed_fn(ids):
            return _embed_w[ids]
        def lm_head_fn(h):
            return (h.astype(mx.float16) @ _lm_w.T).astype(mx.float32)

    # ── Build synthetic inputs ─────────────────────────────────────────────
    mx.random.seed(42)
    H = config.hidden_size
    target_hidden = mx.random.normal(
        (1, args.ctx_len, 5 * H), dtype=mx.float32
    ) * 0.02
    # Simulate a correction_logits vector (proxy for post-verify logits)
    correction_logits = mx.random.normal((config.vocab_size,), dtype=mx.float32)
    mx.eval(target_hidden, correction_logits)

    draft_cache = ane.make_cache()
    last_token = 100
    mask_token_id = ane.mask_token_id
    block_size = ane.block_size  # = seq_q = 16

    # ── Warmup ────────────────────────────────────────────────────────────
    _section(f"Warmup ({args.warmup} runs)")
    for i in range(args.warmup):
        # Reset rope/mask caches to simulate fresh step
        ane._rope_cache_key = None
        ane._attn_mask_ctx_len = None
        run_one(ane, embed_fn, lm_head_fn, target_hidden, draft_cache,
                last_token, mask_token_id, block_size, correction_logits,
                use_ane_lm_head=args.ane_lm_head,
                deferred_lm_head=args.deferred_lm_head)
        print(f"  warmup {i+1}/{args.warmup}", end="\r", flush=True)
    print()

    # ── Timed runs ────────────────────────────────────────────────────────
    _section(f"Timed runs ({args.runs} runs)")
    all_phases: dict[str, list[float]] = defaultdict(list)

    for i in range(args.runs):
        ane._rope_cache_key = None
        ane._attn_mask_ctx_len = None
        p = run_one(ane, embed_fn, lm_head_fn, target_hidden, draft_cache,
                    last_token, mask_token_id, block_size, correction_logits,
                    use_ane_lm_head=args.ane_lm_head,
                    deferred_lm_head=args.deferred_lm_head)
        for k, v in p.items():
            all_phases[k].append(v)
        total = sum(p[ph] for ph in PHASE_ORDER if ph in p)
        print(f"  run {i+1:3d}/{args.runs}: total={total:.1f}ms", end="\r", flush=True)
    print()

    # ── Report ────────────────────────────────────────────────────────────
    _header("PIPELINE REPORT")
    _section("Per-phase latency  (median over timed runs)")

    print(f"  {'Phase':<45}  {'median':>7}  {'min':>7}  {'max':>7}  {'%step':>6}")
    _sep("-", 74)

    # compute total using medians
    total_med = sum(
        statistics.median(all_phases[ph]) for ph in PHASE_ORDER if ph in all_phases
    )

    ane_total = 0.0
    gpu_total = 0.0
    cpu_total = 0.0
    ane_bus_total = 0.0

    for ph in PHASE_ORDER:
        vals = all_phases.get(ph, [])
        if not vals:
            continue
        med = statistics.median(vals)
        mn  = min(vals)
        mx_ = max(vals)
        pct = med / max(total_med, 1e-9) * 100
        label = PHASE_LABELS.get(ph, ph)
        print(f"  {label:<45}  {med:>7.2f}  {mn:>7.2f}  {mx_:>7.2f}  {pct:>5.1f}%")

        # Tally by executor
        if "[GPU]" in label:
            gpu_total += med
        elif "[ANE]" in label and "→" not in label:
            ane_total += med
        elif "[ANE→CPU]" in label or "[CPU→ANE]" in label:
            ane_bus_total += med
        else:
            cpu_total += med

    _sep("-", 74)
    print(f"  {'TOTAL step (measured)':<45}  {total_med:>7.2f}")

    _section("By executor")
    print(f"  GPU (embed + context + lm_head)   : {gpu_total:>7.2f} ms  "
          f"({100*gpu_total/max(total_med,1e-9):.1f}%)")
    print(f"  ANE (run_kernels)                  : {ane_total:>7.2f} ms  "
          f"({100*ane_total/max(total_med,1e-9):.1f}%)")
    print(f"  ANE bus (prepare + read)           : {ane_bus_total:>7.2f} ms  "
          f"({100*ane_bus_total/max(total_med,1e-9):.1f}%)")
    print(f"  CPU overhead (launch + join)       : {cpu_total:>7.2f} ms  "
          f"({100*cpu_total/max(total_med,1e-9):.1f}%)")
    print()
    print(f"  Reference: run_kernels alone (bench_ane_power.py)  ~57 ms")
    kernels_med = statistics.median(all_phases.get("run_kernels", [0]))
    overhead_ms = total_med - kernels_med
    print(f"  Non-kernel overhead this bench     : {overhead_ms:>7.2f} ms")
    print(f"  (= total {total_med:.1f}ms − run_kernels {kernels_med:.1f}ms)")

    _section("Config summary (paste into optimization-attempts.md)")
    print(f"  model={args.model}  ctx_len={args.ctx_len}  seq_q={args.seq_q}")
    print(f"  step_total={total_med:.2f}ms  run_kernels={kernels_med:.2f}ms  "
          f"overhead={overhead_ms:.2f}ms")
    print(f"  gpu={gpu_total:.2f}ms  ane_bus={ane_bus_total:.2f}ms  "
          f"cpu_overhead={cpu_total:.2f}ms")
    print()
    _sep("═")


if __name__ == "__main__":
    main()
