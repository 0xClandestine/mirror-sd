#!/usr/bin/env python3
"""
ANE DFlash Comprehensive Profile
=================================
Measures throughput, latency, per-phase/kernel timing, and acceptance stats for:
  1. AR baseline (GPU autoregressive)
  2. GPU-only DFlash spec decode
  3. ANE DFlash spec decode (parallel ANE||GPU) at multiple context depths

Usage:
    uv run python bench_ane_profile.py
    uv run python bench_ane_profile.py --max-tokens 128 --ctx-depths 64 128 256
"""

import argparse
import statistics
import time
import threading
from collections import defaultdict
from typing import Dict, List, Optional

import mlx.core as mx
import mlx.nn as nn

TARGET_MODEL  = "~/.omlx/models/Qwen3.5-27B-4bit"
DRAFT_MODEL   = "z-lab/Qwen3.5-27B-DFlash"
PROMPT        = (
    "Write a complete, production-ready Python web server using FastAPI that "
    "implements a REST API for a todo list application with full CRUD operations, "
    "authentication, and a PostgreSQL database backend."
)
MAX_TOKENS    = 256
N_WARMUP      = 1
N_RUNS        = 3
CTX_DEPTHS    = [64, 128, 256, 512]   # ANE compiled max ctx len variants
TEMPERATURE   = 0.6


# ── Profiled ANE wrapper ───────────────────────────────────────────────────────

class _ProfiledANEDraftModel:
    """Wraps ANEDraftModel, timing every forward call at the phase and kernel level."""

    def __init__(self, ane_model):
        self._m = ane_model
        self.phase_times: Dict[str, List[float]] = defaultdict(list)   # ms
        self.kernel_times: Dict[str, List[float]] = defaultdict(list)  # ms per call
        self._active = False

        # expose attributes the spec loop needs
        self.block_size    = ane_model.block_size
        self.mask_token_id = ane_model.mask_token_id
        self.config        = ane_model.config
        self.ane           = ane_model.ane    # triggers is_ane detection
        self.max_ctx_len   = ane_model.max_ctx_len
        if hasattr(ane_model, 'gpu_fallback'):
            self.gpu_fallback = ane_model.gpu_fallback

    def reset(self):
        self.phase_times.clear()
        self.kernel_times.clear()

    def enable(self):  self._active = True
    def disable(self): self._active = False

    def make_cache(self):
        return self._m.make_cache()

    def set_block_size(self, new_seq_q: int) -> None:
        self._m.set_block_size(new_seq_q)
        self.block_size = new_seq_q

    # ---- prepare_forward / run_kernels / read_output / run_prepared:
    #      delegate to the real model so the pipelined ANE||GPU path in
    #      _spec_generate_parallel can pre-buffer all Metal work on the main
    #      thread.  run_kernels() is pure ANE (called from thread),
    #      read_output() creates mx.array (called on main thread after join).
    def prepare_forward(self, noise_embedding, precomputed_context, cache,
                        target_hidden):
        return self._m.prepare_forward(noise_embedding, precomputed_context,
                                       cache, target_hidden)

    def run_kernels(self):
        if self._active:
            m = self._m
            k = m.kernels
            t0 = time.perf_counter()
            for i in range(m.n_layers):
                if m.use_q8:
                    m._run_layer_q8(i)
                else:
                    m._run_layer(k, i)
            self.phase_times['layers_total'].append(
                (time.perf_counter() - t0) * 1e3)
            t0 = time.perf_counter()
            k['final_norm'].run_uncached(
                [m.b_hidden, m.w_final_norm], [m.b_output])
            self.phase_times['final_norm'].append(
                (time.perf_counter() - t0) * 1e3)
        else:
            self._m.run_kernels()

    def read_output(self):
        if self._active:
            m = self._m
            t0 = time.perf_counter()
            result = m._read_mlx_2d(m.b_output, m.seq_q, m.hidden)
            self.phase_times['read_output'].append(
                (time.perf_counter() - t0) * 1e3)
            return result
        return self._m.read_output()

    def run_prepared(self):
        self.run_kernels()
        return self.read_output()

    def __call__(self, noise_embedding, target_hidden, cache=None, **kwargs):
        if not self._active:
            return self._m(noise_embedding, target_hidden, cache=cache, **kwargs)

        # ---- replicate ANEDraftModel.forward with per-phase timing ----
        m = self._m
        rope_offset = 0
        if cache is not None and len(cache) > 0 and cache[0].offset > 0:
            rope_offset = cache[0].offset
        ctx_len = target_hidden.shape[1]

        k = m.kernels

        t0 = time.perf_counter()
        m._write_padded(m._padded_hidden, m.b_hidden, noise_embedding)
        self.phase_times['write_noise_emb'].append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        context = m._compute_context(target_hidden)
        m._write_padded(m._padded_context, m.b_context, context)
        self.phase_times['context_fc_write'].append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        m._compute_rope(rope_offset, ctx_len)
        self.phase_times['rope_precompute'].append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        m._compute_attn_mask(ctx_len)
        self.phase_times['attn_mask'].append((time.perf_counter() - t0) * 1e3)

        per_kernel: Dict[str, List[float]] = defaultdict(list)
        t_layers_start = time.perf_counter()
        for i in range(m.n_layers):
            p = f"l{i}_"

            if m.use_q8:
                lk = m.layer_kernels[i]

                t0 = time.perf_counter()
                lk['mega_qkv_q8'].run_uncached(
                    [m.b_hidden, getattr(m, f"w_{p}in_norm"),
                     m.b_context,
                     getattr(m, f"w_{p}k_norm_4d"), m.b_cos_k, m.b_sin_k,
                     getattr(m, f"w_{p}q_norm_4d"), m.b_cos_q, m.b_sin_q],
                    [m.b_k_rope_4d, m.b_v_4d_t, m.b_q_rope_4d],
                )
                per_kernel['mega_qkv'].append((time.perf_counter() - t0) * 1e3)

                t0 = time.perf_counter()
                lk['gqa_tile'].run_uncached(
                    [m.b_k_rope_4d, m.b_v_4d_t], [m.b_kv_tiled],
                )
                per_kernel['gqa_tile'].append((time.perf_counter() - t0) * 1e3)

                t0 = time.perf_counter()
                lk['attn_out'].run_uncached(
                    [m.b_q_rope_4d, m.b_kv_tiled, m.b_attn_mask], [m.b_attn_flat],
                )
                per_kernel['attn_out'].append((time.perf_counter() - t0) * 1e3)

                t0 = time.perf_counter()
                lk['o_proj_residual_q8'].run_uncached(
                    [m.b_attn_flat, m.b_hidden], [m.b_attn_res],
                )
                per_kernel['o_proj_residual'].append((time.perf_counter() - t0) * 1e3)

                t0 = time.perf_counter()
                lk['ffn_residual_q8'].run_uncached(
                    [m.b_attn_res, getattr(m, f"w_{p}post_norm")],
                    [m.b_hidden],
                )
                per_kernel['ffn_residual'].append((time.perf_counter() - t0) * 1e3)
            else:
                t0 = time.perf_counter()
                k['mega_qkv'].run_uncached(
                    [m.b_hidden, getattr(m, f"w_{p}in_norm"),
                     m.b_context, getattr(m, f"w_{p}k_proj"),
                     getattr(m, f"w_{p}k_norm_4d"), m.b_cos_k, m.b_sin_k,
                     getattr(m, f"w_{p}v_proj"), getattr(m, f"w_{p}q_proj"),
                     getattr(m, f"w_{p}q_norm_4d"), m.b_cos_q, m.b_sin_q],
                    [m.b_k_rope_4d, m.b_v_4d_t, m.b_q_rope_4d],
                )
                per_kernel['mega_qkv'].append((time.perf_counter() - t0) * 1e3)

                t0 = time.perf_counter()
                k['gqa_tile'].run_uncached(
                    [m.b_k_rope_4d, m.b_v_4d_t], [m.b_kv_tiled],
                )
                per_kernel['gqa_tile'].append((time.perf_counter() - t0) * 1e3)

                t0 = time.perf_counter()
                k['attn_out'].run_uncached(
                    [m.b_q_rope_4d, m.b_kv_tiled, m.b_attn_mask], [m.b_attn_flat],
                )
                per_kernel['attn_out'].append((time.perf_counter() - t0) * 1e3)

                t0 = time.perf_counter()
                k['o_proj_residual'].run_uncached(
                    [m.b_attn_flat, getattr(m, f"w_{p}o_proj"), m.b_hidden],
                    [m.b_attn_res],
                )
                per_kernel['o_proj_residual'].append((time.perf_counter() - t0) * 1e3)

                t0 = time.perf_counter()
                k['ffn_residual'].run_uncached(
                    [m.b_attn_res, getattr(m, f"w_{p}post_norm"),
                     getattr(m, f"w_{p}gate"), getattr(m, f"w_{p}up"),
                     getattr(m, f"w_{p}down")],
                    [m.b_hidden],
                )
                per_kernel['ffn_residual'].append((time.perf_counter() - t0) * 1e3)

        self.phase_times['layers_total'].append(
            (time.perf_counter() - t_layers_start) * 1e3
        )

        for kname, times in per_kernel.items():
            self.kernel_times[kname].extend(times)
            self.kernel_times[f'{kname}_per_call'].append(sum(times))

        t0 = time.perf_counter()
        k['final_norm'].run_uncached([m.b_hidden, m.w_final_norm], [m.b_output])
        self.phase_times['final_norm'].append((time.perf_counter() - t0) * 1e3)

        t0 = time.perf_counter()
        result = m._read_mlx_2d(m.b_output, m.seq_q, m.hidden)
        self.phase_times['read_output'].append((time.perf_counter() - t0) * 1e3)

        return result


# ── Helpers ────────────────────────────────────────────────────────────────────

def _ms(t: float) -> str:
    return f"{t * 1e3:7.2f} ms"

def _pct(num, den) -> str:
    if den == 0:
        return "  N/A %"
    return f"{100 * num / den:5.1f}%"

def _fmt_stat(vals: List[float], unit: str = "ms") -> str:
    if not vals:
        return "      N/A"
    med = statistics.median(vals)
    mn  = min(vals)
    mx_ = max(vals)
    return f"{med:7.2f} {unit}  [min={mn:.2f}, max={mx_:.2f}]"

def _sep(char="─", width=70):
    print(char * width)

def _header(title: str):
    _sep("═")
    print(f"  {title}")
    _sep("═")

def _section(title: str):
    print()
    _sep()
    print(f"  {title}")
    _sep()


# ── Run helpers ────────────────────────────────────────────────────────────────

def run_ar(target_model, input_ids, max_tokens, stop_ids, tokenizer=None, temperature=TEMPERATURE):
    from mirror_sd.generate import ar_generate
    output_ids, stats = ar_generate(
        target_model, input_ids, max_new_tokens=max_tokens,
        stop_token_ids=stop_ids, temperature=temperature,
    )
    if tokenizer is not None:
        gen_tokens = output_ids[0, input_ids.shape[1]:].tolist()
        print(f"\n  Output: {tokenizer.decode(gen_tokens, skip_special_tokens=True)!r}")
    return stats


def run_spec(target_model, draft_model, input_ids, max_tokens, stop_ids, tokenizer=None, temperature=TEMPERATURE):
    from mirror_sd.generate import spec_generate
    output_ids, stats, *_ = spec_generate(
        target_model, draft_model, input_ids,
        max_new_tokens=max_tokens,
        stop_token_ids=stop_ids,
        temperature=temperature,
    )
    if tokenizer is not None:
        gen_tokens = output_ids[0, input_ids.shape[1]:].tolist()
        print(f"\n  Output: {tokenizer.decode(gen_tokens, skip_special_tokens=True)!r}")
    return stats


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="ANE DFlash profile")
    parser.add_argument("--model",      default=TARGET_MODEL)
    parser.add_argument("--draft",      default=DRAFT_MODEL)
    parser.add_argument("--max-tokens", type=int, default=MAX_TOKENS)
    parser.add_argument("--warmup",     type=int, default=N_WARMUP)
    parser.add_argument("--runs",       type=int, default=N_RUNS)
    parser.add_argument("--ctx-depths", type=int, nargs="+", default=CTX_DEPTHS)
    parser.add_argument("--skip-ar",    action="store_true",
                        help="Skip AR baseline (saves ~5 min)")
    parser.add_argument("--skip-gpu",   action="store_true",
                        help="Skip GPU-only spec decode")
    parser.add_argument("--skip-ane",   action="store_true",
                        help="Skip ANE spec decode section")
    parser.add_argument("--block-size", type=int, default=16,
                        help="ANE draft block size (default: 16; all 1-64 use identical kernels)")
    parser.add_argument("--q8", action="store_true",
                        help="Use W8A16 int8-quantized ANE kernels")
    parser.add_argument("--temperature", type=float, default=None,
                        help=f"Sampling temperature (default: {TEMPERATURE})")
    args = parser.parse_args()

    import os
    model_path  = os.path.expanduser(args.model)
    max_tokens  = args.max_tokens
    n_warmup    = args.warmup
    n_runs      = args.runs
    ctx_depths  = args.ctx_depths
    ane_block_size = args.block_size
    temperature = TEMPERATURE if args.temperature is None else args.temperature

    _header("ANE DFlash Comprehensive Profile")
    print(f"  Target model : {model_path}")
    print(f"  Draft model  : {args.draft}")
    print(f"  Max tokens   : {max_tokens}")
    print(f"  Warmup runs  : {n_warmup}")
    print(f"  Timed runs   : {n_runs}")
    print(f"  CTX depths   : {ctx_depths}")
    print(f"  Prompt len   : {len(PROMPT)} chars")
    print(f"  Temperature  : {temperature}")

    # ── Load models ───────────────────────────────────────────────────────────
    _section("Loading models")
    t0 = time.perf_counter()
    from mlx_lm import load as mlx_load
    target_model, tokenizer = mlx_load(model_path)
    print(f"  Target loaded in {time.perf_counter()-t0:.1f}s")

    t0 = time.perf_counter()
    from mirror_sd.loader import load_dflash_model
    gpu_draft, config = load_dflash_model(args.draft)
    print(f"  Draft loaded  in {time.perf_counter()-t0:.1f}s")
    print(f"  Draft config  : hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"block_size={config.block_size} (ANE override→{ane_block_size}), heads={config.num_attention_heads}")

    from mirror_sd.prompt import format_prompt, get_stop_token_ids
    stop_ids   = get_stop_token_ids(tokenizer)
    formatted  = format_prompt(tokenizer, PROMPT)
    tokens     = tokenizer.encode(formatted)
    input_ids  = mx.array(tokens)[None]
    print(f"  Encoded prompt: {input_ids.shape[1]} tokens")

    # ── Section 1: AR baseline ────────────────────────────────────────────────
    ar_results: List = []
    if not args.skip_ar:
        _section("1 / AR Baseline (GPU autoregressive)")
        print(f"  Warmup ({n_warmup})…", flush=True)
        for _ in range(n_warmup):
            run_ar(target_model, input_ids, max_tokens, stop_ids, temperature=temperature)

        print(f"  Timing ({n_runs} runs)…", flush=True)
        for i in range(n_runs):
            s = run_ar(target_model, input_ids, max_tokens, stop_ids,
                       tokenizer=tokenizer if i == 0 else None, temperature=temperature)
            ar_results.append(s)
            gen_s = s.total_time - s.prefill_time
            tps   = s.total_tokens / max(gen_s, 1e-9)
            print(f"    run {i+1}: {s.total_tokens} tok, {tps:.1f} tok/s, "
                  f"prefill={s.prefill_time*1e3:.0f}ms, gen={gen_s*1e3:.0f}ms")
    else:
        print("\n  [AR baseline skipped]")

    # ── Section 2: GPU-only spec decode ───────────────────────────────────────
    gpu_results: List = []
    if not args.skip_gpu:
        _section("2 / GPU-only DFlash Spec Decode")
        print(f"  Warmup ({n_warmup})…", flush=True)
        for _ in range(n_warmup):
            run_spec(target_model, gpu_draft, input_ids, max_tokens, stop_ids, temperature=temperature)

        print(f"  Timing ({n_runs} runs)…", flush=True)
        for i in range(n_runs):
            s = run_spec(target_model, gpu_draft, input_ids, max_tokens, stop_ids,
                         tokenizer=tokenizer if i == 0 else None, temperature=temperature)
            gpu_results.append(s)
            gen_s = s.total_time - s.prefill_time
            tps   = s.total_tokens / max(gen_s, 1e-9)
            print(f"    run {i+1}: {s.total_tokens} tok, {tps:.1f} tok/s, "
                  f"α={s.avg_acceptance_length:.2f}, "
                  f"draft={s.total_draft_time*1e3:.0f}ms, "
                  f"verify={s.total_verify_time*1e3:.0f}ms")
    else:
        print("\n  [GPU spec decode skipped]")

    # ── Section 3: ANE spec decode at each CTX depth ──────────────────────────
    ane_runs_by_depth: Dict[int, List] = {}
    ane_profiles:      Dict[int, _ProfiledANEDraftModel] = {}

    _section("3 / ANE DFlash Spec Decode (parallel ANE || GPU)")
    if args.skip_ane:
        print("  [ANE spec decode skipped]")
    else:
        from mirror_sd.ane_model import ANEDraftModel

    for ctx_len in ([] if args.skip_ane else ctx_depths):
        print(f"\n  ── ctx_len={ctx_len} ──", flush=True)

        t0 = time.perf_counter()
        raw_ane = ANEDraftModel(seq_q=ane_block_size, ctx_len=ctx_len, config=config)
        if args.q8:
            raw_ane.load_weights_q8(gpu_draft)
        else:
            raw_ane.load_weights(gpu_draft, target_model)
        raw_ane.gpu_fallback = gpu_draft
        print(f"  ANE compiled+loaded in {time.perf_counter()-t0:.1f}s")

        profiled = _ProfiledANEDraftModel(raw_ane)

        # Warmup (profiling off)
        profiled.disable()
        print(f"  Warmup ({n_warmup})…", flush=True)
        for _ in range(n_warmup):
            run_spec(target_model, profiled, input_ids, max_tokens, stop_ids, temperature=temperature)

        # Timed runs (profiling on)
        profiled.enable()
        profiled.reset()
        run_list = []
        print(f"  Timing ({n_runs} runs)…", flush=True)
        for i in range(n_runs):
            s = run_spec(target_model, profiled, input_ids, max_tokens, stop_ids,
                         tokenizer=tokenizer if i == 0 else None, temperature=temperature)
            run_list.append(s)
            gen_s = s.total_time - s.prefill_time
            tps   = s.total_tokens / max(gen_s, 1e-9)
            print(f"    run {i+1}: {s.total_tokens} tok, {tps:.1f} tok/s, "
                  f"α={s.avg_acceptance_length:.2f}, "
                  f"draft={s.total_draft_time*1e3:.0f}ms, "
                  f"verify={s.total_verify_time*1e3:.0f}ms, "
                  f"overlap={s.total_overlap_time*1e3:.0f}ms")

        profiled.disable()
        ane_runs_by_depth[ctx_len] = run_list
        ane_profiles[ctx_len]      = profiled

    # ══════════════════════════════════════════════════════════════════════════
    #  REPORT
    # ══════════════════════════════════════════════════════════════════════════
    _header("PROFILE REPORT")

    # ── 1. Throughput summary ─────────────────────────────────────────────────
    _section("Throughput Summary  (median tok/s over timed runs)")
    print(f"  {'Mode':<30}  {'tok/s':>8}  {'prefill':>10}  {'gen time':>10}  speedup")
    _sep("-", 70)

    def _tps_row(label, results, ar_tps=None):
        if not results:
            print(f"  {label:<30}  {'N/A':>8}")
            return None
        tps_list = [s.total_tokens / max(s.total_time - s.prefill_time, 1e-9)
                    for s in results]
        pre_list = [s.prefill_time * 1e3 for s in results]
        gen_list = [(s.total_time - s.prefill_time) * 1e3 for s in results]
        med_tps  = statistics.median(tps_list)
        spd = f"{med_tps / ar_tps:5.2f}x" if ar_tps else "  ref"
        print(f"  {label:<30}  {med_tps:>8.1f}  "
              f"{statistics.median(pre_list):>8.0f}ms  "
              f"{statistics.median(gen_list):>8.0f}ms  {spd}")
        return med_tps

    ar_tps = None
    if ar_results:
        ar_tps = _tps_row("AR baseline (GPU autoregressive)", ar_results)
    if gpu_results:
        _tps_row("GPU-only DFlash spec decode", gpu_results, ar_tps)
    for ctx in ctx_depths:
        if ctx in ane_runs_by_depth:
            _tps_row(f"ANE DFlash  ctx_len={ctx:<4}", ane_runs_by_depth[ctx], ar_tps)

    # ── 2. Acceptance statistics ───────────────────────────────────────────────
    _section("Acceptance Statistics")
    print(f"  {'Mode':<30}  {'avg α':>7}  {'min':>5}  {'max':>5}  {'std':>6}  draft_steps")
    _sep("-", 70)

    def _acc_row(label, results):
        if not results:
            return
        all_lens = [l for s in results for l in s.acceptance_lengths]
        steps    = sum(s.draft_steps for s in results)
        if not all_lens:
            return
        print(f"  {label:<30}  {statistics.mean(all_lens):>7.2f}  "
              f"{min(all_lens):>5d}  {max(all_lens):>5d}  "
              f"{statistics.stdev(all_lens) if len(all_lens)>1 else 0:>6.2f}  {steps:>6d}")

    if gpu_results:
        _acc_row("GPU-only DFlash", gpu_results)
    for ctx in ctx_depths:
        if ctx in ane_runs_by_depth:
            _acc_row(f"ANE DFlash  ctx_len={ctx:<4}", ane_runs_by_depth[ctx])

    # acceptance length histogram (use largest ctx ANE run)
    best_ctx   = max(ane_runs_by_depth.keys()) if ane_runs_by_depth else None
    if best_ctx is not None:
        all_lens = [l for s in ane_runs_by_depth[best_ctx] for l in s.acceptance_lengths]
        from collections import Counter
        cnt = Counter(all_lens)
        total = sum(cnt.values())
        print(f"\n  Acceptance length histogram  (ANE ctx={best_ctx}, n={total} steps):")
        for length in sorted(cnt):
            bar = "█" * int(40 * cnt[length] / total)
            print(f"    α={length:2d}  {cnt[length]:4d} ({100*cnt[length]/total:5.1f}%)  {bar}")

    # ── 3. Per-step timing: draft vs verify vs overlap ─────────────────────────
    _section("Per-Step Timing Breakdown  (all ANE runs, all steps, median ms)")
    print(f"  {'Mode':<26}  {'draft/step':>10}  {'verify/step':>11}  "
          f"{'overlap%':>8}  {'eff%':>6}")
    _sep("-", 70)

    def _timing_row(label, results):
        if not results:
            return
        draft_per_step  = []
        verify_per_step = []
        overlap_frac    = []
        for s in results:
            n = max(s.draft_steps, 1)
            draft_per_step.append(s.total_draft_time / n * 1e3)
            verify_per_step.append(s.total_verify_time / n * 1e3)
            if s.total_draft_time > 0:
                overlap_frac.append(s.total_overlap_time / s.total_draft_time * 100)
        med_d = statistics.median(draft_per_step)
        med_v = statistics.median(verify_per_step)
        med_o = statistics.median(overlap_frac) if overlap_frac else 0.0
        # "efficiency" = draft fully hidden = overlap/draft
        eff   = med_o
        print(f"  {label:<26}  {med_d:>10.1f}  {med_v:>11.1f}  "
              f"{med_o:>8.1f}  {eff:>6.1f}")

    if gpu_results:
        _timing_row("GPU-only DFlash", gpu_results)
    for ctx in ctx_depths:
        if ctx in ane_runs_by_depth:
            _timing_row(f"ANE ctx={ctx}", ane_runs_by_depth[ctx])

    # ── 4. ANE forward pass phase breakdown ────────────────────────────────────
    _section("ANE Forward Pass Phase Breakdown  (per draft call, ms)")
    print(f"  Phase                   {' '.join(f'ctx={c:<6}' for c in ctx_depths if c in ane_profiles)}")
    _sep("-", 70)

    phases = [
        'write_noise_emb',
        'context_fc_write',
        'rope_precompute',
        'attn_mask',
        'layers_total',
        'final_norm',
        'read_output',
    ]
    phase_labels = {
        'write_noise_emb'  : 'write noise embedding',
        'context_fc_write' : 'context FC + write',
        'rope_precompute'  : 'RoPE precompute',
        'attn_mask'        : 'attn mask precompute',
        'layers_total'     : 'all layers (total)',
        'final_norm'       : 'final norm',
        'read_output'      : 'read output buffer',
    }
    active_depths = [c for c in ctx_depths if c in ane_profiles]

    for ph in phases:
        label = phase_labels.get(ph, ph)
        row = f"  {label:<24}"
        for ctx in active_depths:
            vals = ane_profiles[ctx].phase_times.get(ph, [])
            if vals:
                row += f"  {statistics.median(vals):>7.2f}"
            else:
                row += f"  {'N/A':>7}"
        print(row)

    # total forward time
    print()
    for ctx in active_depths:
        ph_total = ane_profiles[ctx].phase_times
        total_vals = [
            sum(ph_total.get(ph, [0])[i] for ph in phases
                if i < len(ph_total.get(ph, [])))
            for i in range(min(len(ph_total.get('layers_total', [1])),
                               len(ph_total.get('write_noise_emb', [1]))))
        ]
        if total_vals:
            print(f"  Total ANE forward  ctx={ctx}: median {statistics.median(total_vals):.2f}ms "
                  f"[min={min(total_vals):.2f}, max={max(total_vals):.2f}]")

    # ── 5. Per-kernel breakdown ────────────────────────────────────────────────
    use_q8 = any(ane_profiles[c]._m.use_q8 for c in active_depths if c in ane_profiles)
    _section(f"ANE Per-Kernel Timing  (summed over all layers per call, ms){'  [W8A16 q8]' if use_q8 else ''}")
    kernels = ['mega_qkv', 'gqa_tile', 'attn_out', 'o_proj_residual', 'ffn_residual']
    k_labels = {
        'mega_qkv'        : ('mega_qkv_q8 (Q/K/V+RoPE, w8a16)' if use_q8 else 'mega_qkv (Q/K/V + norms + RoPE)'),
        'gqa_tile'        : 'gqa_tile (GQA KV expand)',
        'attn_out'        : 'attn_out (SDPA)',
        'o_proj_residual' : ('o_proj_residual_q8 (w8a16)' if use_q8 else 'o_proj_residual'),
        'ffn_residual'    : ('ffn_residual_q8 (MLP, w8a16)' if use_q8 else 'ffn_residual (MLP)'),
    }
    print(f"  Kernel                      {' '.join(f'ctx={c:<6}' for c in active_depths)}")
    _sep("-", 70)
    for kn in kernels:
        label = k_labels.get(kn, kn)
        row = f"  {label:<28}"
        for ctx in active_depths:
            key  = f'{kn}_per_call'
            vals = ane_profiles[ctx].kernel_times.get(key, [])
            if vals:
                row += f"  {statistics.median(vals):>7.2f}"
            else:
                row += f"  {'N/A':>7}"
        print(row)

    # per-kernel as % of layers_total
    print()
    print("  As % of total layer time:")
    for ctx in active_depths:
        layer_total = statistics.median(ane_profiles[ctx].phase_times.get('layers_total', [1]))
        parts = []
        for kn in kernels:
            key  = f'{kn}_per_call'
            vals = ane_profiles[ctx].kernel_times.get(key, [])
            if vals:
                pct  = statistics.median(vals) / max(layer_total, 1e-9) * 100
                parts.append(f"{kn}={pct:.1f}%")
        print(f"    ctx={ctx}: {', '.join(parts)}")

    # ── 6. Draft thread overlap analysis ──────────────────────────────────────
    _section("Overlap / Pipeline Efficiency")
    for ctx in active_depths:
        results = ane_runs_by_depth.get(ctx, [])
        if not results:
            continue
        total_draft  = sum(s.total_draft_time  for s in results)
        total_verify = sum(s.total_verify_time for s in results)
        total_overlap = sum(s.total_overlap_time for s in results)
        total_steps  = sum(s.draft_steps for s in results)
        # "hidden" = time draft ran concurrently with verify
        pct_hidden   = total_overlap / max(total_draft,  1e-9) * 100
        # "bottleneck" = which phase is longer
        bottleneck   = "verify" if total_verify > total_draft else "draft"
        ideal_gain   = (total_draft + total_verify) / max(total_verify, 1e-9)
        print(f"  ctx={ctx:3d}: draft={total_draft/total_steps*1e3:.1f}ms/step  "
              f"verify={total_verify/total_steps*1e3:.1f}ms/step  "
              f"overlap={total_overlap/total_steps*1e3:.1f}ms/step  "
              f"hidden={pct_hidden:.1f}%  bottleneck={bottleneck}  "
              f"pipeline_gain={ideal_gain:.2f}x")

    # ── 7. Memory estimate ────────────────────────────────────────────────────
    _section("Memory Footprint  (ANE buffer allocation estimate)")
    for ctx in active_depths:
        if ctx not in ane_profiles:
            continue
        m = ane_profiles[ctx]._m
        # sum all ANETensor allocated sizes
        bufs = [
            m.b_noise, m.b_target, m.b_context, m.b_hidden,
            m.b_k_rope_4d, m.b_v_4d_t, m.b_q_rope_4d,
            m.b_kv_tiled, m.b_attn_flat, m.b_attn_res, m.b_output,
            m.b_cos_q, m.b_sin_q, m.b_cos_k, m.b_sin_k, m.b_attn_mask,
        ]
        # shape tuple: (n, c, h, w) → n*c*h*w * 4 bytes (float32)
        total_bytes = 0
        for buf in bufs:
            s = buf.shape
            total_bytes += s[0] * s[1] * s[2] * s[3] * 4
        print(f"  ctx={ctx:3d}: ANE activation buffers ≈ {total_bytes/1024**2:.1f} MB")

    _header("END OF REPORT")


if __name__ == "__main__":
    main()
