"""In-process DFlash vs baseline benchmark.

Methodology matches dflash-mlx/benchmark:
  - Real prompts, not synthetic filler
  - Direct in-process calls — no HTTP, no SSE jitter
  - Accurate prefill/generation separation via internal stats
  - Models reloaded between baseline and DFlash to prevent hook contamination
  - Thermal pressure check + configurable cooldown between runs
  - Token-identity verification (DFlash must match baseline output)

Usage:
    python -m mirror_sd.benchmarks.ctx_bench \\
        --model ~/.omlx/models/Qwen3.5-27B-4bit \\
        --draft z-lab/Qwen3.5-27B-DFlash \\
        --prompt "Write a Python quicksort implementation with tests." \\
        --tg 512 --runs 3

    # Sweep context depths (uses repeated prompt tokens as context filler):
    python -m mirror_sd.benchmarks.ctx_bench \\
        --model ~/.omlx/models/Qwen3.5-27B-4bit \\
        --draft z-lab/Qwen3.5-27B-DFlash \\
        --prompt-file prompts/coding.txt \\
        --tg 512 --depth 0 512 2048 4096 --runs 3 --kod
"""

import argparse
import gc
import json
import os
import platform
import subprocess
import sys
import time
from typing import Any

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from ..dflash.loader import load_dflash_model
from ..dflash.runtime import spec_generate
from ..prompt import get_stop_token_ids


# ── system helpers ─────────────────────────────────────────────────────────────

def _clear_cache() -> None:
    gc.collect()
    for fn in [lambda: mx.clear_cache(), lambda: mx.metal.clear_cache()]:
        try:
            fn()
            return
        except Exception:
            pass


def _thermal_pressure() -> str:
    try:
        out = subprocess.check_output(["pmset", "-g", "therm"], text=True, timeout=2)
        for line in out.splitlines():
            if "CPU_Scheduler_Limit" not in line:
                continue
            v = int(line.strip().split("=")[-1].strip())
            if v == 100:
                return "nominal"
            if v >= 80:
                return "fair"
            if v >= 50:
                return "serious"
            return "critical"
    except Exception:
        pass
    return "unknown"


def _hardware_info() -> dict[str, Any]:
    try:
        chip = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    except Exception:
        chip = platform.processor()
    try:
        mem_gb = int(
            subprocess.check_output(["sysctl", "-n", "hw.memsize"], text=True).strip()
        ) // (1024 ** 3)
    except Exception:
        mem_gb = None
    return {
        "chip": chip,
        "memory_gb": mem_gb,
        "mlx_version": mx.__version__,
        "python": platform.python_version(),
        "platform": platform.platform(),
    }


# ── prompt helpers ─────────────────────────────────────────────────────────────

def _build_context_tokens(base_tokens: list[int], depth: int) -> list[int]:
    """Prepend `depth` tokens of context by repeating the base prompt tokens.

    Repeating the prompt keeps the context in-distribution (real vocabulary,
    coherent structure) rather than using random tokens that artificially
    suppress draft model acceptance rate.
    """
    if depth == 0 or not base_tokens:
        return base_tokens
    reps = depth // len(base_tokens) + 1
    return (base_tokens * reps)[:depth] + base_tokens


# ── baseline run ───────────────────────────────────────────────────────────────

def _run_baseline(
    model_path: str,
    tokens: list[int],
    max_new_tokens: int,
    temperature: float,
    no_eos: bool,
) -> dict[str, Any]:
    """Load a pristine (unpatched) target model, run AR generation, unload."""
    model, tokenizer = mlx_load(os.path.expanduser(model_path))

    if no_eos:
        try:
            tokenizer.eos_token_ids = set()
        except Exception:
            pass
        try:
            tokenizer.eos_token_id = None
        except Exception:
            pass

    generated: list[int] = []
    final_resp = None
    ttft_s: float | None = None

    sampler = make_sampler(temp=temperature)

    t_start = time.perf_counter()
    for resp in stream_generate(
        model, tokenizer, tokens, max_tokens=max_new_tokens, sampler=sampler
    ):
        if ttft_s is None:
            ttft_s = time.perf_counter() - t_start
        final_resp = resp
        generated.append(int(resp.token))
    total_s = time.perf_counter() - t_start

    del model, tokenizer
    _clear_cache()

    if not generated or final_resp is None:
        return {
            "generated_tokens": 0,
            "generation_tps": 0.0,
            "prefill_tps": 0.0,
            "prefill_s": ttft_s or 0.0,
            "total_s": total_s,
            "token_ids": [],
        }

    # Prefer mlx_lm's internal timing (measured inside the generation loop,
    # more accurate than wall-clock TTFT which includes Python overhead).
    if hasattr(final_resp, "generation_tps") and float(final_resp.generation_tps) > 0:
        gen_tps = float(final_resp.generation_tps)
        prompt_tps = float(getattr(final_resp, "prompt_tps", 0.0))
        n_prompt = int(getattr(final_resp, "prompt_tokens", len(tokens)))
        prefill_s = (n_prompt / prompt_tps) if prompt_tps > 0 else (ttft_s or 0.0)
    else:
        prefill_s = ttft_s or 0.0
        gen_s = max(total_s - prefill_s, 1e-9)
        gen_tps = len(generated) / gen_s
        prompt_tps = len(tokens) / prefill_s if prefill_s > 0 else 0.0
        prefill_s = prefill_s

    return {
        "generated_tokens": len(generated),
        "generation_tps": gen_tps,
        "prefill_tps": prompt_tps,
        "prefill_s": prefill_s,
        "total_s": total_s,
        "token_ids": generated,
    }


# ── DFlash run ─────────────────────────────────────────────────────────────────

def _run_dflash(
    model_path: str,
    draft_path: str,
    tokens: list[int],
    max_new_tokens: int,
    temperature: float,
    no_eos: bool,
    block_size: int | None,
    kod: bool,
    adaptive_block: bool,
    quantize_draft: int | None,
    prefill_step_size: int,
) -> dict[str, Any]:
    """Load target + draft fresh, run spec decoding, unload."""
    target_model, tokenizer = mlx_load(os.path.expanduser(model_path))
    draft_model, config = load_dflash_model(draft_path, quantize=quantize_draft)

    if block_size is not None:
        config.block_size = block_size
        draft_model.block_size = block_size

    stop_ids = [] if no_eos else get_stop_token_ids(tokenizer)
    input_ids = mx.array(tokens)[None]

    t_start = time.perf_counter()
    output_ids, stats, _tc, _dc, _th = spec_generate(
        target_model,
        draft_model,
        input_ids,
        max_new_tokens=max_new_tokens,
        stop_token_ids=stop_ids,
        temperature=temperature,
        adaptive_block=adaptive_block,
        kod=kod,
        prefill_step_size=prefill_step_size,
    )
    total_s = time.perf_counter() - t_start

    del target_model, draft_model, tokenizer, _tc, _dc, _th
    _clear_cache()

    gen_tokens = output_ids[0, len(tokens):].tolist()

    # stats.total_time is set by spec_generate just before return;
    # use it directly so prefill/generation split is consistent with
    # the internal timing that also drives stats.tokens_per_sec.
    gen_tps = stats.tokens_per_sec
    prefill_tps = len(tokens) / stats.prefill_time if stats.prefill_time > 0 else 0.0

    spec_active_s = stats.total_draft_time + stats.total_verify_time + stats.total_rollback_time

    return {
        "generated_tokens": len(gen_tokens),
        "generation_tps": gen_tps,
        "prefill_tps": prefill_tps,
        "prefill_s": stats.prefill_time,
        "total_s": total_s,
        "token_ids": gen_tokens,
        # spec-specific stats
        "acceptance_rate": stats.acceptance_rate,
        "avg_acceptance_length": stats.avg_acceptance_length,
        "avg_spec_length": stats.avg_spec_length,
        "draft_steps": stats.draft_steps,
        "draft_time_pct": 100.0 * stats.total_draft_time / max(spec_active_s, 1e-9),
        "verify_time_pct": 100.0 * stats.total_verify_time / max(spec_active_s, 1e-9),
        "rollback_time_pct": 100.0 * stats.total_rollback_time / max(spec_active_s, 1e-9),
    }


# ── benchmark orchestration ────────────────────────────────────────────────────

def benchmark(
    *,
    model_path: str,
    draft_path: str,
    prompt: str,
    depths: list[int],
    max_new_tokens: int,
    runs: int,
    warmup: int,
    temperature: float,
    cooldown: int,
    no_eos: bool,
    block_size: int | None,
    kod: bool,
    adaptive_block: bool,
    quantize_draft: int | None,
    prefill_step_size: int,
    chat_template: bool = False,
) -> dict[str, Any]:
    # Tokenize once upfront so both baseline and DFlash see identical tokens.
    print("  Tokenizing prompt...", flush=True)
    _m, _tok = mlx_load(os.path.expanduser(model_path))
    if chat_template and hasattr(_tok, "apply_chat_template"):
        base_tokens = list(_tok.apply_chat_template(
            [{"role": "user", "content": prompt}],
            tokenize=True,
            add_generation_prompt=True,
        ))
    else:
        base_tokens = _tok.encode(prompt)
    del _m, _tok
    _clear_cache()

    warmup_tokens = base_tokens[:min(32, len(base_tokens))]
    if warmup > 0:
        print(f"  Warming up ({warmup} pass)...", flush=True)
        for _ in range(warmup):
            _run_baseline(model_path, warmup_tokens, 8, temperature, True)
            _run_dflash(
                model_path, draft_path, warmup_tokens, 8, temperature,
                True, block_size, False, False, quantize_draft, prefill_step_size,
            )
        print("  Warmup done.", flush=True)

    results_by_depth: dict[int, list[dict]] = {d: [] for d in depths}

    try:
        for run_idx in range(1, runs + 1):
            thermal = _thermal_pressure()
            if thermal not in ("nominal", "unknown"):
                print(
                    f"  WARNING: thermal pressure '{thermal}' — results may be throttled",
                    file=sys.stderr,
                )

            for depth in depths:
                tokens = _build_context_tokens(base_tokens, depth)
                print(
                    f"  Run {run_idx}/{runs}  depth={depth}  tokens={len(tokens)}",
                    flush=True,
                )

                print("    baseline...", flush=True)
                baseline = _run_baseline(
                    model_path, tokens, max_new_tokens, temperature, no_eos
                )

                print("    dflash...", flush=True)
                dflash = _run_dflash(
                    model_path, draft_path, tokens, max_new_tokens, temperature,
                    no_eos, block_size, kod, adaptive_block, quantize_draft,
                    prefill_step_size,
                )

                match = baseline["token_ids"] == dflash["token_ids"]
                if not match:
                    print(
                        f"  WARNING: token mismatch at depth={depth} run={run_idx}",
                        file=sys.stderr,
                    )

                speedup = (
                    dflash["generation_tps"] / baseline["generation_tps"]
                    if baseline["generation_tps"] > 0
                    else None
                )

                results_by_depth[depth].append({
                    "run": run_idx,
                    "thermal": thermal,
                    "prompt_tokens": len(tokens),
                    "baseline": {k: v for k, v in baseline.items() if k != "token_ids"},
                    "dflash": {k: v for k, v in dflash.items() if k != "token_ids"},
                    "token_match": match,
                    "generation_speedup": speedup,
                })

                sp_str = f"{speedup:.3f}x" if speedup is not None else "  n/a"
                match_str = "✓" if match else "✗"
                avg_spec = dflash["avg_spec_length"]
                true_ar = dflash["acceptance_rate"] / avg_spec * 100 if avg_spec > 0 else 0.0
                print(
                    f"    → baseline {baseline['generation_tps']:>7.2f} t/s  "
                    f"dflash {dflash['generation_tps']:>7.2f} t/s  "
                    f"speedup {sp_str}  "
                    f"AR {true_ar:.1f}%  "
                    f"tok/step {dflash['acceptance_rate']:.2f}  "
                    f"avg_len {dflash['avg_acceptance_length']:.2f}  "
                    f"match {match_str}",
                    flush=True,
                )
                print(
                    f"       time breakdown — "
                    f"draft {dflash['draft_time_pct']:.1f}%  "
                    f"verify {dflash['verify_time_pct']:.1f}%  "
                    f"rollback {dflash['rollback_time_pct']:.1f}%",
                    flush=True,
                )

            if cooldown > 0 and run_idx < runs:
                print(f"  Cooldown {cooldown}s...", flush=True)
                time.sleep(cooldown)
    except KeyboardInterrupt:
        completed = sum(len(v) for v in results_by_depth.values())
        if completed == 0:
            raise
        print(f"\n  Interrupted — saving {completed} completed run(s)...", flush=True)

    # Only summarize depths that have at least one completed run.
    completed_depths = [d for d in depths if results_by_depth[d]]
    summary = _summarize(results_by_depth, completed_depths)

    return {
        "hardware": _hardware_info(),
        "config": {
            "model": os.path.basename(os.path.expanduser(model_path)),
            "draft": draft_path,
            "prompt_preview": prompt[:120],
            "max_new_tokens": max_new_tokens,
            "runs": runs,
            "cooldown_s": cooldown,
            "temperature": temperature,
            "block_size": block_size,
            "kod": kod,
            "adaptive_block": adaptive_block,
            "no_eos": no_eos,
            "chat_template": chat_template,
            "depths": depths,
        },
        "summary": summary,
        "runs": {str(d): results_by_depth[d] for d in completed_depths},
    }


def _median(values: list[float]) -> float:
    s = sorted(values)
    n = len(s)
    return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2


def _summarize(
    results_by_depth: dict[int, list[dict]], depths: list[int]
) -> list[dict]:
    rows = []
    for depth in depths:
        runs = results_by_depth[depth]
        bl_tps = [r["baseline"]["generation_tps"] for r in runs]
        df_tps = [r["dflash"]["generation_tps"] for r in runs]
        speedups = [r["generation_speedup"] for r in runs if r["generation_speedup"] is not None]
        raw_ar = [r["dflash"]["acceptance_rate"] for r in runs]
        avg_spec = [r["dflash"]["avg_spec_length"] for r in runs]
        true_ar = [
            a / s * 100 if s > 0 else 0.0
            for a, s in zip(raw_ar, avg_spec)
        ]
        avg_acc_len = [r["dflash"]["avg_acceptance_length"] for r in runs]
        rows.append({
            "depth": depth,
            "prompt_tokens": runs[0]["prompt_tokens"],
            "baseline_tps_median": _median(bl_tps),
            "dflash_tps_median": _median(df_tps),
            "speedup_median": _median(speedups) if speedups else None,
            "acceptance_rate_pct_median": _median(true_ar),
            "tok_per_step_median": _median(raw_ar),
            "avg_acceptance_length_median": _median(avg_acc_len),
            "token_match_all": all(r["token_match"] for r in runs),
        })
    return rows


# ── output ─────────────────────────────────────────────────────────────────────

def _print_table(result: dict[str, Any]) -> None:
    cfg = result["config"]
    print(f"\n  Model : {cfg['model']}")
    print(f"  Draft : {cfg['draft']}")
    print(f"  KOD={cfg['kod']}  block_size={cfg['block_size']}  no_eos={cfg['no_eos']}")
    print()
    hdr = f"  {'Depth':>6}  {'Tok in':>7}  {'Baseline':>10}  {'DFlash':>10}  {'Speedup':>8}  {'AR%':>6}  {'Tok/step':>8}  {'Avg len':>7}  {'Match':>5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for s in result["summary"]:
        sp = f"{s['speedup_median']:.3f}x" if s["speedup_median"] is not None else "   n/a"
        match = "✓" if s["token_match_all"] else "✗"
        print(
            f"  {s['depth']:>6}  {s['prompt_tokens']:>7}  "
            f"{s['baseline_tps_median']:>9.2f}  "
            f"{s['dflash_tps_median']:>9.2f}  "
            f"{sp:>8}  "
            f"{s['acceptance_rate_pct_median']:>5.1f}%  "
            f"{s['tok_per_step_median']:>8.2f}  "
            f"{s['avg_acceptance_length_median']:>7.2f}  "
            f"{match:>5}"
        )
    print()


# ── CLI ────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="In-process DFlash vs baseline benchmark (real prompts, direct calls)"
    )
    parser.add_argument("--model", required=True, help="Target model path or HF repo")
    parser.add_argument("--draft", required=True, help="DFlash draft model path or HF repo")

    prompt_grp = parser.add_mutually_exclusive_group(required=True)
    prompt_grp.add_argument("--prompt", help="Prompt text")
    prompt_grp.add_argument("--prompt-file", help="Read prompt from file")

    parser.add_argument("--tg", type=int, default=512, help="Max tokens to generate (default: 512)")
    parser.add_argument("--runs", type=int, default=3, help="Measured runs per depth (default: 3)")
    parser.add_argument("--warmup", type=int, default=1, help="Discard runs before measuring, to burn in JIT (default: 1)")
    parser.add_argument("--cooldown", type=int, default=10, help="Seconds between runs (default: 10)")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument(
        "--depth", type=int, nargs="+", default=[0],
        help="Context depth(s) to benchmark. Context is filled by repeating the prompt. (default: 0)",
    )
    parser.add_argument(
        "--no-eos", action="store_true",
        help="Disable EOS — forces exactly --tg tokens regardless of what the model generates. "
             "Recommended for fair fixed-length comparison.",
    )
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--kod", action="store_true", help="Enable Kelly-Optimal Drafting")
    parser.add_argument("--adaptive-block", action="store_true")
    parser.add_argument("--quantize-draft", type=int, default=None, choices=[4, 8])
    parser.add_argument("--prefill-step-size", type=int, default=512)
    parser.add_argument(
        "--chat-template", action="store_true",
        help="Apply model chat template to prompt (matches dflash-mlx benchmark methodology)",
    )
    parser.add_argument("--save-result", help="Write JSON results to this path")
    args = parser.parse_args()

    if args.prompt_file:
        with open(args.prompt_file) as f:
            prompt = f.read().strip()
    else:
        prompt = args.prompt

    print(f"\n  Prompt : {prompt[:100]}{'...' if len(prompt) > 100 else ''}")
    print(f"  tg={args.tg}  runs={args.runs}  warmup={args.warmup}  depths={args.depth}  KOD={args.kod}  no_eos={args.no_eos}  chat_template={args.chat_template}")

    try:
        result = benchmark(
            model_path=args.model,
            draft_path=args.draft,
            prompt=prompt,
            depths=args.depth,
            max_new_tokens=args.tg,
            runs=args.runs,
            warmup=args.warmup,
            temperature=args.temperature,
            cooldown=args.cooldown,
            no_eos=args.no_eos,
            block_size=args.block_size,
            kod=args.kod,
            adaptive_block=args.adaptive_block,
            quantize_draft=args.quantize_draft,
            prefill_step_size=args.prefill_step_size,
            chat_template=args.chat_template,
        )
    except KeyboardInterrupt:
        print("  Interrupted before any results were collected.")
        return

    _print_table(result)

    if args.save_result:
        with open(args.save_result, "w") as f:
            json.dump(result, f, indent=2)
        print(f"  Saved: {args.save_result}")
    else:
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
