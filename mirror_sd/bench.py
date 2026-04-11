"""Benchmark: autoregressive baseline vs DFlash speculative decoding.

Usage:
    python -m mirror_sd.bench --model Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16
    python -m mirror_sd.bench --model Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 --ane
    python -m mirror_sd.bench --model Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 --prompt "Explain quantum computing:" --max-tokens 256
"""

import argparse
import time

import mlx.core as mx

from mlx_lm import load as mlx_load
from mlx_lm.models import cache as cache_module

from .dflash import sample, make_draft_mask
from .loader import load_dflash_model
from .target import forward_with_hidden_states, extract_context_feature
from .generate import spec_generate


def baseline_generate(model, tokenizer, prompt: str, max_tokens: int, temperature: float = 0.0):
    tokens = tokenizer.encode(prompt)
    input_ids = mx.array(tokens)[None]
    cache = cache_module.make_prompt_cache(model)

    t0 = time.perf_counter()

    logits = model(input_ids, cache=cache)
    mx.eval(logits)
    mx.eval([c.state for c in cache])

    next_token = mx.argmax(logits[:, -1:, :], axis=-1)
    mx.eval(next_token)
    generated = [int(next_token[0, 0])]

    eos_ids = _eos_ids(tokenizer)

    for _ in range(max_tokens - 1):
        if generated[-1] in eos_ids:
            break
        logits = model(mx.array([[generated[-1]]]), cache=cache)
        mx.eval(logits)
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(next_token)
        generated.append(int(next_token[0, 0]))

    elapsed = time.perf_counter() - t0
    n_gen = len(generated)
    return tokenizer.decode(generated), n_gen / max(elapsed, 1e-9)


def _eos_ids(tokenizer):
    ids = set()
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            ids.update(tokenizer.eos_token_id)
        else:
            ids.add(tokenizer.eos_token_id)
    return ids


PROMPTS = [
    "The capital of France is",
    "Explain the theory of relativity in simple terms:",
    "Write a Python function to sort a list:",
    "What is the meaning of life?",
    "How does photosynthesis work?",
]

MATH_CODE_PROMPTS = [
    "What is 15% of 200?",
    # "If a train travels 60 mph for 2.5 hours, how far does it go?",
    # "Solve for x: 3x + 7 = 22",
    # "Write a Python function to compute the Fibonacci sequence:",
    # "Write a Python function to check if a string is a palindrome:",
    # "Implement binary search in Python:",
    # "What is the time complexity of merge sort?",
    # "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
]


def main():
    parser = argparse.ArgumentParser(description="Benchmark: baseline vs DFlash speculative decoding")
    parser.add_argument("--model", type=str, required=True, help="Target model (e.g. Qwen/Qwen3-8B)")
    parser.add_argument("--draft", type=str, required=True, help="DFlash draft model (e.g. z-lab/Qwen3-8B-DFlash-b16)")
    parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate per prompt")
    parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    parser.add_argument("--prompt", type=str, default=None, help="Single prompt (default: built-in suite)")
    parser.add_argument("--warmup", type=int, default=1, help="Warmup rounds before measuring")
    parser.add_argument("--math", action="store_true", help="Use math/code prompts (paper's training distribution)")
    parser.add_argument("--block-size", type=int, default=None, help="Override draft block size")
    parser.add_argument("--ane", action="store_true", help="Run draft model on Apple Neural Engine")
    parser.add_argument("--ane-ctx-len", type=int, default=64, help="Max context length for ANE draft (default: 64)")
    args = parser.parse_args()

    print(f"Loading target: {args.model}")
    target_model, tokenizer = mlx_load(args.model)
    print(f"Loading draft:  {args.draft}")
    draft_model, config = load_dflash_model(args.draft)
    if args.block_size is not None:
        config.block_size = args.block_size
        draft_model.block_size = args.block_size

    if args.ane:
        from .ane_model import ANEDraftModel
        print(f"[ANE] Initializing ANE draft model (ctx_len={args.ane_ctx_len})...")
        ane_model = ANEDraftModel(seq_q=config.block_size, ctx_len=args.ane_ctx_len)
        ane_model.load_weights(draft_model, target_model)
        draft_model = ane_model

    if args.prompt:
        prompts = [args.prompt]
    elif args.math:
        prompts = MATH_CODE_PROMPTS
    else:
        prompts = PROMPTS
    max_tokens = args.max_tokens
    temperature = args.temperature
    eos_ids = list(_eos_ids(tokenizer)) or None

    # Warmup
    for _ in range(args.warmup):
        p = prompts[0]
        tokens = tokenizer.encode(p)
        input_ids = mx.array(tokens)[None]
        spec_generate(target_model, draft_model, input_ids, max_new_tokens=16, temperature=temperature, stop_token_ids=eos_ids)

    # --- Baseline ---
    print(f"\n{'='*60}")
    print(f"  BASELINE (autoregressive)")
    print(f"{'='*60}")

    baseline_results = []
    for prompt in prompts:
        text, tps = baseline_generate(target_model, tokenizer, prompt, max_tokens, temperature)
        baseline_results.append((prompt, tps))
        short = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"  {short:55s} {tps:6.1f} tok/s")

    baseline_avg = sum(r[1] for r in baseline_results) / len(baseline_results)
    print(f"  {'AVERAGE':55s} {baseline_avg:6.1f} tok/s")

    # --- DFlash / ANE ---
    mode = "ANE" if args.ane else "DFLASH"
    print(f"\n{'='*60}")
    print(f"  {mode} (speculative decoding, block_size={config.block_size})")
    print(f"{'='*60}")

    dflash_results = []
    for prompt in prompts:
        tokens = tokenizer.encode(prompt)
        input_ids = mx.array(tokens)[None]
        output_ids, stats = spec_generate(
            target_model, draft_model, input_ids,
            max_new_tokens=max_tokens,
            temperature=temperature,
            stop_token_ids=eos_ids,
        )
        dflash_results.append((prompt, stats))
        short = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"  {short:55s} {stats.tokens_per_sec:6.1f} tok/s  accept={stats.avg_acceptance_length:.2f}  steps={stats.draft_steps}")

    dflash_avg = sum(r[1].tokens_per_sec for r in dflash_results) / len(dflash_results)
    dflash_accept_avg = sum(r[1].avg_acceptance_length for r in dflash_results) / len(dflash_results)
    print(f"  {'AVERAGE':55s} {dflash_avg:6.1f} tok/s  accept={dflash_accept_avg:.2f}")

    # --- Summary ---
    speedup = dflash_avg / max(baseline_avg, 1e-9)
    print(f"\n{'='*60}")
    print(f"  SUMMARY")
    print(f"{'='*60}")
    print(f"  Baseline:   {baseline_avg:6.1f} tok/s")
    print(f"  {mode}:     {dflash_avg:6.1f} tok/s")
    print(f"  Speedup:    {speedup:6.2f}x")
    print(f"  Avg accept: {dflash_accept_avg:.2f} tokens/block")
    print(f"  Block size:  {config.block_size}")
    print(f"  Draft mode:  {'ANE' if args.ane else 'GPU'}")

    if args.ane and dflash_results:
        stats0 = dflash_results[0][1]
        if stats0.parallel_mode:
            print(f"  Draft time:  {stats0.total_draft_time:.3f}s")
            print(f"  Verify time: {stats0.total_verify_time:.3f}s")
            print(f"  Overlap:     {stats0.total_overlap_time:.3f}s")
            if stats0.total_verify_time > 0:
                print(f"  Overlap %%:   {100*stats0.total_overlap_time/stats0.total_verify_time:.1f}%")


if __name__ == "__main__":
    main()
