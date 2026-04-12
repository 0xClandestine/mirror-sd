"""Context length scaling benchmark.

Measures decode throughput at varying context lengths by padding prompts
to target lengths, then measuring token generation speed. This reveals how
KV cache size affects both baseline and speculative decoding performance.

Usage:
    python -m mirror_sd.ctx_bench --model Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16
    python -m mirror_sd.ctx_bench --model ~/.omlx/models/Qwen3.5-27B-4bit \
        --draft ~/.cache/huggingface/hub/models--z-lab--Qwen3.5-27B-DFlash/snapshots/... \
        --block-size 4 --kod
"""

import argparse
import time

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm.models import cache as cache_module

from .dflash import sample
from .generate import spec_generate, _flat_cache_states
from .loader import load_dflash_model
from .prompt import format_prompt, get_stop_token_ids


def _pad_prompt(tokenizer, base_prompt: str, target_ctx: int, use_chat: bool) -> mx.array:
    """Pad a prompt to target_ctx tokens by repeating a filler sentence."""
    if use_chat:
        filler = "The quick brown fox jumps over the lazy dog. "
        padded = base_prompt
        while True:
            formatted = format_prompt(tokenizer, padded)
            n = len(tokenizer.encode(formatted))
            if n >= target_ctx:
                break
            padded += filler
    else:
        filler = "The quick brown fox jumps over the lazy dog. "
        padded = base_prompt
        while len(tokenizer.encode(padded)) < target_ctx:
            padded += filler

    formatted = format_prompt(tokenizer, padded) if use_chat else padded
    tokens = tokenizer.encode(formatted)

    if len(tokens) > target_ctx:
        tokens = tokens[:target_ctx]

    return mx.array(tokens)[None], len(tokens)


def measure_baseline(model, tokenizer, input_ids, gen_tokens: int):
    cache = cache_module.make_prompt_cache(model)

    t0 = time.perf_counter()
    logits = model(input_ids, cache=cache)
    first_token = mx.argmax(logits[:, -1:, :], axis=-1)
    mx.eval(first_token)
    generated = [int(first_token[0, 0])]

    eos_ids = _eos_ids(tokenizer)

    for _ in range(gen_tokens - 1):
        if generated[-1] in eos_ids:
            break
        logits = model(mx.array([[generated[-1]]]), cache=cache)
        next_token = mx.argmax(logits[:, -1:, :], axis=-1)
        mx.eval(next_token)
        generated.append(int(next_token[0, 0]))

    t1 = time.perf_counter()
    elapsed = t1 - t0
    n_gen = len(generated)
    return n_gen, elapsed


def _eos_ids(tokenizer):
    ids = set()
    if hasattr(tokenizer, "eos_token_id") and tokenizer.eos_token_id is not None:
        if isinstance(tokenizer.eos_token_id, list):
            ids.update(tokenizer.eos_token_id)
        else:
            ids.add(tokenizer.eos_token_id)
    return ids


def main():
    parser = argparse.ArgumentParser(description="Context length scaling benchmark")
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--draft", type=str, default=None)
    parser.add_argument("--block-size", type=int, default=None)
    parser.add_argument("--gen-tokens", type=int, default=64, help="Tokens to generate per measurement (default: 64)")
    parser.add_argument("--ctx-lengths", type=str, default=None,
                        help="Comma-separated context lengths (default: auto 128..8192)")
    parser.add_argument("--no-baseline", action="store_true", help="Skip baseline autoregressive")
    parser.add_argument("--no-spec", action="store_true", help="Skip speculative decoding")
    parser.add_argument("--no-adaptive", action="store_true")
    parser.add_argument("--kod", action="store_true")
    parser.add_argument("--quantize-draft", type=int, default=None, choices=[4, 8])
    parser.add_argument("--raw-prompt", action="store_true")
    parser.add_argument("--think", action="store_true")
    args = parser.parse_args()

    if not args.no_spec and args.draft is None:
        parser.error("--draft is required unless --no-spec is set")

    print(f"Loading target: {args.model}")
    model, tokenizer = mlx_load(args.model)

    draft_model = None
    config = None
    if not args.no_spec:
        print(f"Loading draft:  {args.draft}")
        draft_model, config = load_dflash_model(args.draft, quantize=args.quantize_draft)
        if args.block_size is not None:
            config.block_size = args.block_size
            draft_model.block_size = args.block_size

    use_chat = not args.raw_prompt
    eos_ids = get_stop_token_ids(tokenizer) or None

    base_prompt = "Explain the theory of relativity in simple terms."

    if args.ctx_lengths:
        ctx_lengths = [int(x.strip()) for x in args.ctx_lengths.split(",")]
    else:
        ctx_lengths = [128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]

    gen_tokens = args.gen_tokens

    print(f"\nGen tokens per measurement: {gen_tokens}")
    print(f"\n{'='*80}")

    baseline_results = []
    spec_results = []

    for ctx_len in ctx_lengths:
        input_ids, actual_ctx = _pad_prompt(tokenizer, base_prompt, ctx_len, use_chat)
        actual_ctx = input_ids.shape[1]

        cur_gen = gen_tokens

        print(f"\n  ctx={actual_ctx:5d} tokens", end="", flush=True)

        if not args.no_baseline:
            n, elapsed = measure_baseline(model, tokenizer, input_ids, cur_gen)
            bl_tps = n / max(elapsed, 1e-9)
            bl_ms_tok = 1000.0 * elapsed / max(n, 1)
            baseline_results.append((actual_ctx, bl_tps, bl_ms_tok))
            print(f"  baseline={bl_tps:5.1f} tok/s ({bl_ms_tok:.1f}ms/tok)", end="", flush=True)

        if not args.no_spec:
            output_ids, stats, _, _, _ = spec_generate(
                model, draft_model, input_ids,
                max_new_tokens=cur_gen,
                temperature=0.0,
                stop_token_ids=eos_ids,
                adaptive_block=not args.no_adaptive,
                kod=args.kod,
            )
            sp_tps = stats.tokens_per_sec
            sp_accept = stats.avg_acceptance_length
            n_gen = stats.total_tokens
            decode_time = stats.total_time - stats.prefill_time
            sp_ms_tok = 1000.0 * decode_time / max(n_gen, 1)
            spec_results.append((actual_ctx, sp_tps, sp_ms_tok, sp_accept))
            print(f"  spec={sp_tps:5.1f} tok/s ({sp_ms_tok:.1f}ms/tok) accept={sp_accept:.2f}", flush=True)
        else:
            print(flush=True)

    print(f"\n{'='*80}")
    print(f"  SUMMARY")
    print(f"{'='*80}")
    print(f"  {'Ctx':>6s}", end="")
    if baseline_results:
        print(f"  {'BL tok/s':>9s} {'BL ms/tok':>9s}", end="")
    if spec_results:
        print(f"  {'SP tok/s':>9s} {'SP ms/tok':>9s} {'Accept':>7s} {'Speedup':>7s}", end="")
    print()
    print(f"  {'-'*70}")

    for i, ctx in enumerate(ctx_lengths):
        actual = baseline_results[i][0] if i < len(baseline_results) else (spec_results[i][0] if i < len(spec_results) else ctx)
        print(f"  {actual:6d}", end="")
        if i < len(baseline_results):
            print(f"  {baseline_results[i][1]:9.1f} {baseline_results[i][2]:9.1f}", end="")
        if i < len(spec_results):
            speedup = spec_results[i][1] / max(baseline_results[i][1], 1e-9) if i < len(baseline_results) else 0
            print(f"  {spec_results[i][1]:9.1f} {spec_results[i][2]:9.1f} {spec_results[i][3]:7.2f} {speedup:7.2f}x", end="")
        print()


if __name__ == "__main__":
    main()
