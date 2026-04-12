"""Interactive profile & debug tool for Mirror-SD.

Loads models once, then runs commands without reloading.

Usage:
    python -m mirror_sd.profile

Commands (type at the REPL):
    bench [--math] [--max-tokens N]    Run benchmark suite
    ane                                 Profile ANE accuracy (layer-by-layer)
    compare                             Compare GPU vs ANE draft output
    profile                             Timing profile of each kernel/phase
    prompts                             Show available prompts
    set prompt <text>                   Change active prompt
    quit                                Exit

Environment variables:
    MIRROR_SD_MODEL   Target model path (default: Qwen/Qwen3-8B)
    MIRROR_SD_DRAFT   Draft model path (default: z-lab/Qwen3-8B-DFlash-b16)
"""

import argparse
import shlex
import sys
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx


PROMPTS = {
    "math": [
        "What is 15% of 200?",
        "If a train travels 60 mph for 2.5 hours, how far does it go?",
        "Solve for x: 3x + 7 = 22",
        "Find the derivative of f(x) = x^3 + 2x^2 - 5x + 1",
    ],
    "code": [
        "Write a Python function to compute the Fibonacci sequence:",
        "Write a Python function to check if a string is a palindrome:",
        "Implement binary search in Python:",
        "What is the time complexity of merge sort?",
    ],
    "open": [
        "Why is the sky blue?",
        "Explain the theory of relativity in simple terms:",
        "What is the meaning of life?",
        "How does photosynthesis work?",
    ],
}


@dataclass
class ProfileState:
    target_model = None
    tokenizer = None
    draft_model = None
    draft_config = None
    ane_model = None
    prompt_category: str = "math"
    active_prompt: str = PROMPTS["math"][0]
    loaded: bool = False


state = ProfileState()


def load_models():
    if state.loaded:
        return
    import os
    from mlx_lm import load as mlx_load
    from .loader import load_dflash_model

    model_path = os.environ.get("MIRROR_SD_MODEL", "Qwen/Qwen3-8B")
    draft_path = os.environ.get("MIRROR_SD_DRAFT", "z-lab/Qwen3-8B-DFlash-b16")

    t0 = time.perf_counter()
    print(f"Loading target: {model_path}")
    state.target_model, state.tokenizer = mlx_load(model_path)
    print(f"Loading draft:  {draft_path}")
    state.draft_model, state.draft_config = load_dflash_model(draft_path)
    print(f"Loaded in {time.perf_counter()-t0:.1f}s")
    state.loaded = True


def load_ane():
    if state.ane_model is not None:
        return
    from .ane_model import ANEDraftModel
    bs = state.draft_config.block_size
    print(f"Initializing ANE (block_size={bs}, ctx_len=64)...")
    state.ane_model = ANEDraftModel(seq_q=bs, ctx_len=64)
    state.ane_model.load_weights(state.draft_model, state.target_model)


def prefill(prompt: Optional[str] = None) -> Tuple[mx.array, mx.array, int]:
    from mlx_lm.models import cache as cache_module
    from .dflash import extract_context_feature
    from .target import forward_with_hidden_states
    from .prompt import format_prompt

    p = prompt or state.active_prompt
    formatted = format_prompt(state.tokenizer, p)
    tokens = state.tokenizer.encode(formatted)
    input_ids = mx.array(tokens)[None]

    target_cache = cache_module.make_prompt_cache(state.target_model)
    logits, embed, hidden_states = forward_with_hidden_states(
        state.target_model, input_ids, cache=target_cache,
        capture_layers=state.draft_config.target_layer_ids,
    )
    mx.eval(logits, embed, *hidden_states)
    mx.eval([c.state for c in target_cache])

    target_hidden = extract_context_feature(hidden_states, state.draft_config.target_layer_ids)
    mx.eval(target_hidden)

    return target_hidden, logits, target_cache


def draft_input(logits, target_hidden):
    from .dflash import sample

    first_token = int(mx.argmax(logits[0, -1, :]))
    block_tokens = [first_token] + [state.draft_config.mask_token_id] * (state.draft_config.block_size - 1)
    noise_embedding = state.target_model.model.embed_tokens(mx.array([block_tokens], dtype=mx.int32))
    mx.eval(noise_embedding)
    return noise_embedding, first_token


def cmd_bench(args):
    from .dflash import sample, make_draft_mask
    from .generate import spec_generate
    from .prompt import format_prompt, get_stop_token_ids

    cat = args[0] if args else state.prompt_category
    max_tokens = 64
    for a in args:
        if a.startswith("--max-tokens="):
            max_tokens = int(a.split("=")[1])

    prompts = PROMPTS.get(cat, PROMPTS["math"])
    temperature = 0.0
    eos_ids = get_stop_token_ids(state.tokenizer) or None

    print(f"\n{'='*60}")
    print(f"  BASELINE (autoregressive, {cat} prompts)")
    print(f"{'='*60}")

    from mlx_lm.models import cache as cache_module

    baseline_results = []
    for p in prompts:
        formatted = format_prompt(state.tokenizer, p)
        tokens = state.tokenizer.encode(formatted)
        input_ids = mx.array(tokens)[None]
        cache = cache_module.make_prompt_cache(state.target_model)

        t0 = time.perf_counter()
        l = state.target_model(input_ids, cache=cache)
        mx.eval(l)
        mx.eval([c.state for c in cache])
        gen = [int(mx.argmax(l[:, -1:, :], axis=-1).flatten()[0])]
        for _ in range(max_tokens - 1):
            l = state.target_model(mx.array([[gen[-1]]]), cache=cache)
            mx.eval(l)
            t = int(mx.argmax(l[:, -1:, :], axis=-1).flatten()[0])
            gen.append(t)
        elapsed = time.perf_counter() - t0
        tps = len(gen) / max(elapsed, 1e-9)
        baseline_results.append(tps)
        short = p[:50] + ("..." if len(p) > 50 else "")
        print(f"  {short:55s} {tps:6.1f} tok/s")

    bl_avg = sum(baseline_results) / len(baseline_results)

    print(f"\n{'='*60}")
    print(f"  DFLASH (block_size={state.draft_config.block_size})")
    print(f"{'='*60}")

    dflash_results = []
    for p in prompts:
        formatted = format_prompt(state.tokenizer, p)
        tokens = state.tokenizer.encode(formatted)
        input_ids = mx.array(tokens)[None]
        output_ids, stats, _, _, _ = spec_generate(
            state.target_model, state.draft_model, input_ids,
            max_new_tokens=max_tokens, temperature=temperature,
            stop_token_ids=eos_ids,
        )
        dflash_results.append((stats, output_ids, input_ids))
        short = p[:50] + ("..." if len(p) > 50 else "")
        print(f"  {short:55s} {stats.tokens_per_sec:6.1f} tok/s  accept={stats.avg_acceptance_length:.2f}")

    df_avg = sum(r[0].tokens_per_sec for r in dflash_results) / len(dflash_results)
    df_tau = sum(r[0].avg_acceptance_length for r in dflash_results) / len(dflash_results)

    print(f"\n  Baseline:  {bl_avg:6.1f} tok/s")
    print(f"  DFlash:    {df_avg:6.1f} tok/s")
    print(f"  Speedup:   {df_avg/bl_avg:.2f}x")
    print(f"  Avg tau:  {df_tau:.2f}")


def cmd_ane(args):
    from .dflash import sample, make_draft_mask, extract_context_feature
    from .prompt import format_prompt, get_stop_token_ids

    load_ane()

    target_hidden, logits, _ = prefill()
    noise_embedding, first_token = draft_input(logits, target_hidden)
    ctx_len = target_hidden.shape[1]

    # GPU reference
    draft_cache = state.draft_model.make_cache()
    draft_mask = make_draft_mask(state.draft_config.block_size, ctx_len, 0)
    gpu_hidden = state.draft_model(
        noise_embedding=noise_embedding, target_hidden=target_hidden,
        mask=draft_mask, cache=draft_cache,
    )
    gpu_logits = state.target_model.lm_head(gpu_hidden[:, -15:, :])
    gpu_tokens = sample(gpu_logits, 0.0)
    mx.eval(gpu_hidden, gpu_logits, gpu_tokens)

    # ANE
    t0 = time.perf_counter()
    ane_hidden = state.ane_model.forward(noise_embedding, target_hidden, rope_offset=0, ctx_len=ctx_len)
    ane_logits = state.target_model.lm_head(ane_hidden[:, -15:, :])
    ane_tokens = sample(ane_logits, 0.0)
    mx.eval(ane_hidden, ane_logits, ane_tokens)
    ane_time = time.perf_counter() - t0

    # Metrics
    gpu_top1 = gpu_tokens[0].tolist()
    ane_top1 = ane_tokens[0].tolist()
    match_count = sum(1 for g, a in zip(gpu_top1, ane_top1) if g == a)

    gf = gpu_hidden[:, -15:, :].reshape(-1).astype(mx.float32)
    af = ane_hidden[:, -15:, :].reshape(-1).astype(mx.float32)
    hidden_cos = float(mx.sum(gf * af) / (mx.sqrt(mx.sum(gf**2)) * mx.sqrt(mx.sum(af**2))))

    gl = gpu_logits.reshape(-1).astype(mx.float32)
    al = ane_logits.reshape(-1).astype(mx.float32)
    logit_cos = float(mx.sum(gl * al) / (mx.sqrt(mx.sum(gl**2)) * mx.sqrt(mx.sum(al**2))))

    print(f"\n  ANE draft:     {ane_time*1000:.0f}ms")
    print(f"  Hidden cosine: {hidden_cos:.4f}")
    print(f"  Logit cosine:  {logit_cos:.4f}")
    print(f"  Top-1 match:   {match_count}/15")

    # Per-position
    print(f"\n  {'pos':>4s}  {'GPU':>7s}  {'ANE':>7s}  {'match':>5s}")
    print(f"  {'-'*28}")
    for i in range(len(gpu_top1)):
        m = "Y" if gpu_top1[i] == ane_top1[i] else ""
        print(f"  {i:4d}  {gpu_top1[i]:7d}  {ane_top1[i]:7d}  {m:>5s}")

    # fc_norm accuracy
    fc_out = state.draft_model.fc(target_hidden.astype(mx.float32))
    fc_normed = state.draft_model.hidden_norm(fc_out)
    mx.eval(fc_normed)

    ane_model = state.ane_model
    ane_model._write_mlx_2d(ane_model.b_target, target_hidden / 2048.0)
    ane_model._compute_rope(0, ctx_len)
    ane_model._compute_attn_mask(ctx_len)
    ane_model.kernels['fc_norm'].run_uncached(
        [ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm],
        [ane_model.b_context],
    )
    from .ane_model import HIDDEN
    ctx_data = ane_model.b_context.read_f32()
    ctx_ane = mx.array(ctx_data, dtype=mx.float32).reshape(1, HIDDEN, 1, 64)[:, :, :, :ctx_len].transpose(0, 3, 1, 2).reshape(1, ctx_len, HIDDEN)
    fc_cos = float(mx.sum(fc_normed.reshape(-1).astype(mx.float32) * ctx_ane.reshape(-1)) /
                   (mx.sqrt(mx.sum(fc_normed.reshape(-1).astype(mx.float32)**2)) * mx.sqrt(mx.sum(ctx_ane.reshape(-1)**2))))
    print(f"\n  fc_norm cosine: {fc_cos:.4f}")

    # Per-layer accuracy (GPU fc_norm, ANE layers)
    for n_layers in [1, 2, 3, 5]:
        ane_model._write_mlx_2d(ane_model.b_context, fc_normed)
        ane_model._write_mlx_2d(ane_model.b_hidden, noise_embedding)
        ane_model._compute_rope(0, ctx_len)
        ane_model._compute_attn_mask(ctx_len)
        for i in range(n_layers):
            ane_model._run_layer(ane_model.kernels, i)
        ane_model.kernels['final_norm'].run_uncached(
            [ane_model.b_hidden, ane_model.w_final_norm],
            [ane_model.b_output],
        )
        ah = ane_model._read_mlx_2d(ane_model.b_output, 16, HIDDEN)
        al2 = state.target_model.lm_head(ah[:, -15:, :])
        at2 = sample(al2, 0.0)
        mx.eval(ah, al2, at2)
        m2 = sum(1 for g, a in zip(gpu_top1, at2[0].tolist()) if g == a)
        af2 = ah[:, -15:, :].reshape(-1).astype(mx.float32)
        c2 = float(mx.sum(gf * af2) / (mx.sqrt(mx.sum(gf**2)) * mx.sqrt(mx.sum(af2**2))))
        label = "  (full)" if n_layers == 5 else ""
        print(f"  {n_layers} ANE layer{'s' if n_layers > 1 else ' '}: cos={c2:.4f}  top1={m2}/15{label}")


def cmd_compare(args):
    from .dflash import sample, make_draft_mask
    from .prompt import format_prompt

    target_hidden, logits, _ = prefill()
    noise_embedding, first_token = draft_input(logits, target_hidden)
    ctx_len = target_hidden.shape[1]

    # GPU
    draft_cache = state.draft_model.make_cache()
    dm = make_draft_mask(state.draft_config.block_size, ctx_len, 0)
    gpu_h = state.draft_model(noise_embedding=noise_embedding, target_hidden=target_hidden, mask=dm, cache=draft_cache)
    gpu_l = state.target_model.lm_head(gpu_h[:, -15:, :])
    gpu_t = sample(gpu_l, 0.0)
    mx.eval(gpu_h, gpu_l, gpu_t)

    # GPU timings
    draft_cache2 = state.draft_model.make_cache()
    t0 = time.perf_counter()
    for _ in range(5):
        gpu_h2 = state.draft_model(noise_embedding=noise_embedding, target_hidden=target_hidden, mask=dm, cache=draft_cache2)
        gpu_l2 = state.target_model.lm_head(gpu_h2[:, -15:, :])
        mx.eval(gpu_h2, gpu_l2)
    gpu_draft_time = (time.perf_counter() - t0) / 5

    # Target verify timing
    from mlx_lm.models import cache as cache_module
    from .target import forward_with_hidden_states
    tc = cache_module.make_prompt_cache(state.target_model)
    p = format_prompt(state.tokenizer, state.active_prompt)
    tokens = state.tokenizer.encode(p)
    input_ids = mx.array(tokens)[None]
    l, _, _ = forward_with_hidden_states(state.target_model, input_ids, cache=tc, capture_layers=state.draft_config.target_layer_ids)
    mx.eval(l)
    mx.eval([c.state for c in tc])

    verify_ids = mx.array([list(range(input_ids.shape[1], input_ids.shape[1]+16))], dtype=mx.int32)
    t0 = time.perf_counter()
    for _ in range(5):
        l2, _, h2 = forward_with_hidden_states(state.target_model, verify_ids, cache=tc, capture_layers=state.draft_config.target_layer_ids)
        mx.eval(l2, *h2)
        mx.eval([c.state for c in tc])
        cache_module.trim_prompt_cache(tc, 16)
    verify_time = (time.perf_counter() - t0) / 5

    # Single-token decode time
    t0 = time.perf_counter()
    for _ in range(10):
        l3 = state.target_model(mx.array([[state.tokenizer.eos_token_id]]), cache=tc)
        mx.eval(l3)
    decode_time = (time.perf_counter() - t0) / 10

    gpu_top1 = gpu_t[0].tolist()
    print(f"\n  GPU draft:     {gpu_draft_time*1000:.1f}ms")
    print(f"  Target verify: {verify_time*1000:.1f}ms")
    print(f"  Target decode: {decode_time*1000:.1f}ms (1 tok)")
    print(f"  DFlash iter:   {(gpu_draft_time + verify_time)*1000:.1f}ms")
    print(f"  GPU top-1:     {gpu_top1}")

    # ANE comparison
    load_ane()
    ane_model = state.ane_model
    t0 = time.perf_counter()
    ane_h = ane_model.forward(noise_embedding, target_hidden, rope_offset=0, ctx_len=ctx_len)
    ane_l = state.target_model.lm_head(ane_h[:, -15:, :])
    ane_t = sample(ane_l, 0.0)
    mx.eval(ane_h, ane_l, ane_t)
    ane_time = time.perf_counter() - t0

    ane_top1 = ane_t[0].tolist()
    match = sum(1 for g, a in zip(gpu_top1, ane_top1) if g == a)
    gf = gpu_h[:, -15:, :].reshape(-1).astype(mx.float32)
    af = ane_h[:, -15:, :].reshape(-1).astype(mx.float32)
    cos = float(mx.sum(gf * af) / (mx.sqrt(mx.sum(gf**2)) * mx.sqrt(mx.sum(af**2))))

    print(f"\n  ANE draft:     {ane_time*1000:.0f}ms")
    print(f"  ANE top-1:    {ane_top1}")
    print(f"  Top-1 match:  {match}/15")
    print(f"  Hidden cos:   {cos:.4f}")

    # Viability analysis
    print(f"\n  === Viability ===")
    for n_ane_layers in [1, 2, 5]:
        layer_time = ane_time / 5 * n_ane_layers
        for mode_name, budget in [("verify", verify_time), ("decode", decode_time)]:
            overlaps = layer_time <= budget
            print(f"  {n_ane_layers}L ANE ({layer_time*1000:.0f}ms) vs {mode_name} ({budget*1000:.0f}ms): {'OVERLAPS' if overlaps else 'NO'}")

    tau_needed = 3.8
    p_needed = tau_needed / (tau_needed + 1)
    print(f"  For ANE to beat GPU DFlash (71 tok/s): need tau>={tau_needed:.1f}, p(match)>={p_needed:.2f}")
    print(f"  Current ANE: p(match)={match/15:.2f}")


def cmd_profile(args):
    from .dflash import sample, make_draft_mask

    target_hidden, logits, _ = prefill()
    noise_embedding, first_token = draft_input(logits, target_hidden)
    ctx_len = target_hidden.shape[1]

    # GPU draft timing
    draft_cache = state.draft_model.make_cache()
    dm = make_draft_mask(state.draft_config.block_size, ctx_len, 0)

    # Warmup
    h = state.draft_model(noise_embedding=noise_embedding, target_hidden=target_hidden, mask=dm, cache=draft_cache)
    mx.eval(h)

    draft_cache2 = state.draft_model.make_cache()
    t0 = time.perf_counter()
    for _ in range(10):
        h = state.draft_model(noise_embedding=noise_embedding, target_hidden=target_hidden, mask=dm, cache=draft_cache2)
        l = state.target_model.lm_head(h[:, -15:, :])
        mx.eval(h, l)
    gpu_draft = (time.perf_counter() - t0) / 10

    # Verify timing
    from mlx_lm.models import cache as cache_module
    from .target import forward_with_hidden_states
    from .prompt import format_prompt

    tc = cache_module.make_prompt_cache(state.target_model)
    p = format_prompt(state.tokenizer, state.active_prompt)
    tokens = state.tokenizer.encode(p)
    input_ids = mx.array(tokens)[None]
    l, _, _ = forward_with_hidden_states(state.target_model, input_ids, cache=tc, capture_layers=state.draft_config.target_layer_ids)
    mx.eval(l)
    mx.eval([c.state for c in tc])

    verify_ids = mx.array([list(range(input_ids.shape[1], input_ids.shape[1]+16))], dtype=mx.int32)
    t0 = time.perf_counter()
    for _ in range(5):
        l2, _, h2 = forward_with_hidden_states(state.target_model, verify_ids, cache=tc, capture_layers=state.draft_config.target_layer_ids)
        mx.eval(l2, *h2)
        mx.eval([c.state for c in tc])
        cache_module.trim_prompt_cache(tc, 16)
    verify_time = (time.perf_counter() - t0) / 5

    # Single-token decode
    t0 = time.perf_counter()
    for _ in range(10):
        l3 = state.target_model(mx.array([[state.tokenizer.eos_token_id]]), cache=tc)
        mx.eval(l3)
    decode_time = (time.perf_counter() - t0) / 10

    # Baseline tok/s
    baseline_tps = 1.0 / decode_time

    # DFlash theoretical
    iteration_time = gpu_draft + verify_time
    dflash_tps_1 = (1 + 1) / iteration_time
    dflash_tps_6 = (6 + 1) / iteration_time

    print(f"\n  === Timing Profile ===")
    print(f"  Target decode (1 tok):  {decode_time*1000:6.1f}ms")
    print(f"  Target verify (16 tok): {verify_time*1000:6.1f}ms")
    print(f"  Draft forward (GPU):    {gpu_draft*1000:6.1f}ms")
    print(f"  DFlash iteration:       {iteration_time*1000:6.1f}ms")
    print(f"\n  === Throughput ===")
    print(f"  Baseline:               {baseline_tps:6.1f} tok/s")
    print(f"  DFlash (tau=1):         {dflash_tps_1:6.1f} tok/s ({dflash_tps_1/baseline_tps:.2f}x)")
    print(f"  DFlash (tau=6):         {dflash_tps_6:6.1f} tok/s ({dflash_tps_6/baseline_tps:.2f}x)")
    print(f"  Breakeven tau:          {gpu_draft/decode_time:.2f}")

    # ANE timing
    try:
        load_ane()
        ane_model = state.ane_model
        # Warmup
        ah = ane_model.forward(noise_embedding, target_hidden, rope_offset=0, ctx_len=ctx_len)
        mx.eval(ah)
        # Measure
        t0 = time.perf_counter()
        for _ in range(3):
            ah = ane_model.forward(noise_embedding, target_hidden, rope_offset=0, ctx_len=ctx_len)
            mx.eval(ah)
        ane_time = (time.perf_counter() - t0) / 3

        print(f"\n  === ANE ===")
        print(f"  ANE forward:            {ane_time*1000:6.1f}ms")
        print(f"  ANE per layer:          {ane_time*1000/5:6.1f}ms")
        print(f"  ANE vs verify:          {'OVERLAPS' if ane_time <= verify_time else 'NO'} ({ane_time*1000:.0f}ms vs {verify_time*1000:.0f}ms)")
    except Exception as e:
        print(f"\n  ANE unavailable: {e}")


def cmd_prompts(args):
    print("\n  Available prompt categories:")
    for cat, ps in PROMPTS.items():
        marker = " <--" if cat == state.prompt_category else ""
        print(f"  {cat}{marker}")
        for i, p in enumerate(ps):
            print(f"    {i}: {p}")
    print(f"\n  Active: {state.active_prompt}")


def cmd_set(args):
    if not args:
        print("  Usage: set prompt <text> | set category <math|code|open>")
        return
    if args[0] == "prompt" and len(args) > 1:
        state.active_prompt = " ".join(args[1:])
        print(f"  Prompt set to: {state.active_prompt}")
    elif args[0] == "category" and len(args) > 1:
        cat = args[1]
        if cat in PROMPTS:
            state.prompt_category = cat
            state.active_prompt = PROMPTS[cat][0]
            print(f"  Category: {cat}, prompt: {state.active_prompt}")
        else:
            print(f"  Unknown category: {cat}")
    else:
        print("  Usage: set prompt <text> | set category <math|code|open>")


COMMANDS = {
    "bench": cmd_bench,
    "ane": cmd_ane,
    "compare": cmd_compare,
    "profile": cmd_profile,
    "prompts": cmd_prompts,
    "set": cmd_set,
    "help": lambda args: print(__doc__),
}


def main():
    load_models()

    if len(sys.argv) > 1:
        cmd = sys.argv[1]
        args = sys.argv[2:]
        if cmd in COMMANDS:
            COMMANDS[cmd](args)
            return

    print(__doc__)
    while True:
        try:
            line = input("\nmirror-sd> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not line:
            continue
        if line in ("quit", "exit", "q"):
            break
        parts = shlex.split(line)
        cmd = parts[0]
        args = parts[1:]
        if cmd in COMMANDS:
            COMMANDS[cmd](args)
        else:
            print(f"  Unknown command: {cmd}. Type 'help' for commands.")


if __name__ == "__main__":
    main()
