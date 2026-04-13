"""CLI for Mirror-SD: DFlash speculative decoding on Apple Silicon.

Usage:
    mirror-sd generate --model Qwen/Qwen3-8B --draft z-lab/Qwen3-8B-DFlash-b16 --prompt "Hello"
    mirror-sd convert --source z-lab/Qwen3-8B-DFlash-b16 --output ./dflash-mlx
"""

import argparse
import sys
from typing import Optional

import mlx.core as mx

from .prompt import format_prompt, get_stop_token_ids


def cmd_generate(args):
    from mlx_lm import load as mlx_load
    from .dflash import DFlashDraftModel, DFlashConfig
    from .generate import spec_generate
    from .loader import load_dflash_model

    print(f"Loading target model: {args.model}")
    target_model, tokenizer = mlx_load(args.model)

    print(f"Loading DFlash draft model: {args.draft}")
    draft_model, config = load_dflash_model(args.draft)

    if args.ane:
        from .ane_model import ANEDraftModel
        print(f"[ANE] Initializing ANE draft model (ctx_len={args.ane_ctx_len})...")
        ane_model = ANEDraftModel(seq_q=config.block_size, ctx_len=args.ane_ctx_len)
        ane_model.load_weights(draft_model, target_model)
        ane_model.gpu_fallback = draft_model
        draft_model = ane_model

    if args.quantize_draft > 0 and not args.ane:
        print(f"Quantizing draft model to {args.quantize_draft}-bit...")
        nn.quantize(draft_model, bits=args.quantize_draft)
        mx.eval(draft_model.parameters())

    print(f"DFlash config: hidden={config.hidden_size}, layers={config.num_hidden_layers}, "
          f"block_size={config.block_size}, target_layers={config.target_layer_ids}")

    use_chat = not args.raw_prompt
    formatted = format_prompt(tokenizer, args.prompt) if use_chat else args.prompt
    tokens = tokenizer.encode(formatted)
    input_ids = mx.array(tokens)[None]

    stop_ids = get_stop_token_ids(tokenizer)

    print(f"\nGenerating (max {args.max_tokens} tokens, temperature={args.temperature})...")
    output_ids, stats, _, _, _ = spec_generate(
        target_model=target_model,
        draft_model=draft_model,
        input_ids=input_ids,
        max_new_tokens=args.max_tokens,
        stop_token_ids=stop_ids,
        temperature=args.temperature,
        mirror_sd=args.mirror_sd,
        failfast=args.failfast,
        failfast_tau=args.failfast_tau,
        failfast_max_spec=args.failfast_max_spec,
        num_draft_layers=args.num_draft_layers,
        adaptive_block=args.adaptive_block,
    )

    text = tokenizer.decode(output_ids[0].tolist())
    print(f"\n{text}")
    print(f"\nStats:")
    print(f"  Total tokens:     {stats.total_tokens}")
    print(f"  Draft steps:      {stats.draft_steps}")
    print(f"  Accepted tokens:  {stats.accepted_tokens}")
    print(f"  Avg acceptance:   {stats.avg_acceptance_length:.2f}")
    print(f"  Prefill time:     {stats.prefill_time:.3f}s")
    print(f"  Tokens/sec:       {stats.tokens_per_sec:.1f}")


def cmd_convert(args):
    from .loader import convert_dflash_to_mlx
    convert_dflash_to_mlx(args.source, args.output)


def main():
    parser = argparse.ArgumentParser(
        prog="mirror-sd",
        description="DFlash speculative decoding on Apple Silicon via MLX",
    )
    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # generate
    gen_parser = subparsers.add_parser("generate", help="Generate text with speculative decoding")
    gen_parser.add_argument("--model", type=str, required=True, help="Target model (mlx-lm path or HF repo)")
    gen_parser.add_argument("--draft", type=str, required=True, help="DFlash draft model path")
    gen_parser.add_argument("--prompt", type=str, default="The meaning of life is", help="Input prompt")
    gen_parser.add_argument("--max-tokens", type=int, default=128, help="Max tokens to generate")
    gen_parser.add_argument("--temperature", type=float, default=0.0, help="Sampling temperature")
    gen_parser.add_argument("--quantize-draft", type=int, default=0, help="Quantize draft to N bits (0=off)")
    gen_parser.add_argument("--ane", action="store_true", help="Run draft model on Apple Neural Engine")
    gen_parser.add_argument("--ane-ctx-len", type=int, default=64, help="Max context length for ANE draft (default: 64)")
    gen_parser.add_argument("--mirror-sd", action="store_true", help="Use Mirror-SD early-exit (prefix/suffix split + parallel draft)")
    gen_parser.add_argument("--failfast", action="store_true", help="Enable FailFast dynamic speculation length (extends draft in high-confidence regions)")
    gen_parser.add_argument("--failfast-tau", type=float, default=0.4, help="FailFast confidence threshold (default: 0.4)")
    gen_parser.add_argument("--failfast-max-spec", type=int, default=64, help="FailFast max speculation length (default: 64)")
    gen_parser.add_argument("--num-draft-layers", type=int, default=None, help="Use only the first N draft layers (1-5)")
    gen_parser.add_argument("--adaptive-block", action="store_true", help="Adaptively adjust block size based on acceptance rate")
    gen_parser.add_argument("--raw-prompt", action="store_true", help="Use raw prompt without chat template (breaks DFlash acceptance)")

    # convert
    conv_parser = subparsers.add_parser("convert", help="Convert DFlash model to MLX format")
    conv_parser.add_argument("--source", type=str, required=True, help="HuggingFace repo ID")
    conv_parser.add_argument("--output", type=str, required=True, help="Output directory")

    args = parser.parse_args()

    if args.command == "generate":
        cmd_generate(args)
    elif args.command == "convert":
        cmd_convert(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
