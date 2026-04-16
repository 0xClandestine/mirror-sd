# mirror-sd

DFlash block-diffusion speculative decoding on Apple Silicon, with an ANE||GPU heterogeneous execution path.

The draft model runs on the Neural Engine in parallel with the target model on the GPU, using W8A16 quantized CoreML kernels compiled in Rust.

## The Story

This started as an MLX port of DFlash to prove block-diffusion speculative decoding works on Apple Silicon. It does — 3.55x on Qwen3-8B. Then we pushed further.

**Phase 1 — MLX GPU port**: DFlash on MLX, both models on GPU. Proved the acceptance rates hold.

**Phase 2 — ANE port**: Compiled the draft model as CoreML graphs in Rust, offloading it to the Neural Engine so it runs in parallel with the GPU target. Hit a precision wall: the ANE operates in fp16, and per-head Q/K norms were being computed globally across all channels instead of per-head (128 dims). This produced cosine ~0.5 vs GPU reference and α=2.12.

**Phase 3 — W8A16 quantization**: Fixed the QK norm (per-head in Rust kernel), fixed the RoPE convention (neox split-half to match `mx.fast.rope(traditional=False)`), and quantized all projection weights to int8 with per-channel fp16 scales. Weight bandwidth halved, draft time dropped from 57ms to 36ms per kernel. Result: **85 tok/s on Qwen3.5-27B on M4 Max**.

## Results

**M4 Max (64GB), Qwen3.5-27B-4bit + z-lab/Qwen3.5-27B-DFlash, W8A16, block_size=32**

| Context | tok/s | α | draft/step | overlap |
|--------:|------:|--:|----------:|--------:|
| 64      | **83.6** | 17.9 | 190ms | 90% |
| 2048    | **79.2** | 17.9 | 191ms | 90% |
| 4096    | **74.2** | 17.9 | 194ms | 90% |

Draft time is flat across context lengths — the bottleneck is FFN weight bandwidth (~30ms floor), not attention.

## How It Works

The draft model (5 transformer layers) is compiled as CoreML ANE kernels with projection weights baked in as int8 constants (`constexpr_affine_dequantize`). The GPU verify step and ANE draft run concurrently — 90%+ of draft time is hidden.

```
  GPU (target model)          ANE (draft model, W8A16)
  Qwen3.5-27B-4bit            5 layers, block_size=32
  verify N tokens    <------  parallel draft generation
        unified memory: target_hidden (zero-copy IOSurface)
```

Each decode step:
1. `target.forward(block)` → verify accepted tokens, extract target_hidden
2. ANE draft runs concurrently → produces next block of draft tokens
3. Accept matching prefix + correction token

## Installation

The ANE path requires the Rust toolchain. Install it first:

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```

Then build the ANE extension and install the Python package:

```bash
UV_CONFIG_FILE=/dev/null maturin develop --manifest-path ane/Cargo.toml
pip install -e .
```

`maturin develop` compiles the Rust kernels and installs the `ane` Python extension in-place. The `UV_CONFIG_FILE=/dev/null` bypasses any local uv config that can interfere with maturin's build.

## Benchmark

```bash
python scripts/bench_ane_profile.py \
    --model ~/.omlx/models/Qwen3.5-27B-4bit \
    --draft z-lab/Qwen3.5-27B-DFlash \
    --q8 --skip-ar --skip-gpu \
    --ctx-depths 64 2048 4096 \
    --runs 3
```

First run compiles ANE kernels (~80s). Subsequent runs in the same session reuse them.

## Supported Models

| Target | Draft |
|--------|-------|
| Qwen3-4B | `z-lab/Qwen3-4B-DFlash-b16` |
| Qwen3-8B | `z-lab/Qwen3-8B-DFlash-b16` |
| Qwen3.5-4B | `z-lab/Qwen3.5-4B-DFlash` |
| Qwen3.5-9B | `z-lab/Qwen3.5-9B-DFlash` |
| Qwen3.5-27B | `z-lab/Qwen3.5-27B-DFlash` |
| LLaMA-3.1-8B-Instruct | `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` |

## Project Structure

```
mirror_sd/          MLX GPU implementation
ane/                ANE implementation (Rust + PyO3)
  src/dflash.rs     kernel graph builders (W8A16, QK-norm, neox RoPE)
  src/wrapper.rs    Python bindings
  ANE_RULES.md      compiler constraints and timing data
scripts/            benchmarks and correctness tests
```

## References

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036) — Ye et al., 2025
- [Mirror Speculative Decoding](https://arxiv.org/abs/2510.13161) — Zhao et al., 2025
- [DFlash model weights](https://huggingface.co/collections/z-lab/dflash)

## License

MIT
