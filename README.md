# Mirror-SD: DFlash Speculative Decoding on Apple Silicon

DFlash block-diffusion speculative decoding running on Apple Silicon via MLX, with an ANE execution path that explores heterogeneous accelerator dispatch.

Combines [DFlash](https://arxiv.org/abs/2602.06036) block-diffusion draft models with the heterogeneous execution concept from [Mirror-SD](https://arxiv.org/abs/2510.13161).

## How It Works

DFlash replaces the autoregressive draft model in traditional speculative decoding with a block-diffusion model. Instead of generating tokens one at a time, the draft model produces an entire block of tokens (typically 16) in a single forward pass.

The draft model uses **target-aware attention**: K/V projections attend to both the target model's intermediate hidden states (context) and the draft model's own hidden states (noise). This lets the draft "see what the target has processed" while generating, enabling parallel block generation.

```
PREFILL: target(prompt) → first token + target_hidden
DECODE LOOP:
  1. Create block: [last_token, mask, mask, ...]
  2. Draft: embed_tokens(block) → DFlash → target.lm_head → sample
  3. Verify: target(block) → posterior tokens
  4. Accept matching prefix + correction token
  5. Crop caches, update target_hidden
```

## Benchmarks

M4 Max  (64GB), MLX, Qwen3.5-27B-4bit, llama-benchy with prompt caching (3 runs per depth).

![DFlash vs Baseline](benchmarks/fixed_vs_baseline.png)

| Context Depth | Baseline (tok/s) | DFlash bs=4 (tok/s) | Speedup |
|--------------:|-----------------:|--------------------:|--------:|
| 0             | 26.3             | 32.4                | 1.23x   |
| 512           | 26.6             | 47.5                | 1.78x   |
| 2048          | 17.6             | 38.5                | 2.19x   |
| 8192          | 18.1             | 25.0                | 1.38x   |
| 16384         | 15.9             | 31.1                | 1.96x   |

Fixed block_size=4 outperforms KOD (Kelly-Optimal Drafting) at every depth for this model. KOD's cost model overestimates the penalty of wasted draft compute, causing it to shrink block sizes too aggressively. With bs=4, the draft+verify pipeline consistently wins despite moderate acceptance rates.

### Qwen3-8B (informal benchmark)

| Metric | Value |
|--------|-------|
| Baseline (autoregressive) | 27.0 tok/s |
| DFlash bs=16 | 95.8 tok/s |
| **Speedup** | **3.55x** |

## MLX Implementation

The primary implementation runs both target and draft models on GPU via MLX.

### Key implementation details

- Draft model loads in **bf16** to match the target model's dtype — this was the single biggest acceptance rate improvement (+50%)
- DFlash generates blocks of tokens per forward pass via non-causal attention (block_size=16 for 8B, block_size=4 for 27B)
- 5 target hidden features extracted from layers uniformly distributed through the target model, injected into K/V of every draft layer
- Draft model shares embedding and `lm_head` with target (only transformer layers trained)
- **Fused K/V projections** — concat target_hidden + hidden_states before projection, single matmul per weight instead of two, saves ~40MB bandwidth/forward pass
- **Combined eval** — draft, verify, and cache state updates are fused into a single `mx.eval()` call, eliminating redundant GPU sync points
- **CPU-side accept/reject** — tiny sequences (2-16 elements) processed in Python instead of 5 GPU kernel dispatches
- Repetition detection prevents degenerate accept/reject loops
- **Prompt caching** — LRU cache with correct full-hit handling (re-derive target_hidden from last prefill chunk)

## ANE Execution Path

The `ane/` directory contains a Rust implementation of the DFlash draft model for Apple Neural Engine, using the [ane](https://github.com/ncdrone/ane) crate for direct ANE graph compilation.

The design: run the target model on GPU and the draft model on ANE **in parallel**, matching the Mirror-SD paper's heterogeneous accelerator design. On Apple Silicon with unified memory, the draft model's inputs (target hidden states) and outputs (draft logits) are exchanged with zero-copy.

### Results (M4 Max, Qwen3.5-27B-4bit + z-lab/Qwen3.5-27B-DFlash, ctx=64, 256 tokens)

| Mode | tok/s | α | draft/step | overlap | speedup |
|---|---:|---:|---:|---:|---:|
| GPU autoregressive (baseline) | 14.5 | — | — | — | 1.0x |
| GPU-only DFlash spec decode | ~22 | ~4.5 | ~230ms | — | ~1.5x |
| **ANE‖GPU DFlash (fp16)** | **40** | 9.6 | 231ms | 95% | **2.8x** |
| **ANE‖GPU DFlash (W8A16 q8)** | **58** | 17.9 | 191ms | 93% | **4.0x** |

W8A16 quantization cuts draft bandwidth in half (~1.2x kernel speedup) while the quantized draft model achieves higher token acceptance against the fp16 target, combining to **4x over GPU autoregressive**.

### Architecture

```
  ┌─────────────────────────┐     ┌──────────────────────────────┐
  │  GPU (target model)     │     │  ANE (draft model, W8A16)    │
  │  Qwen3.5-27B-4bit       │     │  5 compiled layers/block     │
  │  verify block (N steps) │◄────│  parallel draft generation   │
  └─────────────────────────┘     └──────────────────────────────┘
        unified memory: target_hidden (zero-copy IOSurface)
```

Each transformer layer is compiled as 5 ANE kernels with projection weights **baked in as int8 constants** (`constexpr_affine_dequantize` in CoreML MIL). Per-channel fp16 scales allow ANE-side dequantization at runtime. The GPU verify step runs concurrently on the Apple GPU while the ANE drafts.

### Key Engineering Findings

- **QK-norm was the blocker**: ANE `rmsnorm` is global (normalizes all channels at once), but Q/K norm in DFlash is per-head (128 dims/head). Running the global ANE norm produced cosine ~0.5 vs GPU reference and α=2.12. Fix: compute Q/K norms in Python before writing to ANE input buffers.
- **W8A16 via `constexpr_affine_dequantize`**: CoreML MIL requires attribute-style syntax (data inline in `[]` brackets, not positional args). Weights stored as int8 BLOBFILE with 1D fp16 per-channel scales.
- **Per-dispatch overhead**: 0.05ms (run_cached/XPC bypass), 0.095ms (daemon path). Use `run_uncached` from background threads.
- **ANE op-count limit**: ~15–20 ops per kernel compile; >30 ops → `ANECCompile FAILED`.

Full details in `ane/ANE_RULES.md`.

## Supported Models

DFlash provides draft models for (see [Model Zoo](https://huggingface.co/collections/z-lab/dflash)):

| Target Model | DFlash Draft |
|---|---|
| Qwen3-4B | `z-lab/Qwen3-4B-DFlash-b16` |
| Qwen3-8B | `z-lab/Qwen3-8B-DFlash-b16` |
| Qwen3.5-4B | `z-lab/Qwen3.5-4B-DFlash` |
| Qwen3.5-9B | `z-lab/Qwen3.5-9B-DFlash` |
| Qwen3.5-27B | `z-lab/Qwen3.5-27B-DFlash` |
| LLaMA-3.1-8B-Instruct | `z-lab/LLaMA3.1-8B-Instruct-DFlash-UltraChat` |

## Installation

```bash
pip install -e .
```

Requires: `mlx`, `mlx-lm`, `safetensors`, `huggingface-hub`, `transformers`

For the ANE path (experimental):
```bash
cd ane && pip install maturin && maturin develop
```

## Usage

### Server (OpenAI-compatible)

```bash
# Start DFlash server
python -m mirror_sd.server \
  --model ~/.omlx/models/Qwen3.5-27B-4bit \
  --draft z-lab/Qwen3.5-27B-DFlash

# Or with Qwen3-8B
python -m mirror_sd.server \
  --model Qwen/Qwen3-8B \
  --draft z-lab/Qwen3-8B-DFlash-b16

# Benchmark with llama-benchy
python -m mirror_sd.llama_benchy --base-url http://localhost:8989
```

### CLI

```bash
# Generate with speculative decoding
mirror-sd generate \
  --model Qwen/Qwen3-8B \
  --draft z-lab/Qwen3-8B-DFlash-b16 \
  --prompt "How many positive whole-number divisors does 196 have?" \
  --max-tokens 512 \
  --temperature 0.0

# Benchmark speculative vs autoregressive decoding
mirror-sd bench \
  --model Qwen/Qwen3-8B \
  --draft z-lab/Qwen3-8B-DFlash-b16
```

### Python API

```python
import mlx.core as mx
from mlx_lm import load as mlx_load
from mirror_sd.loader import load_dflash_model
from mirror_sd.generate import spec_generate

target_model, tokenizer = mlx_load("Qwen/Qwen3-8B")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")

tokens = tokenizer.encode("The meaning of life is")
input_ids = mx.array(tokens)[None]

output_ids, stats, target_cache, draft_cache, target_hidden = spec_generate(
    target_model=target_model,
    draft_model=draft_model,
    input_ids=input_ids,
    max_new_tokens=128,
    temperature=0.0,
)

print(tokenizer.decode(output_ids[0].tolist()))
print(f"Speed: {stats.tokens_per_sec:.1f} tok/s, "
      f"Avg acceptance: {stats.avg_acceptance_length:.2f}")
```

## Project Structure

```
mirror_sd/          # MLX implementation
├── dflash.py       # DFlash draft model (target-aware attention + block diffusion)
├── target.py       # Target model integration (hidden state capture + Qwen3.5 support)
├── generate.py     # Speculative decoding loop (KOD, adaptive block, combined eval)
├── server.py       # OpenAI-compatible server with prompt caching
├── loader.py       # Weight loading + quantization support
├── bench.py        # CLI benchmark: baseline vs DFlash
├── llama_benchy.py # llama-benchy benchmark runner
├── compare_bench.py# Benchmark comparison + chart generation
├── prompt.py       # Chat template formatting (/no_think for DFlash compatibility)
└── cli.py          # CLI entry point

ane/                # ANE implementation (Rust + PyO3) — W8A16 quantized, ~3.2x over AR baseline
├── ANE_RULES.md    # ANE compiler constraints, dispatch timing, op-count limits

benchmarks/         # Benchmark data and charts
references/         # Reference implementations and papers
```

## References

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [Mirror Speculative Decoding](https://arxiv.org/abs/2510.13161)
<!--- [Token Wagering Model: Kelly-Optimal Drafting for Speculative Decoding](https://arxiv.org/abs/2504.08816) HAVEN'T WRITTEN YET-->
- [DFlash Models on HuggingFace](https://huggingface.co/collections/z-lab/dflash)

## License

MIT
