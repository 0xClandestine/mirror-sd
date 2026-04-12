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

M4 Pro (64GB), MLX, greedy decoding (`temperature=0.0`), `/no_think` chat template, 8 math/code prompts, 256 tokens each.

### Qwen3-8B

| Metric | Value |
|--------|-------|
| Baseline (autoregressive) | 27.0 tok/s |
| DFlash + KOD | 95.8 tok/s |
| **Speedup** | **3.55x** |
| Avg acceptance length | 8.29 |
| Block size | 16 (KOD-adapted) |

### Qwen3.5-27B-4bit

| Metric | Value |
|--------|-------|
| Baseline (autoregressive) | 25.0 tok/s |
| DFlash + KOD | 34.7 tok/s |
| **Speedup** | **1.39x** |
| Avg acceptance length | 3.89 |
| Block size | 4 (KOD-adapted) |

The 27B model uses a smaller block size because verify cost scales with block size (~20ms/tok for 4 tokens vs ~37ms for single-token decode). With block_size=16, verify is too expensive for the acceptance rate, resulting in a net slowdown. Adaptive block sizing automatically shrinks the block when recent acceptance drops, avoiding wasted verify compute.

### Block size sweep (Qwen3.5-27B-4bit, 128 tokens)

| Block Size | Avg Accept | tok/s | Speedup |
|---|---|---|---|
| 2 | 1.89 | 22.2 | 0.90x |
| **4** | **3.33** | **29.0** | **1.17x** |
| 8 | 5.33 | 22.8 | 0.92x |
| 16 | 5.28 | 22.2 | 0.94x |

## MLX Implementation

The primary implementation runs both target and draft models on GPU via MLX.

### Key implementation details

- Draft model loads in **bf16** to match the target model's dtype — this was the single biggest acceptance rate improvement (+50%)
- DFlash generates blocks of tokens per forward pass via non-causal attention (block_size=16 for 8B, block_size=4 for 27B)
- 5 target hidden features extracted from layers uniformly distributed through the target model, injected into K/V of every draft layer
- Draft model shares embedding and `lm_head` with target (only transformer layers trained)
- **Combined eval** — draft, verify, and cache state updates are fused into a single `mx.eval()` call, eliminating redundant GPU sync points
- **Adaptive block sizing** — block size shrinks when recent acceptance is low (avg <1.0 → block 2, avg <2.0 → block 3), avoiding wasted verify compute on rejected drafts
- Repetition detection prevents degenerate accept/reject loops

## ANE Execution Path

The `ane/` directory contains a Rust implementation of the DFlash draft model for Apple Neural Engine, using the [ane](https://github.com/ncdrone/ane) crate for direct ANE graph compilation.

The goal: run the target model on GPU and the draft model on ANE in parallel, matching the Mirror-SD paper's heterogeneous accelerator design. On Apple Silicon with unified memory, the draft model's inputs (target hidden states) and outputs (draft logits) can be exchanged with zero-copy.

### Architecture

The DFlash forward pass is decomposed into 7 ANE kernels per layer:

1. **fc_norm** — conv1x1 (feature projection) + scaled rmsnorm
2. **mega_qkv** — rmsnorm(input) → Q/K/V projections (input-pack) → per-head norms → RoPE
3. **gqa_tile** — tile KV heads for grouped query attention
4. **attn_out** — scaled dot-product attention with attention score softcapping
5. **o_proj_residual** — conv1x1 (output projection) + residual add + softcapping
6. **ffn_residual** — rmsnorm → SwiGLU MLP → residual add + softcapping
7. **final_norm** — rmsnorm

This achieves **cosine similarity 0.91** vs GPU reference with no inf/nan values.

### The bf16 → f16 Precision Wall

The ANE path is currently **not viable** for speculative decoding. The fundamental problem:

- DFlash models are trained and distributed in **bf16** (their native quantization)
- The ANE operates internally in **f16**
- Converting bf16 weights → f16 introduces precision loss that tanks the draft model's acceptance rate
- Unlike the MLX path where both target and draft run in bf16, the ANE cannot preserve the precision the model was trained with

The workarounds implemented (scaled rmsnorm, residual softcapping, attention score softcapping) successfully prevent fp16 overflow and produce numerically stable output — but the inherent precision loss from the dtype conversion is enough to degrade acceptance rate below what's needed for speculative decoding to provide a speedup.

This is a hardware limitation: until Apple Silicon supports bf16 computation on the ANE, or DFlash models are trained in f16, this path cannot succeed.

### ANE implementation details

Several techniques were developed to make the DFlash model numerically stable in fp16:

- **Scaled rmsnorm** — multiply input by 1/128 before variance computation; rmsnorm is scale-invariant so the output is approximately unchanged, but intermediate values stay within fp16 range
- **Residual stream softcapping** — `cap * tanh(residual / cap)` after each residual addition (cap=30000) prevents fp16 overflow while preserving normal-range values
- **Attention score softcapping** — same approach before softmax prevents fp16 overflow from large attention scores
- **Input-pack QKV projections** — pack context + normed hidden into one tensor for a single conv1x1 per projection, avoiding the ANE's broken concat→reshape→transpose pattern
- **Interleaved RoPE** — reorder Q/K projection weights to match ANE's interleaved pairing, with cos/sin repeated per pair element for correct half-rotation

Full details on ANE constraints and bug history are documented in `ane/ANE_RULES.md`.

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

### CLI

```bash
# Generate with speculative decoding
mirror-sd generate \
  --model Qwen/Qwen3-8B \
  --draft z-lab/Qwen3-8B-DFlash-b16 \
  --prompt "How many positive whole-number divisors does 196 have?" \
  --max-tokens 512 \
  --temperature 0.0

# Convert DFlash weights to MLX format (optional, speeds up loading)
mirror-sd convert \
  --source z-lab/Qwen3-8B-DFlash-b16 \
  --output ./dflash-mlx

# Benchmark speculative vs autoregressive decoding
mirror-sd bench \
  --model Qwen/Qwen3-8B \
  --draft z-lab/Qwen3-8B-DFlash-b16
```

### Python API

```python
import mlx.core as mx
from mlx_lm import load as mlx_load
from mirror_sd import DFlashDraftModel, DFlashConfig, spec_generate
from mirror_sd.loader import load_dflash_model

target_model, tokenizer = mlx_load("Qwen/Qwen3-8B")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")

tokens = tokenizer.encode("The meaning of life is")
input_ids = mx.array(tokens)[None]

output_ids, stats = spec_generate(
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
├── generate.py     # Speculative decoding loop (combined eval + adaptive block)
├── loader.py       # Weight loading + quantization support
├── bench.py        # Benchmark: baseline vs DFlash
├── prompt.py       # Chat template formatting (/no_think for DFlash compatibility)
└── cli.py          # CLI entry point

ane/                # ANE implementation (Rust + PyO3)
├── src/
│   ├── dflash.rs   # ANE kernel graph builders (7-kernel pipeline)
│   ├── wrapper.rs  # Python bindings (ANETensor, ANEKernel)
│   └── lib.rs      # PyO3 module
├── ANE_RULES.md    # ANE compiler constraints and bug history
├── Cargo.toml
└── pyproject.toml

references/         # Reference implementations and papers
```

## References

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [Mirror Speculative Decoding](https://arxiv.org/abs/2510.13161)
- [DFlash Models on HuggingFace](https://huggingface.co/collections/z-lab/dflash)

## License

MIT
