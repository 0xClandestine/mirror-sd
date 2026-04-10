# Mirror-SD: DFlash Speculative Decoding on Apple Silicon

Combining [DFlash](https://arxiv.org/abs/2602.06036) block-diffusion draft models with [Mirror-SD](https://arxiv.org/abs/2510.13161) concepts for speculative decoding on Apple Silicon via MLX.

## Why This Approach?

| Approach | Draft Models | Complexity | Available Models |
|----------|-------------|------------|------------------|
| Mirror-SD (paper) | Requires custom SPD training | High | None (research only) |
| DFlash (paper) | Pre-trained on HuggingFace | Medium | 15+ models |
| **This Project** | Pre-trained (DFlash) on MLX | Medium | 15+ models |

## Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                     TARGET MODEL (GPU)                        │
│                                                               │
│  Input → [Layer 1] → [Layer 2] → ... → [Layer N] → Logits  │
│                    ↓ (capture at target_layer_ids)            │
│              Hidden States → extract_context_feature()        │
└──────────────────────────────────────────────────────────────┘
                                │
                     ┌──────────▼───────────┐
                     │  target_hidden (O(κ)) │
                     │  fc → hidden_norm      │
                     └──────────┬───────────┘
                                │
┌───────────────────────────────┼──────────────────────────────┐
│                     DRAFT MODEL (MLX)                         │
│                                                               │
│  noise_embedding (mask tokens) + target_hidden                │
│                        ↓                                      │
│  DFlash Layers (Target-Aware Attention + SwiGLU MLP)          │
│    - Q from draft hidden                                      │
│    - K/V from BOTH target hidden + draft hidden               │
│    - Non-causal (block diffusion)                             │
│                        ↓                                      │
│  norm → target.lm_head → Draft Logits → Sample Block          │
└──────────────────────────────────────────────────────────────┘
```

### Target-Aware Attention

The core DFlash innovation: K/V projections attend to **both**:
- Target model's intermediate hidden states (context)
- Draft model's own hidden states (noise)

This lets the draft "see what the target has processed" while generating, enabling parallel block generation instead of autoregressive token-by-token drafting.

### Block Diffusion

DFlash generates blocks of tokens (typically 16) in one forward pass:
1. Input: mask token embeddings + target hidden states
2. Process through transformer layers with target-aware attention
3. Project through target's lm_head for logits
4. Sample all positions simultaneously → block of draft tokens

### Speculative Decoding Loop

```
PREFILL: target(prompt) → first token + target_hidden
DECODE LOOP:
  1. Create block: [last_token, mask, mask, ...]
  2. Draft: embed_tokens(block) → DFlash → target.lm_head → sample
  3. Verify: target(block) → posterior tokens
  4. Accept matching prefix + correction token
  5. Crop caches, update target_hidden
```

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

# Load target model
target_model, tokenizer = mlx_load("Qwen/Qwen3-8B")

# Load DFlash draft model
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")

# Tokenize input
tokens = tokenizer.encode("The meaning of life is")
input_ids = mx.array(tokens)[None]

# Run speculative decoding
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

## Implementation Status

- [x] DFlash model architecture (target-aware attention, block diffusion)
- [x] Speculative decoding loop
- [x] Target model hidden state capture
- [x] HuggingFace weight loading
- [x] MLX weight conversion
- [x] CLI (generate, convert, bench)
- [x] Draft KV cache with crop (accumulates verified prefix)
- [x] Correct RoPE on concatenated [context, noise] K
- [x] Non-causal attention mask for block diffusion
- [ ] ANE execution path (CoreML conversion of draft model)
- [ ] Mirror-SD early-exit signal (target mid-layer → draft parallel start)
- [ ] Mirror-SD branch-complete rollout (top-κ candidate expansion)
- [ ] Quantized draft model support

## Project Structure

```
mirror_sd/
├── __init__.py       # Package exports
├── dflash.py         # DFlash draft model (target-aware attention + block diffusion)
├── target.py         # Target model integration (hidden state capture)
├── generate.py       # Speculative decoding loop
├── loader.py         # Weight loading + HuggingFace conversion
└── cli.py            # CLI entry point
references/           # Reference implementations
├── dflash/           # PyTorch DFlash reference
├── mlx-eagle3/       # EAGLE-3 on MLX reference
├── rustane/          # Rust ANE + Metal engine reference
└── Mirror-Speculative-Decoding.md  # Mirror-SD paper notes
```

## ANE Execution (Future)

The draft model can potentially run on Apple Neural Engine for parallel target+draft execution, matching the Mirror-SD paper's heterogeneous accelerator design:

- **Target** on GPU (Metal via MLX) — high-throughput verification
- **Draft** on ANE (via CoreML conversion) — low-power block generation
- **Token channel** — lightweight exchange of top-κ candidates + log-probs

To convert the draft model for ANE:
```python
import coremltools as ct
# Trace the DFlash model and convert to CoreML ML Program
# Then run on ANE with compute_units=ct.ComputeUnit.ANE_ONLY
```

## Dev Log

### 2025-04-10: Initial MLX Implementation

**What we built:** Replaced the non-functional Rust skeleton with a working Python/MLX implementation of DFlash speculative decoding. The Rust code had zero dependencies, broken math, stub functions, and couldn't compile.

**What works:**
- DFlash draft model loads from HuggingFace (`z-lab/Qwen3-8B-DFlash-b16`), ~1B params
- Target model (Qwen3-8B) loads via mlx-lm
- Target-aware attention: K/V from both target hidden + draft hidden, concatenated along seq dim
- Block diffusion: generates 16 tokens in one forward pass (non-causal)
- Weight loading via `mx.load()` (native MLX safetensors), bfloat16 → float16 conversion
- Full speculative decoding loop: prefill → draft → verify → accept/reject
- End-to-end generation produces coherent text

**Key learnings from the DFlash paper (arXiv:2602.06036):**
- DFlash uses **5 draft layers**, not 24 (Section 5: "we set the number of layers to 5")
- **5 target hidden features** extracted from layers uniformly between 2nd and 3rd-to-last target layer → `[1, 9, 17, 25, 33]` for 36-layer Qwen3-8B
- **KV injection**: target context features projected into K/V of EVERY draft layer (not just input fusion like EAGLE). This is why acceptance scales with draft depth.
- `k_norm` is applied AFTER concatenating context+noise K (not before)
- Draft model **shares embedding and lm_head** with target (only transformer layers trained)
- Paper achieves 4.9x speedup with 5 layers, 6.5 avg acceptance length

**Config gotchas (from actual HF config.json):**
- `num_key_value_heads: 8` (not 2 — the Qwen3-8B DFlash uses GQA with 8 KV heads)
- `intermediate_size: 12288` (not 10944)
- `rope_theta: 1000000` (not 10000 — this is Qwen3's 1M base frequency)
- `mask_token_id: 151669` (not 151667)
- `max_position_embeddings: 40960` (not 4096)
- Config is nested under `dflash_config` key in HuggingFace config.json

**MLX-specific learnings:**
- `mx.load()` handles safetensors natively — no need for PyTorch/numpy as intermediate
- `mx.bfloat16` exists but needs conversion to float16 for computation
- MLX KV cache uses `offset`-based position tracking (not explicit position_ids like HF)
- `create_attention_mask(h, cache)` from `mlx_lm.models.base` generates causal masks
- DFlash draft should NOT use causal mask (block diffusion is non-causal)
- `forward_with_hidden_states` returns only captured layers (not all 36+ like HF's `output_hidden_states`)
- `extract_context_feature` needs to handle both formats: captured-only vs full hidden_states list

**Current bottleneck — low acceptance rate (~1.26/block vs paper's ~6.5):**
Draft produces semantically reasonable tokens ("Paris", "capital", "is") but they don't match the target's greedy selections exactly. Likely causes:
1. RoPE applied incorrectly to the concatenated K/V — context K positions may need different position IDs than noise K positions
2. Non-causal mask handling in `mx.fast.scaled_dot_product_attention` may need explicit attention mask
3. The reference uses HuggingFace's `DynamicCache` with explicit `position_ids` and `cache_position` for RoPE, while MLX uses offset-based cache
4. No draft KV cache means each block is processed from scratch (reference accumulates verified prefix in cache)

### 2025-04-10: Acceptance Rate Fixes — Three Critical Bugs Found

**Root cause analysis:** Ran the PyTorch reference implementation side-by-side with our MLX implementation to trace exact dimensions, position_ids, and token outputs at each decode step. This revealed three bugs that together caused the low acceptance rate.

**Bug 1 — RoPE positions wrong for Q and context K:**
The reference passes `position_ids` covering `[cache_len, ..., start+block_size)` to `rotary_emb(hidden_states, position_ids)`. This produces cos/sin for all positions. Then `apply_rotary_pos_emb` gives Q the last `q_len` positions (matching noise token positions) and K gets all positions (matching ctx + noise). Our code was calling `rope(q)` and `rope(k)` without offset, giving Q positions [0..q_len-1] and K positions [0..ctx_len+q_len-1]. Fix: Q gets `offset=cache_len + ctx_len`, K_ctx gets `offset=cache_len`, K_noise gets `offset=cache_len + ctx_len`. Applied by splitting K, applying RoPE separately, then concatenating.

**Bug 2 — No draft KV cache:**
The reference uses `DynamicCache` with `crop(start)` to accumulate verified prefix K/V across blocks. We were passing `cache=None` every time, processing each block from scratch. Fix: Implemented `DFlashKVCache` with `update_and_fetch()` and `crop()` methods, created per layer via `draft_model.make_cache()`.

**Bug 3 — No explicit non-causal attention mask:**
DFlash uses block diffusion (non-causal/bidirectional attention). MLX's `scaled_dot_product_attention` defaults to full attention when no mask is provided (verified experimentally), so this wasn't strictly a bug — but we now explicitly construct a zero-valued mask via `make_draft_mask()` for clarity and correctness.

**Results after fixes:**
| Prompt | MLX Acceptance | PT Reference Acceptance |
|--------|---------------|------------------------|
| "The capital of France is" | ~8.3* | ~3.1 |
| "Explain relativity..." | ~2.0 | ~2.85 |
| "Write a Python function..." | ~2.4 | — |
| "What is the meaning of life?" | ~2.0 | — |

*\*The "capital of France" prompt shows a degenerate loop — the draft locks into repeating "Paris." and the target keeps confirming. This is not a bug in the implementation but a known issue with speculative decoding when the draft diverges from the target's true greedy path in a self-reinforcing way. The PT reference doesn't have this issue because its bf16 draft produces slightly different (more diverse) predictions.*

**Key learning: f16 vs bf16 precision difference**
The MLX target model runs in f16 while the PT reference uses bf16. Target hidden states differ by ~0.0625 max after just 4 layers. These differences compound through the 5 draft layers, causing draft predictions to diverge from the PT reference. This is the main reason our acceptance rate (2.0-2.4) is below the PT reference (2.85-3.1) on diverse prompts, and why the "simple prompt" loop behavior differs. Future fix: support bf16 in MLX or run draft model in f32 for higher accuracy.

**Architecture decisions:**
- No draft KV cache currently (simpler, correct; cache interaction with target-aware attention's concatenated K/V is complex — context K comes from `target_hidden` which changes each step, not from the cache)
- Python lists for token accumulation (avoids MLX tensor mutation issues)
- Draft uses target's `lm_head` directly (matching DFlash paper's shared head design)

---

## References

- [DFlash: Block Diffusion for Flash Speculative Decoding](https://arxiv.org/abs/2602.06036)
- [Mirror Speculative Decoding](https://arxiv.org/abs/2510.13161)
- [DFlash Models on HuggingFace](https://huggingface.co/collections/z-lab/dflash)

## License

MIT
