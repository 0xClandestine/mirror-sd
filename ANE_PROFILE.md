# ANE DFlash Profile Report

**Model:** Qwen3.5-27B-4bit
**Draft:** z-lab/Qwen3.5-27B-DFlash (5 layers, block_size=16)
**Script:** `bench_ane_profile.py` — 3 timed runs × 256 tokens, 1 warmup
**Prompt:** 53 tokens (FastAPI todo app)

---

## Throughput

| Mode | tok/s | Prefill | Gen time | vs AR |
|------|------:|--------:|---------:|------:|
| AR baseline (pure GPU) | 26.9 | 353 ms | 9,464 ms | 1.00× |
| GPU-only DFlash spec | 23.6 | 378 ms | 10,929 ms | 0.88× |
| ANE spec · ctx=64 | **29.5** | 302 ms | 8,734 ms | **1.10×** |
| ANE spec · ctx=128 | 29.2 | 292 ms | 8,827 ms | 1.08× |
| ANE spec · ctx=256 | 28.6 | 294 ms | 9,023 ms | 1.06× |
| ANE spec · ctx=512 | 27.6 | 292 ms | 9,333 ms | 1.03× |

GPU-only spec is slower than AR. The S=16 batch verify on 27B costs more than 16 sequential S=1 steps, and α=5.47 doesn't offset that. The ANE path recovers the loss by running draft in a background thread, overlapping it with the verify pass.

---

## Acceptance Statistics

| Mode | avg α | min | max | std | steps/run |
|------|------:|----:|----:|----:|----------:|
| GPU-only DFlash | 5.47 | 1 | 15 | 3.66 | 47 |
| ANE spec (all ctx) | 10.52 | 1 | 16 | 6.87 | 23 |

Acceptance length histogram — ANE ctx=512, 69 total steps:

```
α= 1  18 (26.1%)  ██████████
α= 4   6  (8.7%)  ███
α= 5   3  (4.3%)  █
α=15   3  (4.3%)  █
α=16  39 (56.5%)  ██████████████████████
```

> **⚠ Correctness caveat:** `_spec_generate_parallel` only trims the FA KV cache on rollback — it does not restore the GDA SSM/conv states. On Qwen3.5 (a hybrid GDA+FA model) this means the target model's state diverges from AR after every rejected block. The inflated α=10.52 likely reflects this drift rather than true draft quality.

---

## Per-Step Timing

| Mode | Draft/step | Verify/step | Overlap | Hidden | Bottleneck |
|------|----------:|------------:|--------:|-------:|-----------:|
| GPU-only DFlash | 0.2 ms | 231.6 ms | 0% | — | verify |
| ANE ctx=64 | 215.3 ms | 163.8 ms | 163.9 ms | **76.1%** | draft |
| ANE ctx=128 | 219.2 ms | 164.0 ms | 163.6 ms | 74.8% | draft |
| ANE ctx=256 | 228.1 ms | 163.6 ms | 163.4 ms | 71.7% | draft |
| ANE ctx=512 | 241.5 ms | 163.3 ms | 163.3 ms | 67.7% | draft |

Verify time is flat at ~163 ms across all ctx depths — it depends on the growing KV cache, not the ANE context window. The draft is always the bottleneck; at ctx=64 there is still a 52 ms gap where the CPU waits after verify finishes.

---

## ANE Forward Pass Phase Breakdown

Times in ms per draft call (median over all profiled steps).

| Phase | ctx=64 | ctx=128 | ctx=256 | ctx=512 |
|-------|-------:|--------:|--------:|--------:|
| write noise embedding | 0.55 | 0.55 | 0.54 | 0.55 |
| context FC + write | 2.09 | 2.46 | 3.10 | 4.50 |
| RoPE precompute | <0.01 | <0.01 | <0.01 | <0.01 |
| attn mask precompute | <0.01 | <0.01 | <0.01 | <0.01 |
| **all layers (total)** | **155.97** | **161.20** | **166.43** | **176.79** |
| final norm | 0.19 | 0.19 | 0.19 | 0.20 |
| **read output buffer** | **15.58** | **15.62** | **15.66** | **15.71** |
| **Total ANE forward** | **174.7** | **180.5** | **186.5** | **198.8** |

Context FC + write scales linearly with ctx_len (GPU-side matmul `target_hidden @ fc.T`). RoPE and mask precompute are cached and free on repeat calls with the same offset.

The output buffer read is a fixed 15.6 ms regardless of context — it transfers 16 × 4096 × 4 = 256 KB of float32 from ANE memory back to MLX. This is ~9% of total forward time at ctx=64.

---

## Per-Kernel Timing (5 layers summed per call)

| Kernel | ctx=64 | ctx=128 | ctx=256 | ctx=512 |
|--------|-------:|--------:|--------:|--------:|
| `mega_qkv` — Q/K/V projections + per-head norms + RoPE | 10.7 ms | 15.8 ms | 20.0 ms | 28.4 ms |
| `gqa_tile` — GQA KV expand | 0.9 ms | 1.0 ms | 1.2 ms | 1.7 ms |
| `attn_out` — scaled dot-product attention | 1.2 ms | 1.4 ms | 1.9 ms | 3.4 ms |
| `o_proj_residual` — output projection + residual add | 6.2 ms | 6.2 ms | 6.3 ms | 6.4 ms |
| **`ffn_residual` — gate/up/down MLP + residual** | **137.0 ms** | **136.8 ms** | **137.1 ms** | **136.9 ms** |

As % of layer time:

| Kernel | ctx=64 | ctx=128 | ctx=256 | ctx=512 |
|--------|-------:|--------:|--------:|--------:|
| `ffn_residual` | **87.8%** | **84.9%** | **82.4%** | **77.4%** |
| `mega_qkv` | 6.9% | 9.8% | 12.0% | 16.1% |
| `o_proj_residual` | 4.0% | 3.8% | 3.8% | 3.6% |
| `attn_out` | 0.8% | 0.9% | 1.1% | 1.9% |
| `gqa_tile` | 0.6% | 0.6% | 0.7% | 0.9% |

`ffn_residual` is essentially constant across ctx depths — it is compute-bound on the MLP matmuls (hidden=5120, intermediate=12288, 5 layers). `mega_qkv` and `attn_out` scale with ctx_len as attention coverage grows.

---

## Pipeline Efficiency

| ctx | Draft | Verify | Hidden | Pipeline gain |
|-----|------:|-------:|-------:|--------------:|
| 64 | 215.4 ms | 163.9 ms | **76.1%** | 2.31× |
| 128 | 220.0 ms | 163.6 ms | 74.4% | 2.34× |
| 256 | 228.1 ms | 163.4 ms | 71.6% | 2.40× |
| 512 | 241.7 ms | 163.3 ms | 67.6% | 2.48× |

Pipeline gain = `(draft + verify) / verify` — the maximum speedup if draft were free. We capture 68–76% of it. To close the gap: either reduce ANE forward time (faster MLP kernel) or increase verify time (larger model / longer KV cache).

---

## Memory: ANE Activation Buffers

| ctx | Activation buffers |
|-----|------------------:|
| 64 | 19.7 MB |
| 128 | 29.8 MB |
| 256 | 50.0 MB |
| 512 | 90.3 MB |

Dominated by the KV buffers `[1, heads, w_kv, head_dim]`. All fit comfortably within ANE SRAM.

---

## Startup Cost

~20–23 s per ctx_len variant to compile Metal kernels and load 5 layers of weights into ANE buffers. This is one-time per process.

---

## Summary of Bottlenecks

| Priority | Issue | Where |
|----------|-------|--------|
| 1 | `ffn_residual` is 78–88% of ANE time (137 ms flat) | `ANEDraftModel._run_layer` |
| 2 | Output buffer read is 15.6 ms (~9%) regardless of ctx | `_read_mlx_2d` / ANETensor |
| 3 | Draft exceeds verify by 52 ms at ctx=64, growing to 78 ms at ctx=512 | Threading gap |
| 4 | GDA rollback missing from ANE path — state diverges from AR | `_spec_generate_parallel` |
| 5 | GPU-only spec is 0.88× AR — pipelining is load-bearing for any speedup | Architecture |
