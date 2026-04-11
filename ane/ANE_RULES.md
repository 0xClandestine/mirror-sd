# ANE Compiler Rules

Empirically discovered constraints for Apple Neural Engine graph compilation.

## Spatial Width
- Minimum spatial width: **64** (`MIN_SPATIAL_WIDTH`)
- All placeholder tensors must have width >= 64
- Align: `((w + 63) / 64) * 64`

## RMSNorm
- `rsqrt`/`sqrt` after `reduce` ops **fails** — use `pow(-0.5)` instead
- Learned weight: multiply `normed * weight` after `x * inv_std`
- **Weight spatial width MUST match input spatial width** — ANE does NOT broadcast mismatched spatial dims
  - E.g., if input is `[1, C, 1, 128]`, weight must be `[1, C, 1, 128]`, NOT `[1, C, 1, 64]`
- rmsnorm alone always compiles
- rmsnorm after conv1x1 works when conv1x1 input is a single tensor (e.g., fc_norm)
- **rmsnorm after conv1x1 on a concatenated input fails** (concat → conv1x1 → rmsnorm chain)
- rmsnorm + matmul combined works fine

### CRITICAL: ANE rmsnorm is GLOBAL across all channels, NOT per-head
- `reduce_mean(x, axis=1)` reduces over **ALL channels** in the tensor
- For input `[1, N_HEADS*HEAD_DIM, 1, w_sq]` (4096 channels), rmsnorm computes variance over all 4096 values per position — **NOT per-head** (128 values per head)
- GPU `nn.RMSNorm(head_dim)` normalizes each head's 128 dims independently
- **Per-head variance differs significantly** (e.g., 9.2 to 16.5 across heads) — global norm gives completely wrong results
- **This was the ROOT CAUSE of the attention output being wrong** (cosine ~0.005 vs GPU)
- **Fix**: reshape to `[1, HEAD_DIM, 1, N_HEADS*w_sq]` where each spatial position = one (head, seq_pos) pair, then rmsnorm reduces over channels (HEAD_DIM=128) per position = per-head rmsnorm
- Python rearrangement required: `_flat_to_4d_norm` (flat → norm format) and `_4d_norm_to_4d_heads` (norm format → 4D heads)
- Weight for 4d norm: `[1, HEAD_DIM, 1, N_HEADS*w_sq]` with interleaved weight values repeated per (head, position)
- **CRITICAL: weight data must be in channels-first order** — iterate (d, h, pos) not (h, pos, d) when building the flat list for `from_f32`

## Conv1x1 vs Matmul
- `conv1x1_dynamic` preferred over `reshape+transpose+matmul+transpose+reshape` (fewer ops)
- Weight format for conv1x1: `[OC, IC, 1, 1]` (batch=OC, channels=IC)
- Weight placeholder shape: `[1, OC, 1, IC]`, then transpose `[0,3,2,1]` → reshape to `[OC, IC, 1, 1]`
- **CRITICAL: `transpose()` on a placeholder produces garbage output** (cosine sim ~0.06). Must use concat+slice+transpose pattern instead:
  - Concat input and weight along width dim: `packed = concat([input, weight], dim=3)`
  - Slice to recover each: `a = slice(packed, [0,0,0,0], [1,IC,1,seq])`, `w = slice(packed, [0,0,0,seq], [1,IC,1,OC])`
  - Transpose the weight slice: `wt = transpose(w, [0,3,2,1])`
  - Reshape: `w_conv = reshape(wt, [OC, IC, 1, 1])`
  - Conv: `out = conv1x1_dynamic(a, w_conv)`
- Weight data must be stored as **W.T** (transposed) in `[1, IC, 1, OC]` format for the concat-slice pattern
- `nn.Linear` falls off ANE — must use Conv2d(1x1) or conv1x1_dynamic

## Graph Complexity Limit
- ANE compiler has an **undocumented op count limit** per single dispatch
- Graphs with ~15-20 ops compile; ~30+ ops fail with generic "ANECCompile() FAILED"
- Keep each kernel to **1-2 main operations** (e.g., rmsnorm+conv, or conv alone, or rmsnorm alone)

## GQA (Grouped Query Attention)
- Matmul with mismatched batch dimensions (32 query heads, 8 kv heads) **fails**
- Must tile KV heads: slice each kv_head, repeat GQA_RATIO times, concat along channel dim
- `tile_kv_heads`: for each of 8 kv_heads, slice `[0, h, 0, 0]` to `[1, 1, seq, hd]`, push GQA_RATIO copies, concat on dim 1

## IOSurface Reshape Is NOT Data Rearrangement (CRITICAL)
- **`reshape` on an IOSurface-backed tensor reinterprets the same bytes with different shape/stride WITHOUT moving data.**
- This means `reshape([1, N_HEADS*HEAD_DIM, 1, w_sq])` → `[1, N_HEADS, w_sq, HEAD_DIM]` produces **WRONG data** because the channels-first IOSurface layout doesn't match the target shape.
- **Correct pattern**: reshape to `[1, N_HEADS, HEAD_DIM, w_sq]` then `transpose([0,1,3,2])` to `[1, N_HEADS, w_sq, HEAD_DIM]`.
  - The reshape `[1, NH*HD, 1, w_sq]` → `[1, NH, HD, w_sq]` is valid because it only splits the channel dimension (no data rearrangement needed).
  - The transpose then correctly swaps the height and width dimensions.
- **Verified**: 3-way test confirmed this pattern gives cosine sim 1.0 vs GPU reference, while direct reshape gives ~0.06.
- **Applies everywhere**: rope_q, rope_k, gqa_tile, attn_residual — any 4D↔flat conversion.

### Reverse (4D → flat):
- Similarly, `[1, NH, w_sq, HD]` → flat `[1, NH*HD, 1, w_sq]` must go through `transpose([0,1,3,2])` first to get `[1, NH, HD, w_sq]`, then reshape to `[1, NH*HD, 1, w_sq]`.
- Direct reshape from `[1, NH, w_sq, HD]` to `[1, NH*HD, 1, w_sq]` produces wrong data.

### Placeholder transpose is broken (confirmed):
- **`transpose()` directly on a placeholder produces garbage** (cosine sim ~0.06).
- Must use concat+slice+transpose pattern (see Conv1x1 section).
- However, `transpose()` on intermediate tensors (outputs of other ops like reshape, matmul, etc.) works correctly.

## RoPE
- Cannot use built-in ops — must implement manually: reshape to pairs, slice even/odd, negate odd, concat, reshape, multiply by cos/sin, add
- RoPE cos/sin tables must be runtime inputs (not constants) since positions change per iteration
- apply_rope alone compiles fine; **two apply_rope calls in one kernel exceeds op limit** — must split
- K RoPE workaround: use single apply_rope on full K sequence with precomputed cos/sin that already map correct position IDs for ctx and noise segments

### Interleaved vs Half-Rotation RoPE
- Qwen3 uses **half-rotation** RoPE: pairs dimension `d` with `d + head_dim/2` (i.e., `[x0,...,x63, x64,...,x127]` → rotate `[x0,x64], [x1,x65], ...`).
- The ANE `apply_rope` uses **interleaved** pairing: consecutive dimensions `[2k, 2k+1]` (i.e., `[x0,x1], [x2,x3], ...`).
- **These are NOT equivalent** — using interleaved RoPE with half-rotation cos/sin gives cosine sim ~0.84.
- **Solution**: interleave the output dimensions of q_proj/k_proj so that interleaved RoPE produces the same result as half-rotation:
  - Reorder weight rows: `[d0..d63, d64..d127]` → `[d0,d64,d1,d65,...,d63,d127]`
  - Reorder q_norm/k_norm weights per head the same way
  - **CRITICAL**: cos/sin data must use **same angle for BOTH elements** of each pair: `[cos(θ_k), cos(θ_k)]` and `[sin(θ_k), sin(θ_k)]`, NOT `[cos(θ_k), 1.0]` and `[sin(θ_k), 0.0]`
- Since both Q and K are interleaved the same way, `Q @ K^T` is invariant to the reordering.
- V stays in original (non-interleaved) format, so attn_out is also original format and o_proj needs no changes.

### Why cos/sin must repeat the same angle for both pair elements
- The ANE `apply_rope` does: `result = x * cos + rotated * sin` where `rotated[2k] = -x[2k+1]`, `rotated[2k+1] = x[2k]`
- With `cos[2k+1]=1.0, sin[2k+1]=0.0`:
  - `result[2k] = x[2k]*cos(θ_k) - x[2k+1]*sin(θ_k)` ✓ (first element rotated)
  - `result[2k+1] = x[2k+1]*1.0 + x[2k]*0.0 = x[2k+1]` ✗ (second element UNCHANGED)
- Half-rotation rotates BOTH elements: `result[k+64] = x[k]*sin(θ_k) + x[k+64]*cos(θ_k)`
- With `cos[2k+1]=cos(θ_k), sin[2k+1]=sin(θ_k)`:
  - `result[2k+1] = x[2k+1]*cos(θ_k) + x[2k]*sin(θ_k)` ✓ (both elements rotated)
- This was the cause of Q after RoPE having cosine ~0.83 vs GPU (now ~0.9999)

## Per-dispatch overhead
- ~0.05ms via direct evaluation (XPC bypass)
- ~0.095ms via daemon path

## `run_cached` IOSurface Caching (CRITICAL)
- **`run_cached` stores IOSurface references from the first call.** Using the same compiled kernel with different TensorData objects (e.g., kv_concat for K vs V) silently uses the cached surfaces.
- **Solution**: compile **separate kernel instances** for K concat and V concat, even though the graph structure is identical.
- `k_concat` and `v_concat` must be separate `ANEKernel` objects.

## Same-buffer input/output (FIXED)
- Using the same IOSurface buffer for both input and output of a kernel (e.g., `rope_q` reading from and writing to `b_q_4d`) works correctly but is fragile.
- **Fix**: `rope_q` now uses separate output buffer `b_q_rope_4d`. `attn_out` reads from `b_q_rope_4d` (not `b_q_4d`).

## FP16 Overflow (KNOWN LIMITATION)
- ANE operates in **fp16** internally (max representable value ~65504)
- GPU uses **bf16** (max ~3.4×10^38)
- With random inputs, intermediate values in deeper layers can overflow fp16, producing Inf
- This is NOT a correctness bug — in real inference, model inputs are well-behaved
- If overflow is a problem, consider: (1) fp16 clipping, (2) mixed-precision approaches, (3) quantization-aware training

## Proven kernel splits (DFlash per layer — CURRENT)
1. `fc_norm`: conv1x1(fc) + rmsnorm ✓ (cos=0.9998)
2. `q_proj`: rmsnorm(hidden) → conv1x1(Q) ✓ (cos=1.0 vs GPU)
3. `q_norm_4d`: per-head rmsnorm on `[1, HEAD_DIM, 1, N_HEADS*w_sq]` ✓ (cos=0.9999)
4. `k_proj_ctx`: conv1x1(context → K) ✓ (cos=1.0)
5. `k_proj_noise`: rmsnorm(hidden) → conv1x1(K) ✓ (cos=1.0)
6. `k_concat`: concat(ctx_K, noise_K) ✓ (cos=1.0)
7. `k_norm_4d`: per-head rmsnorm on `[1, HEAD_DIM, 1, N_KV_HEADS*w_kv]` ✓ (cos=0.9999)
8. `v_proj_ctx`: conv1x1(context → V) ✓
9. `v_proj_noise`: rmsnorm(hidden) → conv1x1(V) ✓
10. `v_concat`: concat(ctx_V, noise_V) ✓ (separate instance from k_concat)
11. `rope_q`: interleaved apply_rope on 4D Q ✓ (cos=0.9999)
12. `rope_k`: interleaved apply_rope on 4D K ✓ (cos=0.9999)
13. `gqa_tile`: tile_kv_heads + concat K/V ✓
14. `attn_out`: SDPA (Q @ K^T / sqrt(d) → softmax → @ V) ✓ (cos=0.999)
15. `o_proj_residual`: conv1x1(o_proj) + residual add ✓
16. `ffn_residual`: rmsnorm → 2×conv1x1 → swiglu → conv1x1 → residual add ✓
17. `final_norm`: rmsnorm ✓

### Layer 0 end-to-end: cosine = 0.999 vs GPU ✓

### Python rearrangements per layer
- After `q_proj`: `_flat_to_4d_norm` → `q_norm_4d` → `_4d_norm_to_4d_heads` → `rope_q`
- After `k_concat`: `_flat_to_4d_norm` → `k_norm_4d` → `_4d_norm_to_4d_heads` → `rope_k`
- After `v_concat`: `_flat_to_4d_heads` → `gqa_tile`
- After `attn_out`: `_flatten_attn_4d` → `o_proj_residual`

### Known issues
- FP16 overflow in deeper layers with large random inputs (not an issue with real model inputs)
- Weight loading takes ~110s (from_f32 NEON conversion is the bottleneck, not Python loops)
- Per-head norm adds 4 Python read+rearrange+write round-trips per layer

## Weight loading
- IOSurface stores fp32, ANE casts to fp16 internally
- `f32_to_f16_bulk` via NEON SIMD for staging
- **Never run forward() with dummy weights** — shape mismatches cause unkillable hangs in kernel IOKit calls
- Always use timeout when testing ANE execution
- Weight data for conv1x1_proj is stored as **W.T** in `[1, IC, 1, OC]` format
- q_proj and k_proj weights must be **interleaved** (`[d0,d64,d1,d65,...]` per head) to match ANE interleaved RoPE
- q_norm and k_norm weights must be **interleaved** and in 4d norm format: `[1, HEAD_DIM, 1, N_HEADS*w_sq]` with values repeated per (head, position) in **channels-first order**
- `_make_4d_norm_weight` builds this format using MLX broadcast for efficiency
- `_make_per_head_norm_weight` is DEPRECATED — was for the old global-norm approach
- Weight loading takes ~20s per layer (~110s total) due to `from_f32` NEON f32→f16 conversion

## Bug history (lessons learned)

### Bug #1: Global rmsnorm vs per-head rmsnorm (FIXED)
- **Symptom**: ANE attention output cosine ~0.005 vs GPU. Individual components (Q, K, V, RoPE) tested OK but combined pipeline failed.
- **Root cause**: ANE `rmsnorm` with `reduce_mean(x, 1)` reduces over ALL channels. For `[1, N_HEADS*HEAD_DIM, 1, w_sq]` (4096 channels), this computes a single variance across all heads, not per-head. GPU `nn.RMSNorm(head_dim)` normalizes each head's 128 dims independently.
- **Why individual tests passed**: with random inputs, per-head and global variance happen to be similar, giving misleadingly high cosine (~0.994). Real model inputs have significantly different per-head variances (9.2 to 16.5), exposing the bug.
- **Fix**: split q_norm and k_norm into separate 4d norm kernels that take `[1, HEAD_DIM, 1, N_HEADS*w_sq]` format, where each spatial position = one (head, seq_pos) pair. rmsnorm then reduces over HEAD_DIM channels per position = correct per-head norm.
- **Lesson**: always test with REAL model inputs, not just random data. Random data can mask bugs by having uniform statistics.

### Bug #2: Interleaved RoPE only rotates half of each pair (FIXED)
- **Symptom**: Q after RoPE cosine ~0.83 vs GPU (should be ~0.999).
- **Root cause**: cos/sin data was `[cos(θ), 1.0]` and `[sin(θ), 0.0]` per pair. The ANE `apply_rope` computes `result[2k+1] = x[2k+1]*cos[2k+1] + rotated[2k+1]*sin[2k+1]`. With cos=1, sin=0, the odd element is unchanged. But half-rotation rotates BOTH elements.
- **Fix**: use `[cos(θ), cos(θ)]` and `[sin(θ), sin(θ)]` so both elements get the same rotation angle.
- **Lesson**: verify EACH element of the output, not just the overall cosine similarity.

### Bug #3: _make_4d_norm_weight data order (FIXED)
- **Symptom**: k_norm_4d cosine 0.944 vs GPU (should be 0.999).
- **Root cause**: weight data was generated in (h, pos, d) order but IOSurface stores data in channels-first (d, h, pos) order. The `from_f32` function reads flat data as channel-0-spatial, channel-1-spatial, etc.
- **Fix**: iterate (d, h, pos) when generating weight data, matching channels-first IOSurface layout.
- **Lesson**: always match the IOSurface channels-first layout when generating weight data for `from_f32`.

### Bug #4: IOSurface reshape is not data rearrangement (FIXED)
- **Symptom**: after converting flat `[1, C, 1, W]` to 4D `[1, NH, w, HD]`, data was garbage.
- **Root cause**: IOSurface reshape reinterprets bytes without moving data. Channels-first layout ≠ the target 4D layout.
- **Fix**: reshape `[1, NH*HD, 1, w]` → `[1, NH, HD, w]` (splits channels, valid), then transpose to `[1, NH, w, HD]`.
- **Lesson**: all flat↔4D conversions must go through explicit transpose, done in Python (read_f32 → rearrange → write_f32).

### Bug #5: Placeholder transpose produces garbage (FIXED)
- **Symptom**: conv1x1 with direct transpose on weight placeholder gave cosine ~0.06.
- **Root cause**: ANE compiler bug — `transpose()` on a placeholder doesn't work correctly.
- **Fix**: concat+slice+transpose pattern (see Conv1x1 section).

### Bug #6: run_cached IOSurface caching (FIXED)
- **Symptom**: V concat output was all zeros when using same kernel instance as K concat.
- **Root cause**: `run_cached` stores IOSurface references from first call. Same kernel + different TensorData = silently uses cached surfaces.
- **Fix**: compile separate kernel instances for K concat and V concat.

### Bug #7: concat + conv1x1 in same kernel fails at runtime (FIXED)
- **Symptom**: compiles but produces wrong results.
- **Root cause**: ANE runtime can't handle concat → conv1x1 in single dispatch.
- **Fix**: split k_proj/v_proj into separate ctx/noise sub-kernels with Python concat.
