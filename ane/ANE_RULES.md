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
  - Generate interleaved cos/sin: for each pair `k`, output `cos(θ_k), 1.0` and `sin(θ_k), 0.0`
- Since both Q and K are interleaved the same way, `Q @ K^T` is invariant to the reordering.
- V stays in original (non-interleaved) format, so attn_out is also original format and o_proj needs no changes.

## Per-dispatch overhead
- ~0.05ms via direct evaluation (XPC bypass)
- ~0.095ms via daemon path
- With 9 kernels × 5 layers + 2 = 47 dispatches → ~2.35ms overhead (acceptable)

## `run_cached` IOSurface Caching (CRITICAL)
- **`run_cached` stores IOSurface references from the first call.** Using the same compiled kernel with different TensorData objects (e.g., kv_concat for K vs V) silently uses the cached surfaces.
- **Solution**: compile **separate kernel instances** for K concat and V concat, even though the graph structure is identical.
- `k_concat` and `v_concat` must be separate `ANEKernel` objects.

## Proven kernel splits (DFlash per layer)
1. `fc_norm`: conv1x1(fc) + rmsnorm ✓
2. `q_kernel`: rmsnorm(hidden) → conv1x1(Q) → rmsnorm(q_norm) ✓
3. `k_proj_ctx`: conv1x1(context → K) ✓
4. `k_proj_noise`: rmsnorm(hidden) → conv1x1(K) ✓
5. `k_concat`: concat(ctx_K, noise_K) ✓
6. `k_norm`: rmsnorm on K output ✓
7. `v_proj_ctx`: conv1x1(context → V) ✓
8. `v_proj_noise`: rmsnorm(hidden) → conv1x1(V) ✓
9. `v_concat`: concat(ctx_V, noise_V) ✓ (separate instance from k_concat)
10. `rope_q`: reshape+transpose → interleaved apply_rope ✓
11. `rope_k`: reshape+transpose → interleaved apply_rope → transpose+reshape ✓
12. `gqa_tile`: reshape+transpose K/V → tile_kv_heads → concat ✓
13. `attn_residual`: slice Q/K/V from tiled KV → SDPA → transpose+reshape → conv1x1(o_proj) → residual add
14. `ffn_residual`: rmsnorm → 2×conv1x1 → swiglu → conv1x1 → residual add ✓
15. `final_norm`: rmsnorm ✓

### Known issues
- `attn_residual`: the 4D→flat reshape of attention output may still have data layout issues. The transpose+reshape pattern has been applied but full correctness is not yet verified end-to-end.
- `rope_k` reverse reshape (4D→flat after RoPE): uses transpose+reshape pattern but not yet independently verified.

## Weight loading
- IOSurface stores fp32, ANE casts to fp16 internally
- `f32_to_f16_bulk` via NEON SIMD for staging
- **Never run forward() with dummy weights** — shape mismatches cause unkillable hangs in kernel IOKit calls
- Always use timeout when testing ANE execution
- Weight data for conv1x1_proj is stored as **W.T** in `[1, IC, 1, OC]` format
- q_proj and k_proj weights must be **interleaved** (`[d0,d64,d1,d65,...]` per head) to match ANE interleaved RoPE
- q_norm and k_norm weights must be **interleaved per head** the same way
- `_make_per_head_norm_weight` has `interleave=True` option for this purpose