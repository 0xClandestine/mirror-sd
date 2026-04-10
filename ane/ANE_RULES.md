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
- `nn.Linear` falls off ANE — must use Conv2d(1x1) or conv1x1_dynamic

## Graph Complexity Limit
- ANE compiler has an **undocumented op count limit** per single dispatch
- Graphs with ~15-20 ops compile; ~30+ ops fail with generic "ANECCompile() FAILED"
- Keep each kernel to **1-2 main operations** (e.g., rmsnorm+conv, or conv alone, or rmsnorm alone)

## GQA (Grouped Query Attention)
- Matmul with mismatched batch dimensions (32 query heads, 8 kv heads) **fails**
- Must tile KV heads: slice each kv_head, repeat GQA_RATIO times, concat along channel dim
- `tile_kv_heads`: for each of 8 kv_heads, slice `[0, h, 0, 0]` to `[1, 1, seq, hd]`, push GQA_RATIO copies, concat on dim 1

## RoPE
- Cannot use built-in ops — must implement manually: reshape to pairs, slice even/odd, negate odd, concat, reshape, multiply by cos/sin, add
- RoPE cos/sin tables must be runtime inputs (not constants) since positions change per iteration
- apply_rope alone compiles fine; **two apply_rope calls in one kernel exceeds op limit** — must split
- K RoPE workaround: use single apply_rope on full K sequence with precomputed cos/sin that already map correct position IDs for ctx and noise segments

## Per-dispatch overhead
- ~0.05ms via direct evaluation (XPC bypass)
- ~0.095ms via daemon path
- With 9 kernels × 5 layers + 2 = 47 dispatches → ~2.35ms overhead (acceptable)

## Proven kernel splits (DFlash per layer)
1. `fc_norm`: conv1x1(fc) + rmsnorm ✓
2. `q_kernel`: rmsnorm(hidden) → conv1x1(Q) → rmsnorm(q_norm) ✓
3. `k_proj`: concat[target, normed] → conv1x1(K) ✓
4. `k_norm`: rmsnorm on K output ✓
5. `v_proj`: concat[target, normed] → conv1x1(V) ✓
6. `rope_q`: reshape + apply_rope on Q ✓
7. `rope_k`: reshape + single apply_rope on full K (precomputed cos/sin for mixed positions) ✓
8. `gqa_tile`: reshape K/V + tile_kv_heads + concat ✓
9. `attn_residual`: slice Q/K/V → SDPA → conv1x1(o_proj) → residual add ✓
10. `ffn_residual`: rmsnorm → 2×conv1x1 → swiglu → conv1x1 → residual add ✓
11. `final_norm`: rmsnorm ✓

## Weight loading
- IOSurface stores fp32, ANE casts to fp16 internally
- `f32_to_f16_bulk` via NEON SIMD for staging
- **Never run forward() with dummy weights** — shape mismatches cause unkillable hangs in kernel IOKit calls
- Always use timeout when testing ANE execution
