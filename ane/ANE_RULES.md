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

## Input-Pack Approach (Rule #22)
- Instead of separate k_proj_ctx + k_proj_noise + output concat (which fails at runtime due to concat→reshape→transpose pattern), pack context + normed into one tensor [1, HIDDEN, 1, w_kv] and do ONE conv1x1 for K (and V)
- Mathematically equivalent: `W @ [ctx, normed] = [W @ ctx, W @ normed]`
- Avoids the problematic output concat pattern entirely
- `build_kqv_plus_vnorm_qnorm_kernel` (mega_qkv) uses this for both K and V projections

## Per-Dispatch Operation Limit (Rule #23)
- ANE has a per-dispatch operation budget
- QKV paths + GQA tile + SDPA exceeds the budget — compiles but fails at runtime ("Program Inference error")
- Must split attention into 2+ kernels: mega_qkv (QKV paths, 3 outputs) → gqa_tile → attn_out → o_proj_residual
- The standalone `fused_attn_out` kernel (GQA tile + SDPA + o_proj) also fails at runtime

## Residual Stream Softcapping (Rule #24)
- Apply `cap * tanh(output / cap)` after each residual addition inside ANE kernels
- cap=30000: bounds residual stream to [-30000, 30000], preventing fp16 overflow
- Values within normal range (~±20000) are only slightly compressed: `30000 * tanh(20000/30000) ≈ 17490` (12.5% reduction)
- Softcapping is applied inside `o_proj_residual` and `ffn_residual` kernels (no Python round-trip)
- Also applied to attention scores in `attn_out` kernel before softmax

## Scaled RMSNorm (Rule #25)
- Multiply input by 1/128 before computing variance in rmsnorm
- rmsnorm is scale-invariant: `rmsnorm(x/S) ≈ rmsnorm(x)` when mean(x²) >> S²×eps
- For typical hidden states (mean(x²) ~10⁶), error is < 0.001
- Prevents `diff*diff` overflow in fp16: `((x-mean)/128)² = (x-mean)²/16384`
- Implemented in the `rmsnorm()` function in dflash.rs — all rmsnorm calls automatically use scaling

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

## Weight Scaling for RMSNorm Accuracy (Rule #26)
- The ANE uses scaled RMSNorm (input / 128 before variance) to prevent diff² overflow
- When input values are small (e.g., embedding output ~0.003), dividing by 128 makes them ~2.3e-5
- Variance of these ~5e-10, which is much smaller than eps=1e-6
- eps dominates → RMSNorm gives completely wrong normalization (cosine 0.805 at layer 0)
- **Weight scaling approach (from Gemma3 FP16/ANE guide):**
  - Scale embedding input by α at write time
  - Scale `o_proj.weight *= α` and `down_proj.weight *= α` (all layers)
  - Scale `final_norm.weight /= α` to cancel at output
  - On GPU (bf16/fp16 without softcapping): α=8 gives cosine 0.9999, logits match ✓
- **NOT VIABLE on our ANE pipeline**: weight scaling amplifies residual stream into
  softcap's compression range. α=8 → residual peak ~31k → softcap(cap=30000) compresses
  ~3%, but cumulative across 5 layers → cosine DROPS from 0.76 to 0.57
- **Root cause**: softcapping is the dominant error source, not RMSNorm eps.
  Weight scaling fixes RMSNorm but worsens softcap distortion.
- **Real fix**: raise or remove softcap (requires recompiling ANE kernels with higher cap
  or no cap). Our profiling shows peak residual is ~7904 — well below FP16 max (65504).
  Cap=30000 is overly conservative; cap=60000 or removing softcap would eliminate
  most distortion without overflow risk for this model.
- **Alternative**: fix RMSNorm eps issue directly by using larger eps (e.g., 1e-3 instead
  of 1e-6) in the ANE scaled RMSNorm. This changes the kernel constant but doesn't
  affect the residual stream scale. Requires recompiling Rust ANE kernels.

## FP16 Overflow (SOLVED via scaled rmsnorm + residual softcapping)
- ANE operates in **fp16** internally (max representable value ~65504)
- GPU uses **bf16** (max ~3.4×10^38)
- bf16-trained model weights produce intermediate values that overflow fp16
- Three overflow points: (1) rmsnorm `diff*diff` when `|diff| > 256`, (2) residual `hidden + sublayer_output`, (3) attention scores before softmax
- **Fix 1: Scaled rmsnorm** — multiply input by 1/128 before variance computation, then use scaled input for normalization. rmsnorm is scale-invariant, so the output is approximately the same (error < 0.001 when mean(x²) >> 128²×eps). Prevents `diff*diff` overflow.
- **Fix 2: Residual stream softcapping** — apply `cap * tanh(output / cap)` after each residual addition in `o_proj_residual` and `ffn_residual` kernels. cap=30000 prevents residual overflow while preserving values within normal range.
- **Fix 3: Attention score softcapping** — apply `cap * tanh(scores / cap)` before softmax in `attn_out` kernel. Prevents softmax overflow from large attention scores.
- With all three fixes: cosine similarity = 0.91 vs GPU (no inf/nan)

## Proven kernel splits (DFlash per layer — CURRENT with mega_qkv + softcapping)
1. `fc_norm`: conv1x1(fc) + scaled_rmsnorm ✓
2. `mega_qkv`: scaled_rmsnorm(input) → Q/K/V projections (input-pack) → per-head norms → RoPE ✓ (3 outputs: k_rope_4d, v_4d_t, q_rope_4d)
3. `gqa_tile`: tile_kv_heads + concat K/V ✓
4. `attn_out`: SDPA with attention score softcapping (cap * tanh(scores/cap)) ✓
5. `o_proj_residual`: conv1x1(o_proj) + residual add + softcapping ✓
6. `ffn_residual`: scaled_rmsnorm → 2×conv1x1 → swiglu → conv1x1 → residual add + softcapping ✓
7. `final_norm`: scaled_rmsnorm ✓

### Full forward pass: cosine = 0.91 vs GPU ✓ (no inf/nan)

### Python round-trips per layer
- After `attn_out`: `_flatten_attn_4d` (4D→flat rearrangement) — 1 read+write per layer

### Key innovations
- **Input-pack approach (rule #22)**: Instead of separate k_proj_ctx + k_proj_noise + output concat (which fails at runtime), pack context + normed into [1, HIDDEN, 1, w_kv] and do ONE conv1x1. Mathematically equivalent.
- **Scaled rmsnorm (rule #25)**: Multiply input by 1/128 before variance computation. rmsnorm is scale-invariant, so output ≈ original. Prevents diff² overflow in fp16.
- **Residual softcapping (rule #24)**: cap * tanh(residual / cap) after each residual addition. cap=30000 prevents fp16 overflow while preserving values within normal range.
- **Attention softcapping**: cap * tanh(scores / cap) before softmax. Prevents softmax fp16 overflow.

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

### Bug #8: FP16 overflow in rmsnorm diff² (FIXED via scaled rmsnorm)
- **Symptom**: ANE output has inf values starting from layer 1. After layer 0, hidden state max ~30000.
- **Root cause**: rmsnorm computes `diff = x - mean(x)`, then `sq = diff * diff`. When `|diff| > 256`, `diff² > 65504` overflows fp16. With hidden states of ±30000, diff values easily exceed 256.
- **Fix**: multiply input by 1/128 before variance computation. Since rmsnorm is scale-invariant (`rmsnorm(x/S) ≈ rmsnorm(x)` when mean(x²) >> S²×eps), the output is approximately unchanged. The intermediate values (diff/128)² are 16384× smaller, preventing overflow.
- **Lesson**: any computation that squares values in fp16 must ensure the input is < 256. Use input scaling to keep values in safe range.

### Bug #9: FP16 overflow in residual connections (FIXED via softcapping)
- **Symptom**: even with scaled rmsnorm, ANE output has inf from residual `hidden + sublayer_output`.
- **Root cause**: bf16-trained model produces sublayer outputs that, when added to the already-large hidden state, exceed fp16 max (65504).
- **Fix**: apply `cap * tanh(output / cap)` after each residual addition. With cap=30000, values within normal range are nearly unchanged, while values that would overflow are compressed.
- **Lesson**: fp16 residual connections are inherently limited. Softcapping bounds the residual stream while preserving most of the signal.

## Findings from autoresearch-ANE reference

### Dynamic Weight Pipeline (Key Architecture)
- Weights packed into IOSurface input ALONGSIDE activations: `[1, IC, 1, SEQ+OC]`
- Spatial positions `[0:SEQ]` = activations, `[SEQ:SEQ+OC]` = weights
- `slice_by_size` in MIL extracts each part
- Allows compile-once, update-weights-via-memcpy — no recompilation needed

### Kernel Fusion: Mega-Kernels Work
- Their sdpaFwd kernel does: QKV projections + RoPE + GQA tiling + SDPA attention — all in ONE dispatch
- Their ffnFused kernel does: 2 matmuls + SiLU + 1 matmul + residual add — all in ONE dispatch
- **10 compiled kernels total** for entire training step (shared across all layers)
- Key: per-layer IOSurfaces with pre-staged weights; only activations updated per step

### Concatenated Outputs Reduce IO
- sdpaFwd outputs (attn_out, Q_rope, K_rope, V, xnorm) as ONE concatenated tensor
- ffnFused outputs (x_next, h1, h3, gate) as ONE concatenated tensor
- ONE IOSurface read gives all needed values instead of 5 separate reads

### Causal Mask as BLOBFILE Constant
- Pre-computed fp16 mask: `0.0` for allowed, `-65504.0` for blocked (fp16 closest to -inf)
- Stored as weight file, loaded via BLOBFILE path
- For speculative decoding (non-causal), we use runtime mask input instead

### FP16 Overflow Mitigation: Logit Softcapping
- `logits = cap * tanh(logits / cap)` with cap=15.0
- Prevents logits from exceeding [-15, +15], keeping softmax stable in fp16
- **We now use this approach** for attention scores (cap=30000) and residual outputs (cap=30000)
- Our caps are much higher than theirs (15) because our model wasn't trained with softcapping — we use the minimum cap needed to prevent overflow while preserving signal

### MIL reshape+transpose+matmul Pattern (vs our conv1x1)
- Their matmul approach: `reshape([1,1,SEQ,IC]) → transpose([0,1,3,2]) → matmul → transpose → reshape`
- This works because transpose is on INTERMEDIATE tensors (outputs of reshape), not placeholders
- Our conv1x1_dynamic approach is equivalent but avoids the reshape/transpose overhead

### GQA Tiling Inside Kernel
- Uses MIL `concat` to tile KV heads within the sdpaFwd kernel
- Avoids separate tiling kernel and CPU round-trip

### ANE SRAM Limit
- SEQ > 1024 fails (runs out of on-chip SRAM)
- SEQ=1152+ fails compilation
- Our w_kv=128 is well within limits

### Performance: 6-8% ANE Utilization
- Their 48.8M param model gets 6-8% utilization (600-800 GFLOP/s of 10.5 TFLOP/s peak)
- Our 5-layer draft model is much smaller, so 4% utilization is consistent
- **Utilization is fundamentally limited by SRAM and I/O, not compute**

### Per-Layer IOSurface Architecture
- Each layer gets its OWN pre-allocated IOSurface and pre-bound request
- Weights pre-staged into each layer's IOSurface once
- Only activations overwritten per step (weight region stays until Adam update)
- **This is the key to eliminating Python round-trips**: pre-stage everything in native code

### IO Copy Between IOSurfaces
- Direct fp16 copy between IOSurfaces (`io_copy`) — no f16→f32→f16 conversion
- Useful for chaining kernel outputs to next kernel inputs without CPU read/write

### ANE Compile Limit
- ~119 compiles before ANE resources exhausted
- Their solution: compile ONCE (10 kernels), reuse across all layers
- Our solution: compile 13 kernels once, reuse across 5 layers
