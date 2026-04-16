# ANE Optimization Ideas

Running backlog. Move entries to `optimization-attempts.md` when tried.

---

## Kernel Fusion / Dispatch Reduction

### IDEA-01: Fuse gqa_tile into mega_qkv output stage
**Hypothesis**: `gqa_tile` is a KV head-expansion step that reads back the output of
`mega_qkv`. If it can be computed inside the same CoreML program as `mega_qkv`, we
eliminate one round-trip through ANE SRAM and one dispatch overhead (~0.3ms per layer).
**Risk**: increases `mega_qkv` program size; may hurt tiling / occupancy.
**Metric**: per-layer wall time; total forward ms.

### IDEA-02: Fuse final_norm into ffn_residual output slot
**Hypothesis**: `final_norm` runs after the last layer's `ffn_residual`. If the last
layer's `ffn_residual` kernel variant writes a normed output instead of raw hidden, we
save one dispatch + buffer read (~0.5ms).
**Risk**: creates a special-cased last-layer kernel variant; compile complexity.

### IDEA-03: Eliminate per-layer run_uncached calls — compile multi-layer program
**Hypothesis**: dispatching 5 kernels × 5 layers = 25 calls. Compiling a single
multi-layer CoreML program would reduce that to 1 dispatch with all weights bound.
**Risk**: very large program; weight patching between steps may be infeasible.
**Prerequisite**: understand if CoreML programs can accept per-step varying inputs.

---

## Memory / Buffer Bandwidth

### IDEA-04: Reduce b_context width padding
`w_ctx = align_width(ctx_len)` always rounds up to the next multiple of 64.
For ctx_len=64 this is fine, but for ctx_len=48 we waste 25% bandwidth writing context.
**Hypothesis**: smaller minimum alignment (32?) or dynamic shapes could help small-ctx.
**Risk**: alignment is a hard ANE requirement; need to check minimum for the SoC.

### IDEA-05: fp16 context buffer
`b_context` is written as float32. The `_compute_context` GPU matmul outputs fp32, but
ANE reads it as fp16 internally anyway. If we can write fp16 to the ANE buffer we halve
context-write bandwidth.
**Risk**: precision; needs validation that acceptance rate is unchanged.
**Metric**: context_fc_write phase time; acceptance_length distribution.

### IDEA-06: Reuse KV buffer across draft steps (KV cache on ANE)
Currently KV cache lives in MLX (GPU) memory and is written to ANE buffers each step.
If the ANE KV buffer is persistent across draft steps, we only write new tokens each
step instead of the full ctx_len.
**Risk**: ANE buffer lifetime / ownership is not directly controllable from Python.
**Prerequisite**: measure what fraction of forward time is KV buffer write.

---

## Scheduling / Pipeline

### IDEA-07: Overlap context computation with previous verify
`_compute_context` (GPU matmul, ~2ms) runs synchronously on the main thread before the
ANE thread is launched. It could potentially run concurrently with the last part of
the previous GPU verify pass.
**Hypothesis**: Move `_compute_context` earlier (right after verify starts) to hide
it in the verify latency.
**Risk**: verify uses GPU; `_compute_context` uses GPU — may serialize anyway.
**Metric**: `context_fc_write` wall time as seen from main thread.

### IDEA-08: Increase block_size from 16 to 24 or 32
Larger block_size means more tokens drafted per step. With α≈2.12, we currently waste
~30% of draft capacity (drafting 16, accepting ~2.12 on average).
**But**: larger block_size increases ANE forward cost linearly (seq_q dimension).
**Net effect**: needs measurement. α is likely slightly lower for larger blocks too.
**Metric**: end-to-end tok/s, not just per-step metrics.

### IDEA-09: Adaptive block_size based on recent acceptance rate
If recent α is high (e.g. >3), increase block_size. If low (<2), decrease.
**Risk**: requires recompiling ANE model for each block_size (slow startup).
**Mitigation**: pre-compile a set {8, 16, 24, 32} and switch at runtime.

---

## Acceptance Rate

### IDEA-10: Temperature-matched draft sampling
If target runs at temperature > 0, the draft's argmax is mismatched. Use
top-p / top-k sampling in the draft to better match target distribution.
**Expected gain**: higher α → fewer verify steps → higher tok/s even if draft is slower.
**Risk**: requires benchmarking at non-zero temperature.

### IDEA-11: QK-norm calibration for ANE vs GPU parity
The current branch (`fix/ane-qk-norm`) suggests QK norms may differ between ANE and
GPU implementations. Any discrepancy reduces α.
**Action**: quantify logit divergence between ANE and GPU draft on the same inputs.
**Metric**: KL divergence of draft logit distributions; α delta.

---

## Infrastructure

### IDEA-12: Continuous bench harness with result DB
Save every bench run to a SQLite DB (`bench_results.db`) with git commit hash,
timestamp, config, and all metrics. Makes regression tracking automatic.

### IDEA-14: Run read_output + lm_head-submit inside the ANE thread
**Hypothesis**: `read_output()` (3.7ms IOSurface→mx.array copy) currently runs on the
main thread after join. If we allow it to run at the END of the ANE thread (after
`run_kernels()` completes, before the thread exits), then by the time main thread
joins, `draft_hidden` is already a full mx.array and lm_head is already submitted
to `_draft_stream`. Main thread join → check draft_result → start_verify, with 0
main-thread overhead from read_output/lm_head.
**Risk**: `_read_mlx_2d` creates mx.array (Metal buffer alloc) from a non-main thread.
MLX uses a thread-safe allocator so this is likely safe. The IOSurface has no data
race since `run_kernels()` is fully done before `read_output()` is called.
**Expected gain**: 3.7ms + 0ms (lm_head now in thread, runs during verify) = 3.7ms
additional saving on top of EXP-003. Step: 63ms → 59ms for 8B model.
**Metric**: bench_ane_pipeline.py after restructuring `_run_draft` / `_ane_kernels_pending`.

### IDEA-15: Chunked ANE lm_head (split vocab into 4096-channel blocks)
**Hypothesis**: ANE conv1x1 channel limit prevents compiling the full lm_head
(vocab=152K). If we compile N=37 kernels of 4096 output channels each and run them
sequentially, the full vocab projection happens on ANE. NEON argmax over the 37
partial outputs would still be CPU-cheap.
**Risk**: 37 ANE dispatches × dispatch overhead (~0.3ms each?) = ~11ms overhead.
ANE lm_head benefit (~11ms saved) may be fully eaten by dispatch overhead.
**Prerequisite**: Measure actual ANE channel limit and dispatch overhead per kernel.
**Better alternative**: Single chunked kernel that tiles internally (CoreML may do this).

### IDEA-13: Thermal throttle detection
ANE power often drops under sustained load (thermal). bench_ane_power.py already
samples power; add detection: if avg_mw in second half < 80% of first half,
flag as "thermally throttled" and discard or separate results.
