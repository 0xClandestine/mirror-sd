# ANE Optimization Attempts

Log of experiments. Each entry: what was tried, how to reproduce, results, conclusion.

Format:
```
## EXP-NNN: <title>  [status: pending | running | done | reverted]
Date: YYYY-MM-DD
Idea ref: IDEA-XX
Branch: <branch>
```

---

## Baseline Numbers (reference)

Measured on: 2026-04-16
Hardware: Apple Silicon (M-series)
Model: Qwen3.5-27B-4bit (target) + Qwen3.5-27B-DFlash (draft)
Block size: 16, ctx_len: 256 (best ctx)

| Mode              | tok/s | α     | draft/step | verify/step | overlap |
|-------------------|-------|-------|------------|-------------|---------|
| AR baseline       | ~23   | —     | —          | —           | —       |
| GPU-only spec     | ~23   | 4.61  | serial     | 197ms       | 0%      |
| ANE\|\|GPU spec   | ~9.9  | 2.12  | 153ms      | 197ms       | 95.8%   |

ANE utilization during GPU verify: ~25–32%
Root cause of α gap: staleness (draft N+1 anchored to correction N-1, not N)

---

## EXP-001: QK-norm ANE/GPU parity fix  [status: in_progress]
Date: 2026-04-16
Idea ref: IDEA-11
Branch: fix/ane-qk-norm

### Motivation
ANE mega_qkv applies QK norms differently than the GPU path. Mismatched norms
→ divergent attention patterns → lower acceptance rate.

### Hypothesis
Fixing QK-norm parity will close part of the α gap between ANE (2.12) and GPU (4.61).

### Reproduction
```bash
git checkout fix/ane-qk-norm
uv run python bench_ane_profile.py --max-tokens 256 --runs 3
```

### Results
- TBD

### Conclusion
- TBD

---

## EXP-002: Prefix-split provisional token  [status: reverted]
Date: 2025-04  (pre-log)
Idea ref: (predates this log)
Branch: reverted to main

### What was tried
Use argmax(lm_head(norm(h_K[:, 0:1, :]))) as a "provisional" correction token to
eliminate the one-step staleness lag.

### Why it failed
Provisional = prediction of what follows position-0, which equals correction_N only
when acceptance_length=0 (47.5% of steps). For the other 52.5%, it predicts the
wrong position → P(provisional correct) ≈ 33% < stale's 55%.
Additionally, `inner.norm` must be applied before lm_head (lm_head trained on
post-norm output); this was a bug in the implementation.

### Conclusion
Fundamentally unsound without knowing acceptance_length (only known after verify).
Serial draft-after-verify gives α≈3.5 but step≈374ms → 9.4 tok/s (worse).

---

## EXP-003: Deferred GPU lm_head onto _draft_stream  [status: done]
Date: 2026-04-16
Idea ref: (new — discovered from pipeline profiling)
Branch: fix/ane-qk-norm
Commit: afeaf4f

### Motivation
bench_ane_pipeline.py showed `lm_head + argmax + eval` = 7.69ms running
serially on the main stream before verify started. This is pure waste:
verify takes 197ms and lm_head only needs to be ready after verify.

### Hypothesis
Submit lm_head to `_draft_stream` without eval so it runs concurrently
with `_start_draft` prep (~4ms) and the first 8ms of verify.
By the time `int(sampled_tokens[0, i])` forces materialization (after
verify), lm_head finished ~190ms ago.

### Change
`generate.py` `_ane_kernels_pending` block: wrap lm_head in
`with mx.stream(_draft_stream):` and remove `mx.eval(sampled_tokens)`.

### Results
```
                        baseline    deferred
lm_head phase (ms)        7.69        0.01
step total (ms)          71.0        63.2
savings                              -11%
```

### Conclusion
Confirmed. ~8ms/step saved with zero complexity cost. For 27B production
model the saving scales similarly (lm_head hidden=5120 → larger matmul,
larger absolute saving). Step time for 27B: ~203ms → ~195ms estimated.

---

## EXP-004: ANE fused final_norm+lm_head kernel  [status: blocked]
Date: 2026-04-16
Idea ref: IDEA-02 (variant)
Branch: fix/ane-qk-norm

### Motivation
Infrastructure already exists: `compile_lm_head_kernel()`, `run_kernels()`
uses `final_norm_lm_head` when `b_logits is not None`, `read_draft_tokens()`
reads argmax via NEON (~0.5ms). This would eliminate both `read_output`
(3.7ms) and GPU lm_head (7.7ms) — saving ~11ms vs EXP-003's 8ms.

### Blocker
ANE `convolution_2d_1x1_dynamic` has an output-channel limit well below
vocab_size=151936. `compile_lm_head_kernel(seq_q=16, vocab_size=151936)`
fails: `_ANECompiler: ANECCompile() FAILED`.

### Next steps
- Find the exact ANE channel limit (likely 4096 or 8192)
- Consider chunked lm_head: split vocab into K chunks, run K ANE passes,
  merge argmax. K=37 at 4096 ch/chunk adds dispatch overhead.
- Alternative: run the final_norm on ANE (saves the final_norm kernel from
  run_kernels), then do lm_head entirely on GPU from b_output.
  This doesn't help since final_norm is only 0.16ms.

---

## Template

## EXP-NNN: <title>  [status: pending]
Date: YYYY-MM-DD
Idea ref: IDEA-XX
Branch: <branch or 'n/a'>

### Motivation


### Hypothesis


### Reproduction
```bash

```

### Results


### Conclusion

