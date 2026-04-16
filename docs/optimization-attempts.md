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

