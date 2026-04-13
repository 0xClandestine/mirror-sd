# Benchmarking Methodology

## One command, walk away

```
python -m mirror_sd.benchmarks.llama_benchy \
  --model ~/.omlx/models/Qwen3.5-27B-4bit \
  --draft z-lab/Qwen3.5-27B-DFlash
```

This runs all three spec configs (dflash, dflash+KOD, dflash+auto_ar) plus the AR baseline, sequentially, saves timestamped JSON results to `mirror_sd/benchmarks/data/`.

## Configurations

| Config | Description |
|--------|-------------|
| baseline | AR decode via `mlx_lm.server` |
| dflash | Spec decode, block_size=16 |
| dflash+kod | Spec decode + Kelly-Optimal Drafting |

Each spec config runs: start server → llama-benchy → kill server → next. Only one server at a time on the GPU.

## Parameters

| Parameter | Value |
|-----------|-------|
| pp | 128 |
| tg | 512 |
| depths | 0, 512, 2048, 4096, 8192, 16384 |
| runs | 3 |
| temperature | 0.0 (greedy) |

## Output

Speedup curve per config across context depths:

```
Depth    Baseline    DFlash    Speedup    DFlash+KOD    Speedup
    0       24.0      37.0      1.54x         35.2       1.47x
  512       23.5      23.7      1.01x         22.1       0.94x
 2048       22.3      18.2      0.82x         ...
 4098       ...
 8192       ...
16384       ...
```

Crossover point (speedup < 1.0x) is the key result.

## Rules

1. **End-to-end wall-clock only** — no per-phase `mx.eval()` timing in published results
2. **≥3 trials per depth** — Apple Silicon thermal variance means single runs are noise
3. **Losslessness is a hard gate** — no speedup claim without losslessness verification first
4. **Report the curve** — speedup varies with context depth, a single number is meaningless

## Losslessness

Before benchmarking a new config, verify token-identical output vs AR:

```
python -m mirror_sd.benchmarks.llama_benchy \
  --model ~/.omlx/models/Qwen3.5-27B-4bit \
  --draft z-lab/Qwen3.5-27B-DFlash \
  --verify-losslessness
```

Known lossy: `adaptive_block=True` (SSM drift causes argmax flip at ~pos 88).

## View results

```
python -m mirror_sd.benchmarks.view spec.json baseline.json --chart comparison.png
```
