# mirror_sd

DFlash block-diffusion speculative decoding for Apple Silicon via MLX.

Implements the DFlash algorithm: a 5-layer draft model predicts block-level tokens using the target model's hidden states, then the target verifies all draft tokens in a single forward pass. On high-acceptance inputs (math, code), this produces 1.5x+ speedup over autoregressive decoding for Qwen3.5-27B.

## Files

### Core decode loop

| File | Description |
|------|-------------|
| `generate.py` | Main decode loop. `spec_generate()` runs the draft→verify→rollback cycle. `ar_generate()` runs single-token autoregressive decode through the same code path (for fair benchmarking). `SpecDecodeStats` tracks throughput, acceptance lengths, and per-phase timing. KOD (Kelly-Optimal Drafting) block-size selection and auto-AR fallback are implemented here. |

### Model implementations

| File | Description |
|------|-------------|
| `dflash.py` | DFlash draft model and KV cache. `DFlashDraftModel` is the 5-layer SSM+dense-attention draft model that takes target hidden states + noise embeddings and outputs draft token predictions. `DFlashKVCache` handles the draft's cache with `trim()` for removing speculative positions after verification. `extract_context_feature()` selects and concatenates hidden states from target layers. |
| `target.py` | Target model forward passes with hidden-state capture and rollback support. `forward_with_hidden_states()` runs the full 64-layer target model while capturing hidden states at specified layers. `forward_with_hidden_states_and_rollback()` records SSM/conv/KV state for rollback on partial rejection. `rollback_linear_caches()` restores cache state after rejection. Compiled variants (`_compiled`, `_compiled_whole`) wrap layers in `mx.compile` for graph optimization. |
| `ssm_kernel.py` | Custom Metal GPU kernel for GatedDeltaNet SSM state replay. `advance_gated_delta_states_metal()` batch-replays all 48 SSM layers in a single GPU kernel call (23% speedup over per-layer Python loop). Used during rollback to restore SSM states to a checkpoint. |
| `ane_model.py` | DFlash draft model running on Apple Neural Engine. `ANEDraftModel` compiles each draft layer as a separate CoreML ANE kernel, enabling parallel ANE||GPU execution (Mirror-SD mode). Has GPU fallback for contexts exceeding ANE limits. Experimental — accuracy is lower than GPU draft. |
| `turboquant.py` | TurboQuant KV cache integration. `TurboQuantKVCache` wraps the standard KV cache with quantization to reduce memory at long contexts. `make_turboquant_cache()` creates quantized caches for all layers. Currently slower than standard cache due to quantization overhead. |

### Infrastructure

| File | Description |
|------|-------------|
| `loader.py` | Weight loading and model conversion. `load_dflash_model()` resolves HuggingFace cache paths, loads the DFlash config and weights, and returns `(DFlashDraftModel, DFlashConfig)`. `convert_dflash_to_mlx()` converts PyTorch DFlash checkpoints to MLX format. |
| `prompt.py` | Chat template utilities. `format_prompt()` wraps text in the Qwen3.5 chat template with `/no_think` injection (required for good DFlash acceptance). `get_stop_token_ids()` returns EOS token IDs for the loaded tokenizer. |
| `server.py` | OpenAI-compatible HTTP server. Implements `/v1/chat/completions` and `/v1/completions` endpoints. Supports prompt caching (LRU), streaming, and all spec-decode flags (KOD, auto-AR, TurboQuant, etc.). |
| `cli.py` | Command-line interface. Subcommands: `generate` (run spec decode), `bench` (run benchmark), `convert` (convert PyTorch weights to MLX). |
| `train.py` | DFlash draft model training. `DFlashTrainModel` wraps the draft model with a training forward pass. `train()` runs the training loop with teacher-forcing on target model hidden states. `prepare_data()` tokenizes and precomputes target hidden states for training. |

### Benchmarks

| File | Description |
|------|-------------|
| `benchmarks/llama_benchy.py` | Benchmark runner. Starts spec and baseline servers as subprocesses, runs the external `llama-benchy` tool at varying context depths, and saves timestamped JSON results. This is the primary benchmark per METHOD.md. |
| `benchmarks/view.py` | CLI to view and compare llama-benchy results. `python -m mirror_sd.benchmarks.view spec.json` shows a single result; `python -m mirror_sd.benchmarks.view spec.json baseline.json` compares with speedup table and chart. |
| `benchmarks/mmlu.py` | MMLU accuracy benchmark. Evaluates both accuracy and throughput on the MMLU multiple-choice benchmark using 5-shot prompting. |

## Data flow

```
Input prompt
    │
    ▼
format_prompt() ──► tokenized input_ids
    │
    ▼
forward_with_hidden_states(input_ids, capture_layers=[0,16,32,48])
    │
    ├──► logits ──► first token (greedy/sample)
    └──► hidden_states ──► extract_context_feature() ──► target_hidden
                                                    │
                                        ┌───────────┘
                                        ▼
                              DFlashDraftModel(noise_emb, target_hidden)
                                        │
                                        ▼
                              draft_logits ──► sampled_tokens
                                        │
                                        ▼
              forward_with_hidden_states_and_rollback(anchor + sampled_tokens)
                                        │
                        ┌───────────────┼───────────────┐
                        ▼               ▼               ▼
                   verify_logits   verify_hidden   rollback_records
                        │
                        ▼
               accept/reject comparison
                        │
              ┌─────────┼─────────┐
              ▼                   ▼
         accepted tokens    rollback_linear_caches()
              │                   │
              ▼                   ▼
         output_ids       advance_gated_delta_states_metal()
              │                   │
              └───────┬───────────┘
                      ▼
              next iteration (draft → verify → rollback)
```
