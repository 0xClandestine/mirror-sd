"""Speculative decoding loop for Mirror-SD.

Implements the DFlash block-diffusion speculative decoding algorithm
translated from the PyTorch reference to MLX.

Key differences from standard speculative decoding:
- Draft model generates entire BLOCKS of tokens in parallel (not autoregressive)
- Draft uses target's lm_head for final logits (shares vocabulary)
- Draft attention is non-causal (block diffusion)
- Draft KV cache is cropped after each block (not accumulated)
- Target hidden states are captured and fused as context for the draft

Flow:
1. PREFILL: Run target on prompt, get first token + hidden states
2. DECODE LOOP:
   a. Create block of tokens at current position (first is real, rest are mask)
   b. Compute noise_embedding from target's embed_tokens
   c. Run DFlash forward with target hidden -> draft hidden states
   d. Pass through target's lm_head for logits
   e. Sample draft tokens from logits, update block
   f. VERIFY: Run target model on block
   g. Accept matching prefix + correction token from target
   h. Crop caches and extract new target hidden states
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from .dflash import DFlashDraftModel, DFlashConfig, extract_context_feature, sample
from .target import forward_with_hidden_states


@dataclass
class SpecDecodeStats:
    total_tokens: int = 0
    accepted_tokens: int = 0
    draft_steps: int = 0
    total_time: float = 0.0
    prefill_time: float = 0.0
    acceptance_lengths: List[int] = field(default_factory=list)

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.draft_steps, 1)

    @property
    def avg_acceptance_length(self) -> float:
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    @property
    def tokens_per_sec(self) -> float:
        gen_time = self.total_time - self.prefill_time
        return self.total_tokens / max(gen_time, 1e-9)


def spec_generate(
    target_model,
    draft_model: DFlashDraftModel,
    input_ids: mx.array,
    max_new_tokens: int,
    stop_token_ids: Optional[List[int]] = None,
    temperature: float = 0.0,
    target_layer_ids: Optional[List[int]] = None,
) -> Tuple[mx.array, SpecDecodeStats]:
    """DFlash speculative decoding loop.

    Follows the algorithm from the DFlash reference implementation:
    - Draft model generates blocks using target-aware attention
    - Draft logits come from target's lm_head (not draft's own)
    - Draft KV cache is fresh each block (cropped to current position)
    - Target hidden states are extracted and fused as draft context

    Args:
        target_model: The target mlx-lm model
        draft_model: The DFlash draft model
        input_ids: Input token IDs [1, seq_len]
        max_new_tokens: Maximum tokens to generate
        stop_token_ids: Token IDs that stop generation
        temperature: Sampling temperature (0 = greedy)
        target_layer_ids: Which target layers to capture hidden states from

    Returns:
        output_ids: Generated token IDs [1, total_len]
        stats: Generation statistics
    """
    from mlx_lm.models import cache as cache_module

    if target_layer_ids is None:
        target_layer_ids = draft_model.config.target_layer_ids

    stats = SpecDecodeStats()
    t_start = time.perf_counter()

    block_size = draft_model.block_size
    mask_token_id = draft_model.mask_token_id
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids_list = input_ids.tolist()[0] if input_ids.ndim == 2 else input_ids.tolist()
    if isinstance(output_ids_list, int):
        output_ids_list = [output_ids_list]

    target_cache = cache_module.make_prompt_cache(target_model)

    # --- Prefill stage ---
    logits, embed, hidden_states = forward_with_hidden_states(
        target_model, input_ids, cache=target_cache, capture_layers=target_layer_ids,
    )
    mx.eval(logits, embed, *hidden_states)
    mx.eval([c.state for c in target_cache])

    first_token = sample(logits[:, -1:, :], temperature)
    mx.eval(first_token)
    output_ids_list.append(int(first_token[0, 0]))

    target_hidden = extract_context_feature(hidden_states, target_layer_ids)

    stats.prefill_time = time.perf_counter() - t_start

    # --- Decode stage ---
    start = num_input_tokens + 1
    while start < max_length:
        remaining = max_length - start
        current_block_size = min(block_size, remaining + 1)

        block_tokens = [output_ids_list[start - 1]]
        block_tokens.extend([mask_token_id] * (current_block_size - 1))
        block_output_ids = mx.array([block_tokens], dtype=mx.int32)

        noise_embedding = target_model.model.embed_tokens(block_output_ids)

        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
        )

        draft_logits = target_model.lm_head(draft_hidden[:, -current_block_size + 1:, :])
        mx.eval(draft_logits)

        sampled_tokens = sample(draft_logits, temperature)
        mx.eval(sampled_tokens)

        block_tokens_updated = block_tokens.copy()
        for i in range(sampled_tokens.shape[1]):
            block_tokens_updated[i + 1] = int(sampled_tokens[0, i])
        block_output_ids = mx.array([block_tokens_updated], dtype=mx.int32)

        verify_logits, _, verify_hidden = forward_with_hidden_states(
            target_model,
            block_output_ids,
            cache=target_cache,
            capture_layers=target_layer_ids,
        )
        mx.eval(verify_logits, *verify_hidden)
        mx.eval([c.state for c in target_cache])

        posterior = sample(verify_logits, temperature)
        mx.eval(posterior)

        draft_tokens = block_tokens_updated[1:]
        target_tokens = posterior[0, :-1].tolist()
        if isinstance(target_tokens, int):
            target_tokens = [target_tokens]

        acceptance_length = 0
        for i in range(len(draft_tokens)):
            if draft_tokens[i] == target_tokens[i]:
                acceptance_length += 1
            else:
                break

        for i in range(acceptance_length):
            output_ids_list.append(draft_tokens[i])
        correction_token = int(posterior[0, acceptance_length])
        output_ids_list.append(correction_token)

        start += acceptance_length + 1

        n_to_trim = current_block_size - acceptance_length - 1
        if n_to_trim > 0:
            cache_module.trim_prompt_cache(target_cache, n_to_trim)

        target_hidden = extract_context_feature(verify_hidden, target_layer_ids)[:, :acceptance_length + 1, :]

        stats.acceptance_lengths.append(acceptance_length + 1)
        stats.accepted_tokens += acceptance_length
        stats.draft_steps += 1
        stats.total_tokens = len(output_ids_list) - num_input_tokens

        if stop_token_ids is not None:
            for stop_id in stop_token_ids:
                if stop_id in output_ids_list[num_input_tokens:]:
                    break
            else:
                continue
            break

    stats.total_time = time.perf_counter() - t_start

    if stop_token_ids is not None:
        for i, tid in enumerate(output_ids_list[num_input_tokens:]):
            if tid in stop_token_ids:
                output_ids_list = output_ids_list[:num_input_tokens + i + 1]
                break

    output_ids = mx.array([output_ids_list], dtype=mx.int32)
    stats.total_tokens = len(output_ids_list) - num_input_tokens
    return output_ids, stats
