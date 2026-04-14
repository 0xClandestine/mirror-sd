"""Autoregressive generation for Mirror-SD.

Also re-exports sample() and SpecDecodeStats for use by dflash.runtime.
"""

import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx

from .target import forward_with_hidden_states


def sample(logits: mx.array, temperature: float = 0.0) -> mx.array:
    if temperature < 1e-5:
        return mx.argmax(logits, axis=-1)
    return mx.random.categorical(logits / temperature, axis=-1)


def _flat_cache_states(cache):
    out = []
    for c in cache:
        s = c.state
        if isinstance(s, list):
            out.extend(s)
        else:
            out.append(s)
    return out


@dataclass
class SpecDecodeStats:
    total_tokens: int = 0
    accepted_tokens: int = 0
    draft_steps: int = 0
    total_time: float = 0.0
    prefill_time: float = 0.0
    acceptance_lengths: List[int] = field(default_factory=list)
    spec_lengths: List[int] = field(default_factory=list)
    parallel_mode: bool = False
    total_draft_time: float = 0.0
    total_verify_time: float = 0.0
    total_rollback_time: float = 0.0
    total_misc_time: float = 0.0
    total_overlap_time: float = 0.0

    @property
    def acceptance_rate(self) -> float:
        return self.accepted_tokens / max(self.draft_steps, 1)

    @property
    def avg_acceptance_length(self) -> float:
        if not self.acceptance_lengths:
            return 0.0
        return sum(self.acceptance_lengths) / len(self.acceptance_lengths)

    @property
    def avg_spec_length(self) -> float:
        if not self.spec_lengths:
            return 0.0
        return sum(self.spec_lengths) / len(self.spec_lengths)

    @property
    def tokens_per_sec(self) -> float:
        gen_time = self.total_time - self.prefill_time
        return self.total_tokens / max(gen_time, 1e-9)


def ar_generate(
    target_model,
    input_ids: mx.array,
    max_new_tokens: int,
    stop_token_ids=None,
    temperature: float = 0.0,
    target_layer_ids=None,
    prefill_step_size: int = 512,
    stream_callback=None,
) -> Tuple[mx.array, SpecDecodeStats]:
    from mlx_lm.models import cache as cache_module

    if target_layer_ids is None:
        target_layer_ids = []

    stats = SpecDecodeStats()
    t_start = time.perf_counter()

    target_cache = cache_module.make_prompt_cache(target_model)
    output_ids_list = input_ids.tolist()[0] if input_ids.ndim == 2 else input_ids.tolist()
    if isinstance(output_ids_list, int):
        output_ids_list = [output_ids_list]

    num_input = input_ids.shape[1]

    if num_input > prefill_step_size:
        for start in range(0, num_input - prefill_step_size, prefill_step_size):
            chunk = input_ids[:, start:start + prefill_step_size]
            logits_chunk, _, _ = forward_with_hidden_states(
                target_model, chunk, cache=target_cache, capture_layers=[],
            )
            mx.eval(logits_chunk, *_flat_cache_states(target_cache))
            del logits_chunk
        last_start = (num_input // prefill_step_size) * prefill_step_size
        if last_start >= num_input:
            last_start = max(0, num_input - prefill_step_size)
        last_chunk = input_ids[:, last_start:]
        logits, _, hidden_states = forward_with_hidden_states(
            target_model, last_chunk, cache=target_cache,
            capture_layers=target_layer_ids,
        )
        first_token = sample(logits[:, -1:, :], temperature)
        mx.eval(logits, *hidden_states, first_token, *_flat_cache_states(target_cache))
    else:
        logits, _, hidden_states = forward_with_hidden_states(
            target_model, input_ids, cache=target_cache,
            capture_layers=target_layer_ids,
        )
        first_token = sample(logits[:, -1:, :], temperature)
        mx.eval(logits, *hidden_states, first_token, *_flat_cache_states(target_cache))

    first_tok = int(first_token[0, 0])
    output_ids_list.append(first_tok)
    if stream_callback is not None:
        stream_callback(first_tok)

    stats.prefill_time = time.perf_counter() - t_start

    start = num_input + 1
    max_length = num_input + max_new_tokens

    while start < max_length:
        last_tok = mx.array([[output_ids_list[-1]]], dtype=mx.int32)
        logits, _, _ = forward_with_hidden_states(
            target_model, last_tok, cache=target_cache,
            capture_layers=[],
        )
        next_token = sample(logits[:, -1:, :], temperature)
        mx.eval(next_token, *_flat_cache_states(target_cache))
        next_tok = int(next_token[0, 0])
        output_ids_list.append(next_tok)
        if stream_callback is not None:
            stream_callback(next_tok)
        start += 1
        stats.total_tokens += 1
        if stop_token_ids is not None:
            for stop_id in stop_token_ids:
                if stop_id in output_ids_list[num_input + 1:]:
                    break
            else:
                continue
            break

    stats.total_time = time.perf_counter() - t_start
    stats.draft_steps = stats.total_tokens

    output_ids = mx.array([output_ids_list])
    return output_ids, stats
