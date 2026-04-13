"""Speculative decoding loop for Mirror-SD.

Implements the DFlash block-diffusion speculative decoding algorithm
translated from the PyTorch reference to MLX.

Optimized for MLX lazy evaluation: reduces sync points from 5 to 2 per
decode iteration by building draft and verify as separate lazy graphs.

When --ane is used with an ANEDraftModel, supports parallel ANE||GPU
execution per Mirror-SD Eq. 10:
  - Target runs prefix layers 0:exit_layer → emits hidden states
  - ANE draft starts in parallel with target suffix layers
  - Rendezvous: both results meet for accept/reject
"""

import threading
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import mlx.core as mx

from .dflash import (
    DFlashKVCache,
    extract_context_feature, sample,
)
from .target import (
    forward_with_hidden_states, forward_with_hidden_states_and_rollback,
    forward_with_hidden_states_compiled,
    forward_verifier_states, forward_verifier_states_compiled,
    forward_accept_all_block, forward_accept_all_block_compiled,
    forward_prefix, forward_suffix,
    get_embed_tokens, get_lm_head,
    is_qwen35, rollback_linear_caches, _apply_lm_head,
    _forward_full_attention_layer_compiled,
)


def _flat_cache_states(cache):
    out = []
    for c in cache:
        s = c.state
        if isinstance(s, list):
            out.extend(s)
        else:
            out.append(s)
    return out


def _kod_optimal_gamma(alpha, cost_a, cost_b, min_gamma=2, max_gamma=16):
    """KOD: find block_size that maximizes expected throughput.

    Kelly-Optimal Drafting selects the block size gamma that maximizes
    expected tokens per millisecond, given an estimated per-token
    acceptance probability alpha and a linear cost model.

    Cost model: cost(gamma) = cost_a + cost_b * gamma  (ms)
    Expected tokens: alpha*(1-alpha^(gamma-1))/(1-alpha) + 1

    The expected tokens formula assumes iid acceptance with probability
    alpha, forming a geometric series that models the consecutive-match
    acceptance rule of speculative decoding.

    Args:
        alpha: estimated per-token acceptance probability
        cost_a: fixed cost per iteration (ms)
        cost_b: marginal cost per block_size unit (ms)
        min_gamma, max_gamma: search range for block_size
    Returns:
        Optimal block_size (gamma)
    """
    best_gamma = min_gamma
    best_tp = 0.0
    for gamma in range(min_gamma, max_gamma + 1):
        n = gamma - 1
        if alpha > 0.999:
            ea = float(n)
        elif alpha < 0.001:
            ea = 0.0
        else:
            ea = alpha * (1.0 - alpha ** n) / (1.0 - alpha)
        tp = (ea + 1.0) / (cost_a + cost_b * gamma)
        if tp > best_tp:
            best_tp = tp
            best_gamma = gamma
    return best_gamma


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


def spec_generate(
    target_model,
    draft_model,
    input_ids: mx.array,
    max_new_tokens: int,
    stop_token_ids: Optional[List[int]] = None,
    temperature: float = 0.0,
    target_layer_ids: Optional[List[int]] = None,
    mirror_sd: bool = False,
    failfast: bool = False,
    failfast_tau: float = 0.4,
    failfast_max_spec: int = 64,
    num_draft_layers: Optional[int] = None,
    adaptive_block: bool = True,
    kod: bool = False,
    stream_callback=None,
    prefill_step_size: int = 512,
    prompt_cache=None,
    prefill_callback=None,
    use_compiled: bool = False,
    compile_full: bool = False,
    compiled_whole: bool = False,
    lazy_logits: bool = False,
    logit_chunk_size: int = 1,
    accept_all_first: bool = False,
    turboquant_bits: float = 0.0,
) -> Tuple[mx.array, SpecDecodeStats, list, list, mx.array]:
    from mlx_lm.models import cache as cache_module

    is_ane = hasattr(draft_model, 'ane')
    if is_ane:
        return _spec_generate_parallel(
            target_model, draft_model, input_ids, max_new_tokens,
            stop_token_ids, temperature, target_layer_ids,
        )

    if mirror_sd:
        return _spec_generate_mirror_sd(
            target_model, draft_model, input_ids, max_new_tokens,
            stop_token_ids, temperature, target_layer_ids,
            failfast=failfast, failfast_tau=failfast_tau,
            failfast_max_spec=failfast_max_spec,
        )

    if target_layer_ids is None:
        target_layer_ids = draft_model.config.target_layer_ids

    if num_draft_layers is not None and hasattr(draft_model, 'num_draft_layers'):
        original = draft_model.num_draft_layers
        draft_model.num_draft_layers = min(num_draft_layers, original)
    elif num_draft_layers is not None:
        original = None
    else:
        original = None

    stats = SpecDecodeStats()
    t_start = time.perf_counter()

    q35 = is_qwen35(target_model)
    embed_fn = get_embed_tokens(target_model)
    lm_head_fn = get_lm_head(target_model)

    base_block_size = draft_model.block_size
    block_size = base_block_size
    mask_token_id = draft_model.mask_token_id
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids_list = input_ids.tolist()[0] if input_ids.ndim == 2 else input_ids.tolist()
    if isinstance(output_ids_list, int):
        output_ids_list = [output_ids_list]

    if prompt_cache is not None:
        target_cache = prompt_cache
    elif turboquant_bits > 0:
        from .turboquant import make_turboquant_cache
        target_cache = make_turboquant_cache(target_model, bits=turboquant_bits)
    else:
        target_cache = cache_module.make_prompt_cache(target_model)
    draft_cache = draft_model.make_cache()

    recent_acceptances = []
    hybrid_window = 3
    min_block_size = 2

    kod_confs = []
    kod_accepts = []
    kod_obs = []

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
        logits, embed, hidden_states = forward_with_hidden_states(
            target_model, last_chunk, cache=target_cache,
            capture_layers=target_layer_ids,
        )
        first_token = sample(logits[:, -1:, :], temperature)
        mx.eval(logits, embed, *hidden_states, first_token, *_flat_cache_states(target_cache))
        target_hidden = extract_context_feature(hidden_states, target_layer_ids)
    else:
        logits, embed, hidden_states = forward_with_hidden_states(
            target_model, input_ids, cache=target_cache,
            capture_layers=target_layer_ids,
        )
        first_token = sample(logits[:, -1:, :], temperature)
        mx.eval(logits, embed, *hidden_states, first_token, *_flat_cache_states(target_cache))
        target_hidden = extract_context_feature(hidden_states, target_layer_ids)

    first_tok = int(first_token[0, 0])
    output_ids_list.append(first_tok)
    if stream_callback is not None:
        stream_callback(first_tok)

    if prefill_callback is not None:
        prefill_callback(target_cache, target_hidden, first_tok)

    stats.prefill_time = time.perf_counter() - t_start

    # --- Decode ---
    start = num_input_tokens
    while start < max_length:
        remaining = max_length - start

        if kod and len(kod_obs) >= 2:
            # KOD: Kelly-Optimal Drafting block_size selection
            # Uses observed acceptance rate as alpha estimate (draft confidence
            # is poorly calibrated for DFlash — max softmax ~1.0 always)
            window = min(8, len(kod_accepts))
            alpha_est = sum(kod_accepts[-window:]) / window

            # Auto-calibrate cost model from observed (block_size, time) pairs
            # Need at least 2 different block_sizes for a meaningful fit
            unique_gammas = len(set(g for g, _ in kod_obs[-16:]))
            if unique_gammas >= 2 and len(kod_obs) >= 4:
                gammas = [g for g, _ in kod_obs[-16:]]
                times = [t for _, t in kod_obs[-16:]]
                n_obs = len(gammas)
                sum_g = sum(gammas)
                sum_t = sum(times)
                sum_gg = sum(g * g for g in gammas)
                sum_gt = sum(g * t for g, t in zip(gammas, times))
                denom = n_obs * sum_gg - sum_g ** 2
                if abs(denom) > 0.001:
                    cost_b = (n_obs * sum_gt - sum_g * sum_t) / denom
                    cost_a = (sum_t - cost_b * sum_g) / n_obs
                    cost_a = max(cost_a, 1.0)
                    cost_b = max(cost_b, 0.1)
                else:
                    cost_a, cost_b = 47.0, 8.7
            else:
                cost_a, cost_b = 47.0, 8.7

            current_block_size = _kod_optimal_gamma(
                alpha_est, cost_a, cost_b,
                min_gamma=min_block_size,
                max_gamma=min(base_block_size * 2, 16),
            )
        elif adaptive_block and len(recent_acceptances) >= hybrid_window:
            avg_accept = sum(recent_acceptances[-hybrid_window:]) / hybrid_window
            if avg_accept < 1.0:
                current_block_size = min_block_size
            elif avg_accept < 2.0:
                current_block_size = min(3, base_block_size)
            else:
                current_block_size = base_block_size
        else:
            current_block_size = base_block_size

        current_block_size = min(current_block_size, remaining + 1)

        if current_block_size < 2:
            break

        t_iter_start = time.perf_counter()

        # --- Draft phase ---
        anchor_token = output_ids_list[-1]
        block_tokens = [anchor_token] + [mask_token_id] * (current_block_size - 1)
        block_ids = mx.array([block_tokens], dtype=mx.int32)
        noise_embedding = embed_fn(block_ids)

        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = lm_head_fn(draft_hidden[:, -current_block_size + 1:, :])
        sampled_tokens = sample(draft_logits, temperature)

        # --- Crop draft cache BEFORE start increment ---
        for c in draft_cache:
            c.crop(start)

        # --- Verify phase ---
        anchor_arr = mx.array([[anchor_token]], dtype=mx.int32)
        verify_input = mx.concatenate([anchor_arr, sampled_tokens], axis=1)
        draft_toks = sampled_tokens[0].tolist()
        block_size_actual = len(draft_toks) + 1

        prev_full_accept = (
            accept_all_first
            and len(stats.acceptance_lengths) > 0
            and stats.acceptance_lengths[-1] >= block_size_actual - 1
        )

        if lazy_logits or prev_full_accept:
            if q35:
                if use_compiled:
                    norm_hidden, _, verify_hidden, rollback_records = forward_verifier_states_compiled(
                        target_model,
                        verify_input,
                        cache=target_cache,
                        capture_layers=target_layer_ids,
                    )
                else:
                    norm_hidden, _, verify_hidden, rollback_records = forward_verifier_states(
                        target_model,
                        verify_input,
                        cache=target_cache,
                        capture_layers=target_layer_ids,
                    )
            else:
                norm_hidden, _, verify_hidden, rollback_records = forward_verifier_states(
                    target_model,
                    verify_input,
                    cache=target_cache,
                    capture_layers=target_layer_ids,
                    compile_full=compile_full,
                )

            mx.eval(norm_hidden, *verify_hidden, *_flat_cache_states(target_cache))

            acceptance_length = 0
            correction_token = None
            chunk_size = max(1, logit_chunk_size)

            if prev_full_accept:
                chunk_size = block_size_actual - 1

            for chunk_start in range(0, block_size_actual, chunk_size):
                chunk_end = min(chunk_start + chunk_size, block_size_actual)
                logits_chunk = _apply_lm_head(target_model, norm_hidden[:, chunk_start:chunk_end, :])
                posterior_chunk = sample(logits_chunk, temperature)
                mx.eval(posterior_chunk)

                chunk_tokens = posterior_chunk[0].tolist()

                for local_idx, tok in enumerate(chunk_tokens):
                    pos = chunk_start + local_idx
                    if pos == block_size_actual - 1:
                        correction_token = tok
                        break
                    if tok == draft_toks[pos]:
                        acceptance_length += 1
                    else:
                        correction_token = tok
                        break

                if correction_token is not None:
                    break
        else:
            if q35:
                if compiled_whole:
                    from mirror_sd.target import forward_with_hidden_states_compiled_whole
                    verify_logits, _, verify_hidden, rollback_records = forward_with_hidden_states_compiled_whole(
                        target_model,
                        verify_input,
                        cache=target_cache,
                        capture_layers=target_layer_ids,
                    )
                elif use_compiled:
                    verify_logits, _, verify_hidden, rollback_records = forward_with_hidden_states_compiled(
                        target_model,
                        verify_input,
                        cache=target_cache,
                        capture_layers=target_layer_ids,
                    )
                else:
                    verify_logits, _, verify_hidden, rollback_records = forward_with_hidden_states_and_rollback(
                        target_model,
                        verify_input,
                        cache=target_cache,
                        capture_layers=target_layer_ids,
                    )
            else:
                verify_logits, _, verify_hidden = forward_with_hidden_states(
                    target_model,
                    verify_input,
                    cache=target_cache,
                    capture_layers=target_layer_ids,
                    compile_full=compile_full,
                )
                rollback_records = None
            posterior = sample(verify_logits, temperature)
            mx.eval(posterior, *verify_hidden, *_flat_cache_states(target_cache))

            target_toks = posterior[0, :-1].tolist()
            if isinstance(target_toks, int):
                target_toks = [target_toks]

            acceptance_length = 0
            for i in range(len(draft_toks)):
                if draft_toks[i] == target_toks[i]:
                    acceptance_length += 1
                else:
                    break
            correction_token = int(posterior[0, acceptance_length])

        t_iter_end = time.perf_counter()
        iter_time_ms = (t_iter_end - t_iter_start) * 1000

        if kod:
            draft_probs = mx.softmax(draft_logits, axis=-1)
            draft_max_probs = mx.max(draft_probs, axis=-1)
            avg_conf = float(mx.mean(draft_max_probs))
            kod_confs.append(avg_conf)

        if acceptance_length > 0:
            accepted_toks = draft_toks[:acceptance_length]
            output_ids_list.extend(accepted_toks)
            if stream_callback is not None:
                for t in accepted_toks:
                    stream_callback(t)
        output_ids_list.append(correction_token)
        if stream_callback is not None:
            stream_callback(correction_token)

        start += acceptance_length + 1

        n_to_trim_target = current_block_size - acceptance_length - 1
        if n_to_trim_target > 0:
            if q35 and rollback_records is not None:
                accepted_inputs = acceptance_length + 1
                rollback_tensors = [
                    v for r in rollback_records.values()
                    for v in r.values() if isinstance(v, mx.array)
                ]
                if rollback_tensors:
                    mx.eval(*rollback_tensors)
                rollback_linear_caches(target_cache, rollback_records, accepted_inputs)
            else:
                cache_module.trim_prompt_cache(target_cache, n_to_trim_target)

        target_hidden = mx.concatenate([h[:, :acceptance_length + 1, :] for h in verify_hidden], axis=-1)

        recent_acceptances.append(acceptance_length)
        if len(recent_acceptances) > hybrid_window * 2:
            recent_acceptances = recent_acceptances[-hybrid_window:]

        if kod:
            accept_rate = acceptance_length / max(current_block_size - 1, 1)
            kod_accepts.append(accept_rate)
            kod_obs.append((current_block_size, iter_time_ms))

        stats.spec_lengths.append(current_block_size - 1)
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

    if original is not None:
        draft_model.num_draft_layers = original

    if stop_token_ids is not None:
        for i, tid in enumerate(output_ids_list[num_input_tokens:]):
            if tid in stop_token_ids:
                output_ids_list = output_ids_list[:num_input_tokens + i + 1]
                break

    output_ids = mx.array([output_ids_list], dtype=mx.int32)
    stats.total_tokens = len(output_ids_list) - num_input_tokens
    return output_ids, stats, target_cache, draft_cache, target_hidden


def _spec_generate_mirror_sd(
    target_model,
    draft_model,
    input_ids: mx.array,
    max_new_tokens: int,
    stop_token_ids: Optional[List[int]] = None,
    temperature: float = 0.0,
    target_layer_ids: Optional[List[int]] = None,
    failfast: bool = False,
    failfast_tau: float = 0.4,
    failfast_max_spec: int = 64,
) -> Tuple[mx.array, SpecDecodeStats]:
    """Mirror-SD speculative decoding with early-exit + parallel draft.

    Per Mirror-SD Eq. 10:
      1. Run target prefix layers (0..exit_layer) → get hidden states
      2. Start draft on ANE in parallel thread
      3. Run target suffix layers (exit_layer+1..N) → overlaps with ANE draft
      4. Wait for draft, accept/reject

    IMPORTANT: True parallelism requires the draft on a SEPARATE processor
    (ANE). Metal command buffers are NOT thread-safe, so GPU draft + GPU
    suffix in two threads crashes with:
      -[_MTLCommandBuffer addCompletedHandler:]: failed assertion

    When draft runs on GPU, we fall back to serial execution within the
    same thread: prefix → draft → suffix. Total time equals the non-split
    version, but the code path is validated for future ANE improvements.
    """
    from mlx_lm.models import cache as cache_module

    if target_layer_ids is None:
        target_layer_ids = draft_model.config.target_layer_ids

    from .target import _get_inner_model as _gim
    num_target_layers = len(_gim(target_model).layers)
    exit_layer = _pick_exit_layer(target_layer_ids, num_target_layers)

    is_ane = hasattr(draft_model, 'ane')
    can_parallel = is_ane  # Only ANE runs on a separate processor

    stats = SpecDecodeStats(parallel_mode=True)
    t_start = time.perf_counter()

    embed_fn = get_embed_tokens(target_model)
    lm_head_fn = get_lm_head(target_model)

    block_size = draft_model.block_size
    mask_token_id = draft_model.mask_token_id
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids_list = input_ids.tolist()[0] if input_ids.ndim == 2 else input_ids.tolist()
    if isinstance(output_ids_list, int):
        output_ids_list = [output_ids_list]

    target_cache = cache_module.make_prompt_cache(target_model)
    draft_cache = draft_model.make_cache()

    # --- Prefill ---
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

    # --- Decode with Mirror-SD early-exit ---
    start = num_input_tokens
    repeat_window = []
    max_repeat_window = 4

    draft_result = None
    draft_thread = None

    def _run_draft_gpu(th, ne, dc):
        nonlocal draft_result
        q_len = ne.shape[1]
        draft_hidden = draft_model(
            noise_embedding=ne,
            target_hidden=th,
            cache=dc,
        )
        draft_logits = lm_head_fn(draft_hidden[:, -(q_len - 1):, :])
        sampled_tokens = sample(draft_logits, temperature)
        mx.eval(sampled_tokens)
        draft_result = sampled_tokens

    def _run_draft_ane(th, ne, dc):
        nonlocal draft_result
        ctx_len = th.shape[1]
        gpu_fallback = getattr(draft_model, 'gpu_fallback', None)
        use_gpu = (gpu_fallback is not None and
                   ctx_len > getattr(draft_model, 'max_ctx_len', ctx_len))
        active_draft = gpu_fallback if use_gpu else draft_model
        if use_gpu:
            dc_active = active_draft.make_cache()
            for c_old, c_new in zip(dc, dc_active):
                if c_old.keys is not None:
                    c_new.keys = c_old.keys
                    c_new.values = c_old.values
                    c_old.offset = c_new.offset
        else:
            dc_active = dc
        q_len = ne.shape[1]
        draft_hidden = active_draft(
            noise_embedding=ne,
            target_hidden=th,
            cache=dc_active,
        )
        draft_logits = lm_head_fn(draft_hidden[:, -(q_len - 1):, :])
        sampled_tokens = sample(draft_logits, temperature)
        mx.eval(sampled_tokens)
        if use_gpu:
            for c_old, c_new in zip(dc, dc_active):
                c_old.keys = c_new.keys
                c_old.values = c_new.values
                c_old.offset = c_new.offset
        draft_result = sampled_tokens

    _run_draft_fn = _run_draft_ane if is_ane else _run_draft_gpu

    # --- First iteration: serial draft+verify to seed draft_result ---
    # We must run a normal serial iteration first because:
    # 1. The draft needs target_hidden from the CURRENT verify (not stale)
    # 2. Running draft between prefix/suffix on the first iter would pollute
    #    draft_cache (precomputed draft adds entries, then first-iter draft adds more)
    # After this, target_hidden is current and draft_cache is consistent.
    if start < max_length:
        block_tokens = [output_ids_list[-1]] + [mask_token_id] * (block_size - 1)
        noise_embedding = embed_fn(mx.array([block_tokens], dtype=mx.int32))

        q_len = noise_embedding.shape[1]
        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = lm_head_fn(draft_hidden[:, -(q_len - 1):, :])
        sampled_tokens = sample(draft_logits, temperature)
        mx.eval(sampled_tokens)

        block_tokens_updated = block_tokens.copy()
        for i in range(sampled_tokens.shape[1]):
            block_tokens_updated[i + 1] = int(sampled_tokens[0, i])
        block_output_ids = mx.array([block_tokens_updated], dtype=mx.int32)

        verify_logits, _, verify_hidden = forward_with_hidden_states(
            target_model, block_output_ids, cache=target_cache,
            capture_layers=target_layer_ids,
        )
        posterior = sample(verify_logits, temperature)
        mx.eval(posterior, *verify_hidden)
        mx.eval([c.state for c in target_cache])

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

        for c in draft_cache:
            c.crop(start)

        start += acceptance_length + 1

        for tid in draft_tokens[:acceptance_length] + [correction_token]:
            repeat_window.append(tid)
            if len(repeat_window) > max_repeat_window:
                repeat_window.pop(0)

        n_to_trim_target = block_size - acceptance_length - 1
        if n_to_trim_target > 0:
            cache_module.trim_prompt_cache(target_cache, n_to_trim_target)

        target_hidden = mx.concatenate([h[:, :acceptance_length + 1, :] for h in verify_hidden], axis=-1)

        stats.acceptance_lengths.append(acceptance_length + 1)
        stats.spec_lengths.append(block_size - 1)
        stats.accepted_tokens += acceptance_length
        stats.draft_steps += 1
        stats.total_tokens = len(output_ids_list) - num_input_tokens

    # Seed draft_result for the Mirror-SD loop (uses updated target_hidden
    # and cropped draft_cache from the serial first iteration)
    if start < max_length and draft_result is None:
        draft_block_tokens = [output_ids_list[-1]] + [mask_token_id] * (block_size - 1)
        noise_embedding_seed = embed_fn(
            mx.array([draft_block_tokens], dtype=mx.int32)
        )
        mx.eval(noise_embedding_seed)
        _run_draft_fn(target_hidden, noise_embedding_seed, draft_cache)

    # --- Subsequent iterations: Mirror-SD prefix/suffix + draft ---
    while start < max_length:
        remaining = max_length - start
        current_block_size = min(block_size, remaining + 1)

        if len(repeat_window) >= max_repeat_window:
            recent = output_ids_list[-max_repeat_window:]
            if len(set(recent)) == 1:
                repeat_window = []
                if draft_thread is not None:
                    draft_thread.join()
                    draft_thread = None
                token_id = mx.array([[output_ids_list[-1]]], dtype=mx.int32)
                logits = target_model(token_id, cache=target_cache)
                mx.eval(logits)
                mx.eval([c.state for c in target_cache])
                next_token = sample(logits[:, -1:, :], temperature)
                mx.eval(next_token)
                output_ids_list.append(int(next_token[0, 0]))
                start += 1
                stats.total_tokens = len(output_ids_list) - num_input_tokens
                continue

        # Wait for previous draft if still running
        if draft_thread is not None:
            draft_thread.join()
            draft_thread = None

        sampled_tokens = draft_result
        block_tokens = [output_ids_list[-1]] + [mask_token_id] * (current_block_size - 1)
        block_tokens_updated = block_tokens.copy()

        if sampled_tokens is not None:
            n_sampled = sampled_tokens.shape[1] if sampled_tokens.ndim > 1 else 1
            for i in range(min(n_sampled, current_block_size - 1)):
                block_tokens_updated[i + 1] = int(sampled_tokens[0, i])
        block_output_ids = mx.array([block_tokens_updated], dtype=mx.int32)

        # --- Verify: prefix (0..exit_layer) ---
        t_prefix_start = time.perf_counter()
        h, embed_verify, prefix_captured, attn_mask = forward_prefix(
            target_model,
            block_output_ids,
            cache=target_cache,
            exit_layer=exit_layer,
            capture_layers=target_layer_ids,
        )
        mx.eval(h, *prefix_captured)
        mx.eval([c.state for c in target_cache[:exit_layer + 1]])
        t_prefix_done = time.perf_counter()

        # --- Verify: suffix (exit_layer+1..N) ---
        # (On ANE, the draft would start here and overlap with suffix)
        t_suffix_start = time.perf_counter()
        verify_logits, suffix_captured = forward_suffix(
            target_model,
            h,
            cache=target_cache,
            start_layer=exit_layer + 1,
            mask=attn_mask,
            capture_layers=target_layer_ids,
        )
        posterior = sample(verify_logits, temperature)
        mx.eval(posterior, *suffix_captured)
        mx.eval([c.state for c in target_cache[exit_layer + 1:]])
        t_suffix_done = time.perf_counter()

        verify_hidden = prefix_captured + suffix_captured

        # --- Accept/reject ---
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

        for c in draft_cache:
            c.crop(start)

        start += acceptance_length + 1

        for tid in draft_tokens[:acceptance_length] + [correction_token]:
            repeat_window.append(tid)
            if len(repeat_window) > max_repeat_window:
                repeat_window.pop(0)

        n_to_trim_target = current_block_size - acceptance_length - 1
        if n_to_trim_target > 0:
            cache_module.trim_prompt_cache(target_cache, n_to_trim_target)

        target_hidden = mx.concatenate([h[:, :acceptance_length + 1, :] for h in verify_hidden], axis=-1)

        # --- Draft for next iteration ---
        # Runs AFTER accept/reject + crop + target_hidden update so it uses
        # CURRENT target_hidden and CROPPED draft_cache — same state as serial.
        # CRITICAL: noise_embedding must use MASK tokens (DFlash denoises masks).
        t_draft_start = time.perf_counter()

        if failfast:
            all_sampled = []
            total_spec_len = 0
            prev_accept = stats.acceptance_lengths[-1] if stats.acceptance_lengths else 0
            do_extend = prev_accept >= failfast_tau

            for ext in range(failfast_max_spec // block_size + 1):
                ext_bs = min(block_size, max_length - start - total_spec_len + 1)
                if ext_bs <= 1:
                    break

                if ext > 0 and not do_extend:
                    break

                if ext == 0:
                    ext_tokens = [output_ids_list[-1]] + [mask_token_id] * (ext_bs - 1)
                else:
                    ext_tokens = [all_sampled[-1]] + [mask_token_id] * (ext_bs - 1)

                ne = embed_fn(mx.array([ext_tokens], dtype=mx.int32))
                mx.eval(ne)

                ql = ne.shape[1]

                dh = draft_model(noise_embedding=ne, target_hidden=target_hidden, cache=draft_cache)
                dl = lm_head_fn(dh[:, -(ql - 1):, :])
                st = sample(dl, temperature)
                mx.eval(st)

                new_tokens = st[0].tolist()
                all_sampled.extend(new_tokens)
                total_spec_len += len(new_tokens)

                if total_spec_len >= failfast_max_spec:
                    break

            draft_result = mx.array([all_sampled], dtype=mx.int32)
            next_spec_len = total_spec_len
        else:
            draft_block_tokens = [output_ids_list[-1]] + [mask_token_id] * (current_block_size - 1)
            noise_embedding = embed_fn(
                mx.array([draft_block_tokens], dtype=mx.int32)
            )
            mx.eval(noise_embedding)

            if can_parallel:
                draft_thread = threading.Thread(
                    target=_run_draft_fn,
                    args=(target_hidden, noise_embedding, draft_cache),
                )
                draft_thread.start()
            else:
                _run_draft_fn(target_hidden, noise_embedding, draft_cache)

            next_spec_len = current_block_size - 1

        prefix_time = t_prefix_done - t_prefix_start
        suffix_time = t_suffix_done - t_suffix_start
        stats.total_verify_time += prefix_time + suffix_time
        stats.total_draft_time += time.perf_counter() - t_draft_start
        stats.total_overlap_time += suffix_time if can_parallel else 0

        stats.acceptance_lengths.append(acceptance_length + 1)
        stats.spec_lengths.append(next_spec_len)
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

    if draft_thread is not None:
        draft_thread.join()

    stats.total_time = time.perf_counter() - t_start

    if stop_token_ids is not None:
        for i, tid in enumerate(output_ids_list[num_input_tokens:]):
            if tid in stop_token_ids:
                output_ids_list = output_ids_list[:num_input_tokens + i + 1]
                break

    output_ids = mx.array([output_ids_list], dtype=mx.int32)
    stats.total_tokens = len(output_ids_list) - num_input_tokens
    return output_ids, stats


def _pick_exit_layer(target_layer_ids: List[int], num_target_layers: int) -> int:
    """Pick the exit layer for Mirror-SD early-exit.

    Chooses the median target_layer_id as the exit point, balancing
    prefix and suffix compute. This ensures roughly equal numbers of
    capture layers on each side for feature extraction.
    """
    mid = len(target_layer_ids) // 2
    return target_layer_ids[mid]


def _spec_generate_parallel(
    target_model,
    draft_model,
    input_ids: mx.array,
    max_new_tokens: int,
    stop_token_ids: Optional[List[int]] = None,
    temperature: float = 0.0,
    target_layer_ids: Optional[List[int]] = None,
) -> Tuple[mx.array, SpecDecodeStats]:
    """Parallel ANE||GPU speculative decoding (Mirror-SD Eq. 10).

    Pipelines draft and verify so that the next iteration's draft on ANE
    overlaps with the current iteration's verify on GPU:

      Iteration N:   [draft_N on ANE] → [verify_N on GPU]
      Iteration N+1:                  [draft_N+1 on ANE] → [verify_N+1 on GPU]
                                            ↑ starts while verify_N is still running

    If draft_N+1 finishes within verify_N's time, it's effectively free.
    """
    from mlx_lm.models import cache as cache_module

    if target_layer_ids is None:
        target_layer_ids = draft_model.config.target_layer_ids

    stats = SpecDecodeStats(parallel_mode=True)
    t_start = time.perf_counter()

    embed_fn = get_embed_tokens(target_model)
    lm_head_fn = get_lm_head(target_model)

    block_size = draft_model.block_size
    mask_token_id = draft_model.mask_token_id
    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids_list = input_ids.tolist()[0] if input_ids.ndim == 2 else input_ids.tolist()
    if isinstance(output_ids_list, int):
        output_ids_list = [output_ids_list]

    target_cache = cache_module.make_prompt_cache(target_model)
    draft_cache = draft_model.make_cache()

    # --- Prefill ---
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

    # --- Decode with pipelined ANE||GPU ---
    start = num_input_tokens
    repeat_window = []
    max_repeat_window = 4

    # Precompute the first draft while nothing is running on GPU
    draft_result = None
    draft_thread = None

    gpu_fallback = getattr(draft_model, 'gpu_fallback', None)

    def _run_draft(th, ne, dc, rope_offset):
        nonlocal draft_result
        ctx_len = th.shape[1]
        use_gpu = (gpu_fallback is not None and
                   ctx_len > getattr(draft_model, 'max_ctx_len', ctx_len))
        active_draft = gpu_fallback if use_gpu else draft_model
        if use_gpu:
            dc_active = active_draft.make_cache()
            for c_old, c_new in zip(dc, dc_active):
                if c_old.keys is not None:
                    c_new.keys = c_old.keys
                    c_new.values = c_old.values
                    c_new.offset = c_old.offset
        else:
            dc_active = dc
        q_len = ne.shape[1]
        draft_hidden = active_draft(
            noise_embedding=ne,
            target_hidden=th,
            cache=dc_active,
        )
        draft_logits = lm_head_fn(draft_hidden[:, -(q_len - 1):, :])
        sampled_tokens = sample(draft_logits, temperature)
        mx.eval(sampled_tokens)
        if use_gpu:
            for c_old, c_new in zip(dc, dc_active):
                c_old.keys = c_new.keys
                c_old.values = c_new.values
                c_old.offset = c_new.offset
        draft_result = sampled_tokens

    # Kick off first draft
    block_tokens = [output_ids_list[-1]] + [mask_token_id] * (block_size - 1)
    noise_embedding = embed_fn(mx.array([block_tokens], dtype=mx.int32))
    mx.eval(noise_embedding)
    t_draft_start = time.perf_counter()
    draft_thread = threading.Thread(target=_run_draft, args=(target_hidden, noise_embedding, draft_cache, 0))
    draft_thread.start()

    while start < max_length:
        remaining = max_length - start
        current_block_size = min(block_size, remaining + 1)

        if len(repeat_window) >= max_repeat_window:
            recent = output_ids_list[-max_repeat_window:]
            if len(set(recent)) == 1:
                repeat_window = []
                if draft_thread is not None:
                    draft_thread.join()
                    draft_thread = None
                token_id = mx.array([[output_ids_list[-1]]], dtype=mx.int32)
                logits = target_model(token_id, cache=target_cache)
                mx.eval(logits)
                mx.eval([c.state for c in target_cache])
                next_token = sample(logits[:, -1:, :], temperature)
                mx.eval(next_token)
                output_ids_list.append(int(next_token[0, 0]))
                start += 1
                stats.total_tokens = len(output_ids_list) - num_input_tokens
                continue

        # --- Wait for draft (may have overlapped with previous verify) ---
        t_draft_wait = time.perf_counter()
        if draft_thread is not None:
            draft_thread.join()
            draft_thread = None
        t_draft_done = time.perf_counter()
        draft_time = t_draft_done - t_draft_start

        sampled_tokens = draft_result
        block_tokens = [output_ids_list[-1]] + [mask_token_id] * (current_block_size - 1)
        block_tokens_updated = block_tokens.copy()
        n_sampled = sampled_tokens.shape[1] if sampled_tokens.ndim > 1 else 1
        for i in range(min(n_sampled, current_block_size - 1)):
            block_tokens_updated[i + 1] = int(sampled_tokens[0, i])
        block_output_ids = mx.array([block_tokens_updated], dtype=mx.int32)

        # --- Verify on GPU + start next draft on ANE in parallel ---
        t_verify_start = time.perf_counter()

        verify_logits, _, verify_hidden = forward_with_hidden_states(
            target_model,
            block_output_ids,
            cache=target_cache,
            capture_layers=target_layer_ids,
        )
        posterior = sample(verify_logits, temperature)
        mx.eval(posterior, *verify_hidden)
        mx.eval([c.state for c in target_cache])

        # --- Accept/reject ---
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

        for c in draft_cache:
            c.crop(start)

        start += acceptance_length + 1

        for tid in draft_tokens[:acceptance_length] + [correction_token]:
            repeat_window.append(tid)
            if len(repeat_window) > max_repeat_window:
                repeat_window.pop(0)

        n_to_trim_target = current_block_size - acceptance_length - 1
        if n_to_trim_target > 0:
            cache_module.trim_prompt_cache(target_cache, n_to_trim_target)

        new_target_hidden = mx.concatenate([h[:, :acceptance_length + 1, :] for h in verify_hidden], axis=-1)

        t_verify_done = time.perf_counter()
        verify_time = t_verify_done - t_verify_start
        overlap = max(0, draft_time - verify_time) if draft_time > 0 else 0
        stats.total_draft_time += draft_time
        stats.total_verify_time += verify_time
        stats.total_overlap_time += min(draft_time, verify_time)

        stats.acceptance_lengths.append(acceptance_length + 1)
        stats.accepted_tokens += acceptance_length
        stats.draft_steps += 1
        stats.total_tokens = len(output_ids_list) - num_input_tokens

        if stop_token_ids is not None:
            for stop_id in stop_token_ids:
                if stop_id in output_ids_list[num_input_tokens:]:
                    break
            else:
                # Start next draft in parallel with upcoming verify
                if start < max_length:
                    next_block_tokens = [output_ids_list[-1]] + [mask_token_id] * (block_size - 1)
                    next_noise_embedding = embed_fn(
                        mx.array([next_block_tokens], dtype=mx.int32)
                    )
                    mx.eval(next_noise_embedding)
                    t_draft_start = time.perf_counter()
                    draft_thread = threading.Thread(
                        target=_run_draft,
                        args=(new_target_hidden, next_noise_embedding, draft_cache, start),
                    )
                    draft_thread.start()
                target_hidden = new_target_hidden
                continue
            break

        # Start next draft in parallel with upcoming verify
        if start < max_length:
            next_block_tokens = [output_ids_list[-1]] + [mask_token_id] * (block_size - 1)
            next_noise_embedding = embed_fn(
                mx.array([next_block_tokens], dtype=mx.int32)
            )
            mx.eval(next_noise_embedding)
            t_draft_start = time.perf_counter()
            draft_thread = threading.Thread(
                target=_run_draft,
                args=(new_target_hidden, next_noise_embedding, draft_cache, start),
            )
            draft_thread.start()
        target_hidden = new_target_hidden

    # Clean up any remaining draft thread
    if draft_thread is not None:
        draft_thread.join()

    stats.total_time = time.perf_counter() - t_start

    if stop_token_ids is not None:
        for i, tid in enumerate(output_ids_list[num_input_tokens:]):
            if tid in stop_token_ids:
                output_ids_list = output_ids_list[:num_input_tokens + i + 1]
                break

    output_ids = mx.array([output_ids_list], dtype=mx.int32)
    stats.total_tokens = len(output_ids_list) - num_input_tokens
    return output_ids, stats
