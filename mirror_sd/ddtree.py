"""DDTree: Diffusion Draft Tree for block-diffusion speculative decoding.

Implements the DDTree algorithm from "Accelerating Speculative Decoding with
Block Diffusion Draft Trees" (reference: references/ddtree/).

Instead of verifying a single linear draft path (vanilla DFlash), DDTree
constructs a draft tree from the per-position marginal distributions produced
by the DFlash draft model. The tree is verified in a single target forward
pass using tree attention (ancestor-only mask), increasing the expected number
of accepted tokens per verification round.

Algorithm overview:
  1. Draft: Run DFlash to get logits at each draft position
  2. Tree build: Use best-first heap (Algorithm 1) to select top-B most
     probable token prefixes forming a tree
  3. Tree compile: Flatten tree into a batch with position IDs and
     ancestor-only attention mask
  4. Verify: Single target forward pass over the flattened tree
  5. Commit: Walk the verified tree, accept the longest matching path,
     compact KV caches

NOTE: Currently only works with pure full-attention target models (no SSM/
linear attention layers). For Qwen3.5 targets, the spec_generate entry point
falls back to vanilla DFlash because SSM layers cannot use tree attention
masks — they process tokens sequentially and their recurrent state would be
corrupted by out-of-order tree branches.
"""

import heapq
import time
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import KVCache, ArraysCache

from .dflash import extract_context_feature, sample
from .target import (
    _get_inner_model,
    forward_with_hidden_states,
    get_embed_tokens,
    get_lm_head,
    is_qwen35,
)


@dataclass
class DDTreeResult:
    node_token_ids: mx.array
    node_depths: mx.array
    parents: list
    child_maps: list
    visibility: mx.array


def _compute_greedy_path(tree: DDTreeResult) -> list:
    """Compute the greedy (top-1) path through the tree.

    Starting from root, follow the first child at each depth (which is the
    top-1 token from the heap — the most probable prefix).

    Returns list of node indices (0-indexed into the verify input sequence).
    """
    path = [0]
    current = 0
    while True:
        children = tree.child_maps[current]
        if not children:
            break
        first_child_token = next(iter(children))
        next_node = children[first_child_token]
        path.append(next_node)
        current = next_node
    return path


def build_ddtree_tree(
    draft_logits: mx.array,
    budget: int,
) -> Optional[DDTreeResult]:
    """Build a DDTree from DFlash draft logits using Algorithm 1 (best-first heap).

    Args:
        draft_logits: [draft_horizon, vocab_size] logits from DFlash draft model.
        budget: Maximum number of tree nodes (excluding root).

    Returns:
        DDTreeResult with the constructed tree, or None if budget <= 0.
    """
    if budget <= 0 or draft_logits.shape[0] == 0:
        return None

    draft_horizon = draft_logits.shape[0]
    topk = min(budget, draft_logits.shape[-1])

    logits_f = draft_logits.astype(mx.float32)
    log_z = mx.logsumexp(logits_f, axis=-1, keepdims=True)
    log_probs = logits_f - log_z

    top_log_probs_mx = mx.topk(log_probs, k=topk, axis=-1)
    top_token_ids_mx = mx.argsort(-log_probs, axis=-1)[:, :topk]

    top_log_probs = np.array(top_log_probs_mx)
    top_token_ids = np.array(top_token_ids_mx)

    first_logw = float(top_log_probs[0, 0])
    heap: list = [(-first_logw, (0,), 0, 1, 0, first_logw)]

    node_token_ids_list = []
    node_depths_list = []
    parents = [-1]
    child_maps: list = [dict()]
    node_count = 0

    while heap and node_count < budget:
        _, ranks, parent_index, depth, rank, logw = heapq.heappop(heap)

        token_id = int(top_token_ids[depth - 1, rank])
        current_index = node_count + 1
        node_token_ids_list.append(token_id)
        node_depths_list.append(depth)
        parents.append(parent_index)
        child_maps.append(dict())
        child_maps[parent_index][token_id] = current_index
        node_count += 1

        if rank + 1 < topk:
            sibling_logw = logw - float(top_log_probs[depth - 1, rank]) + float(top_log_probs[depth - 1, rank + 1])
            sibling_ranks = ranks[:-1] + (rank + 1,)
            heapq.heappush(heap, (-sibling_logw, sibling_ranks, parent_index, depth, rank + 1, sibling_logw))

        if depth < draft_horizon:
            child_logw = logw + float(top_log_probs[depth, 0])
            child_ranks = ranks + (0,)
            heapq.heappush(heap, (-child_logw, child_ranks, current_index, depth + 1, 0, child_logw))

    current_length = 1 + node_count

    visibility_np = np.zeros((current_length, current_length), dtype=np.bool_)
    visibility_np[0, 0] = True
    for index in range(1, current_length):
        parent_index = parents[index]
        visibility_np[index, :index] = visibility_np[parent_index, :index]
        visibility_np[index, index] = True

    return DDTreeResult(
        node_token_ids=mx.array(node_token_ids_list, dtype=mx.int32),
        node_depths=mx.array(node_depths_list, dtype=mx.int32),
        parents=parents,
        child_maps=child_maps,
        visibility=mx.array(visibility_np, dtype=mx.bool_),
    )


def compile_ddtree_tree(
    root_token_id: int,
    start: int,
    tree: DDTreeResult,
    past_length: int,
    dtype: mx.Dtype,
) -> Tuple[mx.array, mx.array, mx.array]:
    """Compile DDTree into flat verify inputs with tree attention mask.

    Returns:
        verify_input_ids: [1, N] token IDs (root + tree nodes)
        verify_position_ids: [1, N] position IDs
        attention_mask: [1, 1, N, past_length + N] tree attention mask
    """
    node_count = len(tree.node_token_ids)
    current_length = 1 + node_count

    input_ids = [root_token_id] + tree.node_token_ids.tolist()
    verify_input_ids = mx.array([input_ids], dtype=mx.int32)

    positions = [start] + [start + int(d) for d in tree.node_depths.tolist()]
    verify_position_ids = mx.array([positions], dtype=mx.int32)

    mask_val = mx.finfo(dtype).min

    past_block = mx.zeros((1, 1, current_length, past_length), dtype=dtype)
    vis_4d = tree.visibility[:current_length, :current_length].reshape(1, 1, current_length, current_length)
    tree_block = mx.where(vis_4d, mx.array(0, dtype=dtype), mask_val)

    attn_mask = mx.concatenate([past_block, tree_block], axis=3)

    # SDPA requires mask dtype to match query dtype, not input_ids dtype
    # We'll cast at the call site, but also ensure the mask is in the right type here

    return verify_input_ids, verify_position_ids, attn_mask


def follow_verified_tree(
    child_maps: list,
    posterior: mx.array,
) -> Tuple[list, int]:
    """Walk the verified tree following the target model's posterior tokens.

    Starting at the root, at each node check if the target's posterior token
    at that node matches a child edge. If so, descend; otherwise stop.

    Returns:
        accepted_indices: list of node indices accepted (starting with root=0)
        next_token: the correction token (target's posterior at the rejection point)
    """
    posterior_tokens = posterior[0].tolist()
    accepted_indices = [0]
    current_index = 0
    next_token = int(posterior_tokens[current_index])

    while next_token in child_maps[current_index]:
        current_index = child_maps[current_index][next_token]
        accepted_indices.append(current_index)
        next_token = int(posterior_tokens[current_index])

    return accepted_indices, next_token


def compact_kv_cache(cache: list, past_length: int, keep_indices: list):
    """Compact target model KV cache after DDTree partial acceptance.

    The verify pass appended tree tokens starting at `past_length`.
    We keep only the entries at `keep_indices` (relative to the appended block)
    and discard the rest.
    """
    n_keep = len(keep_indices)
    if n_keep == 0:
        for c in cache:
            if isinstance(c, KVCache) and c.keys is not None:
                c.keys = c.keys[..., :past_length, :]
                c.values = c.values[..., :past_length, :]
                c.offset = past_length
        return

    keep_idx = mx.array(keep_indices, dtype=mx.int32)

    for c in cache:
        if isinstance(c, KVCache) and c.keys is not None:
            past_k = c.keys[..., :past_length, :]
            past_v = c.values[..., :past_length, :]
            appended_k = c.keys[..., past_length:, :]
            appended_v = c.values[..., past_length:, :]

            kept_k = mx.take(appended_k, keep_idx, axis=2)
            kept_v = mx.take(appended_v, keep_idx, axis=2)

            c.keys = mx.concatenate([past_k, kept_k], axis=2)
            c.values = mx.concatenate([past_v, kept_v], axis=2)
            c.offset = past_length + n_keep


def _flat_cache_states(cache):
    out = []
    for c in cache:
        s = c.state
        if isinstance(s, list):
            out.extend(s)
        else:
            out.append(s)
    return out


def _get_attn_dims(attn):
    n_heads = getattr(attn, 'num_attention_heads', None) or getattr(attn, 'n_heads', None)
    n_kv_heads = getattr(attn, 'num_key_value_heads', None) or getattr(attn, 'n_kv_heads', None)
    head_dim = getattr(attn, 'head_dim', None)
    if head_dim is None:
        head_dim = attn.q_proj.weight.shape[0] // n_heads
    return n_heads, n_kv_heads, head_dim


def _apply_rope_with_positions(attn, x, position_ids):
    """Apply RoPE using explicit position IDs for tree attention.

    Uses batched RoPE by reshaping [B, n_heads, L, D] to [L, n_heads, 1, D]
    and passing position offsets as a vector.

    x: [B, n_heads, L, head_dim]
    position_ids: [1, L] absolute position for each token
    """
    B, n_heads, L, head_dim_val = x.shape
    pos_offsets = position_ids[0].astype(mx.int32)

    x_transposed = x.transpose(2, 0, 1, 3).reshape(L, n_heads, 1, head_dim_val)

    x_roped = mx.fast.rope(
        x_transposed,
        head_dim_val,
        traditional=attn.rope.traditional,
        base=attn.rope.base,
        scale=attn.rope.scale,
        offset=pos_offsets,
    )

    return x_roped.reshape(L, B, n_heads, head_dim_val).transpose(1, 2, 0, 3)


def _forward_full_attn_with_tree_mask(layer, hidden_states, attention_mask, cache, position_ids):
    """Forward a full-attention layer with tree attention mask and custom position IDs."""
    attn = layer.self_attn
    n_heads, n_kv_heads, head_dim = _get_attn_dims(attn)
    has_qk_norm = hasattr(attn, 'q_norm') and hasattr(attn, 'k_norm')
    q_out_dim = attn.q_proj.weight.shape[0]
    has_gate = (q_out_dim != n_heads * head_dim)

    residual = hidden_states
    inputs = layer.input_layernorm(hidden_states)
    B, L, _ = inputs.shape

    q_proj_out = attn.q_proj(inputs)
    if has_gate:
        queries, gate = mx.split(
            q_proj_out.reshape(B, L, n_heads, -1), 2, axis=-1,
        )
        gate = gate.reshape(B, L, -1)
    else:
        queries = q_proj_out.reshape(B, L, n_heads, -1)

    new_keys = attn.k_proj(inputs)
    new_values = attn.v_proj(inputs)

    if has_qk_norm:
        queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
        new_keys = attn.k_norm(new_keys.reshape(B, L, n_kv_heads, -1)).transpose(0, 2, 1, 3)
    else:
        queries = queries.transpose(0, 2, 1, 3)
        new_keys = new_keys.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
    new_values = new_values.reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

    queries = _apply_rope_with_positions(attn, queries, position_ids)
    new_keys = _apply_rope_with_positions(attn, new_keys, position_ids)

    if cache is not None and isinstance(cache, KVCache):
        k_full, v_full = cache.update_and_fetch(new_keys, new_values)
    else:
        k_full = new_keys
        v_full = new_values

    n_rep = n_heads // n_kv_heads
    if n_rep > 1:
        k_full = mx.repeat(k_full, n_rep, axis=1)
        v_full = mx.repeat(v_full, n_rep, axis=1)

    output = mx.fast.scaled_dot_product_attention(
        queries, k_full, v_full, scale=attn.scale, mask=attention_mask,
    )
    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

    if has_gate:
        output = attn.o_proj(output * mx.sigmoid(gate))
    else:
        output = attn.o_proj(output)

    hidden_states = residual + output
    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)

    return hidden_states


def _forward_target_with_tree_mask(
    model,
    input_ids: mx.array,
    position_ids: mx.array,
    attention_mask: mx.array,
    cache=None,
    capture_layers: Optional[List[int]] = None,
):
    """Forward pass through target model with tree attention mask.

    For pure full-attention models, all layers use the tree mask.
    This is NOT compatible with models that have SSM/linear attention layers
    (e.g. Qwen3.5) because SSM layers process tokens sequentially and their
    recurrent state would be corrupted by out-of-order tree branches.
    """
    if capture_layers is None:
        capture_layers = []

    capture_set = set(capture_layers)
    inner = _get_inner_model(model)
    h = inner.embed_tokens(input_ids)

    if cache is None:
        from mlx_lm.models import cache as cache_module
        cache = cache_module.make_prompt_cache(model)

    captured = {}

    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        if getattr(layer, 'is_linear', False):
            raise NotImplementedError(
                "DDTree tree attention is not compatible with SSM/linear attention layers. "
                "Use a pure full-attention target model."
            )
        h = _forward_full_attn_with_tree_mask(layer, h, attention_mask, c, position_ids)
        if i in capture_set:
            captured[i] = h

    h = inner.norm(h)

    if hasattr(model, 'lm_head') and model.lm_head is not None:
        logits = model.lm_head(h)
    elif hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        logits = model.language_model.lm_head(h)
    else:
        logits = inner.embed_tokens.as_linear(h)

    hidden_states = [captured[i] for i in capture_layers]
    embed = inner.embed_tokens(input_ids)

    return logits, embed, hidden_states


def ddtree_generate(
    target_model,
    draft_model,
    input_ids: mx.array,
    max_new_tokens: int,
    stop_token_ids: Optional[List[int]] = None,
    temperature: float = 0.0,
    target_layer_ids: Optional[List[int]] = None,
    ddtree_budget: int = 0,
    stream_callback=None,
    prefill_step_size: int = 512,
) -> Tuple[mx.array, 'SpecDecodeStats', list, list, mx.array]:
    """DDTree speculative decoding: build draft tree, verify with tree attention.

    Currently only works with pure full-attention target models (no SSM layers).
    For Qwen3.5 targets, spec_generate falls back to vanilla DFlash.

    Args:
        target_model: The target language model (must be pure full-attention)
        draft_model: DFlashDraftModel for drafting
        input_ids: [1, seq_len] input token IDs
        max_new_tokens: Maximum tokens to generate
        stop_token_ids: Token IDs that stop generation
        temperature: Sampling temperature
        target_layer_ids: Target model layers to capture hidden states from
        ddtree_budget: Max tree nodes (excluding root). 0 = use draft_horizon.
        stream_callback: Called for each generated token
        prefill_step_size: Chunk size for prefilling

    Returns:
        (output_ids, stats, target_cache, draft_cache, target_hidden)
    """
    from .generate import SpecDecodeStats
    from mlx_lm.models import cache as cache_module

    if target_layer_ids is None:
        target_layer_ids = draft_model.config.target_layer_ids

    block_size = draft_model.block_size
    mask_token_id = draft_model.mask_token_id
    draft_horizon = block_size - 1
    if ddtree_budget <= 0:
        ddtree_budget = draft_horizon

    embed_fn = get_embed_tokens(target_model)
    lm_head_fn = get_lm_head(target_model)

    stats = SpecDecodeStats()
    t_start = time.perf_counter()

    num_input_tokens = input_ids.shape[1]
    max_length = num_input_tokens + max_new_tokens

    output_ids_list = input_ids.tolist()[0] if input_ids.ndim == 2 else input_ids.tolist()
    if isinstance(output_ids_list, int):
        output_ids_list = [output_ids_list]

    target_cache = cache_module.make_prompt_cache(target_model)
    draft_cache = draft_model.make_cache()

    if num_input_tokens > prefill_step_size:
        for start in range(0, num_input_tokens - prefill_step_size, prefill_step_size):
            chunk = input_ids[:, start:start + prefill_step_size]
            logits_chunk, _, _ = forward_with_hidden_states(
                target_model, chunk, cache=target_cache, capture_layers=[],
            )
            mx.eval(logits_chunk, _flat_cache_states(target_cache))
            del logits_chunk
        last_start = (num_input_tokens // prefill_step_size) * prefill_step_size
        if last_start >= num_input_tokens:
            last_start = max(0, num_input_tokens - prefill_step_size)
        last_chunk = input_ids[:, last_start:]
        logits, embed, hidden_states = forward_with_hidden_states(
            target_model, last_chunk, cache=target_cache,
            capture_layers=target_layer_ids,
        )
        first_token = sample(logits[:, -1:, :], temperature)
        mx.eval(logits, embed, *hidden_states, first_token, _flat_cache_states(target_cache))
        target_hidden = extract_context_feature(hidden_states, target_layer_ids)
    else:
        logits, embed, hidden_states = forward_with_hidden_states(
            target_model, input_ids, cache=target_cache,
            capture_layers=target_layer_ids,
        )
        first_token = sample(logits[:, -1:, :], temperature)
        mx.eval(logits, embed, *hidden_states, first_token, _flat_cache_states(target_cache))
        target_hidden = extract_context_feature(hidden_states, target_layer_ids)

    first_tok = int(first_token[0, 0])
    output_ids_list.append(first_tok)
    if stream_callback is not None:
        stream_callback(first_tok)

    stats.prefill_time = time.perf_counter() - t_start

    start = num_input_tokens
    while start < max_length:
        remaining = max_length - start
        if remaining < 2:
            break

        t_draft_start = time.perf_counter()

        anchor_token = output_ids_list[-1]
        current_block_size = min(block_size, remaining + 1)
        draft_horizon_actual = current_block_size - 1

        block_tokens = [anchor_token] + [mask_token_id] * draft_horizon_actual
        block_ids = mx.array([block_tokens], dtype=mx.int32)
        noise_embedding = embed_fn(block_ids)

        draft_hidden = draft_model(
            noise_embedding=noise_embedding,
            target_hidden=target_hidden,
            cache=draft_cache,
        )
        draft_logits = lm_head_fn(draft_hidden[:, -draft_horizon_actual:, :])

        for c in draft_cache:
            c.trim(current_block_size)

        stats.total_draft_time += time.perf_counter() - t_draft_start

        tree = build_ddtree_tree(draft_logits[0], ddtree_budget)
        if tree is None:
            break

        node_count = len(tree.node_token_ids)
        current_length = 1 + node_count

        past_length = start
        inner = _get_inner_model(target_model)
        dtype = inner.layers[0].self_attn.q_proj.weight.dtype
        verify_input_ids, verify_position_ids, verify_attention_mask = compile_ddtree_tree(
            root_token_id=anchor_token,
            start=start,
            tree=tree,
            past_length=past_length,
            dtype=dtype,
        )

        t_verify_start = time.perf_counter()
        verify_logits, _, verify_hidden = _forward_target_with_tree_mask(
            target_model, verify_input_ids, verify_position_ids, verify_attention_mask,
            cache=target_cache, capture_layers=target_layer_ids,
        )

        posterior = sample(verify_logits, temperature)
        mx.eval(posterior, *verify_hidden, _flat_cache_states(target_cache))

        stats.total_verify_time += time.perf_counter() - t_verify_start

        t_commit_start = time.perf_counter()
        accepted_indices, next_token = follow_verified_tree(tree.child_maps, posterior)

        accepted_count = len(accepted_indices)
        correction_token = next_token

        accepted_tokens = []
        for idx in accepted_indices:
            if idx == 0:
                accepted_tokens.append(anchor_token)
            else:
                accepted_tokens.append(int(tree.node_token_ids[idx - 1]))

        for t in accepted_tokens[1:]:
            output_ids_list.append(t)
            if stream_callback is not None:
                stream_callback(t)
        output_ids_list.append(correction_token)
        if stream_callback is not None:
            stream_callback(correction_token)

        start += accepted_count

        compact_kv_cache(target_cache, past_length, accepted_indices)

        target_hidden = mx.concatenate([h[:, :accepted_count, :] for h in verify_hidden], axis=-1)

        stats.total_rollback_time += time.perf_counter() - t_commit_start

        stats.spec_lengths.append(draft_horizon_actual)
        stats.acceptance_lengths.append(accepted_count)
        stats.accepted_tokens += accepted_count - 1
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
    gen_time = stats.total_time - stats.prefill_time
    stats.total_misc_time = max(0, gen_time - stats.total_draft_time - stats.total_verify_time - stats.total_rollback_time)

    if stop_token_ids is not None:
        for i, tid in enumerate(output_ids_list[num_input_tokens:]):
            if tid in stop_token_ids:
                output_ids_list = output_ids_list[:num_input_tokens + i + 1]
                break

    output_ids = mx.array([output_ids_list], dtype=mx.int32)
    stats.total_tokens = len(output_ids_list) - num_input_tokens
    return output_ids, stats, target_cache, draft_cache, target_hidden
