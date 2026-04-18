"""Target model utilities for Mirror-SD.

Provides forward_with_hidden_states() which runs a target model while
capturing intermediate hidden states at specified layer indices. These
hidden states are the key input to the DFlash draft model.

Provides forward_split() for Mirror-SD early-exit: runs prefix layers,
emits hidden states for the draft model, then continues suffix layers.
This enables parallel ANE draft || GPU suffix execution (Eq. 10).

For Qwen3.5 models (mixed full + linear attention), provides:
- Proper fa_mask / ssm_mask handling
- Linear attention rollback on partial block rejection
"""

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import KVCache


def _get_inner_model(model):
    inner = getattr(model, 'model', None) or getattr(model, 'language_model', None)
    if inner is not None and not hasattr(inner, 'embed_tokens'):
        inner = getattr(inner, 'model', inner)
    return inner


def get_embed_tokens(model):
    return _get_inner_model(model).embed_tokens


def get_lm_head(model):
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        return model.language_model.lm_head
    return _get_inner_model(model).embed_tokens.as_linear


def is_qwen35(model):
    """Detect if model has linear attention layers (Qwen3.5 architecture)."""
    inner = _get_inner_model(model)
    if not hasattr(inner, 'layers') or len(inner.layers) == 0:
        return False
    return any(getattr(l, 'is_linear', False) for l in inner.layers)


def _find_fa_mask_cache_idx(inner):
    """Find the index of the first KVCache entry for fa_mask creation."""
    if hasattr(inner, 'fa_idx'):
        return inner.fa_idx
    for i, layer in enumerate(inner.layers):
        if not getattr(layer, 'is_linear', False):
            return i
    return 0


def _make_masks(h, cache, inner):
    """Create proper fa_mask and ssm_mask for Qwen3.5 models."""
    fa_idx = _find_fa_mask_cache_idx(inner)
    fa_mask = create_attention_mask(h, cache[fa_idx])
    ssm_idx = getattr(inner, 'ssm_idx', 0)
    if cache[ssm_idx] is not None and hasattr(cache[ssm_idx], 'make_mask'):
        ssm_mask = cache[ssm_idx].make_mask(h.shape[1])
    else:
        ssm_mask = None
    return fa_mask, ssm_mask


def forward_with_hidden_states(
    model,
    inputs: mx.array,
    cache=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array]]:
    if capture_layers is None:
        capture_layers = []

    capture_set = set(capture_layers)
    inner = _get_inner_model(model)
    h = inner.embed_tokens(inputs)
    embed = h

    if cache is None:
        from mlx_lm.models import cache as cache_module
        cache = cache_module.make_prompt_cache(model)

    if is_qwen35(model):
        fa_mask, ssm_mask = _make_masks(h, cache, inner)
    else:
        try:
            fa_mask = create_attention_mask(h, cache[0])
        except TypeError:
            fa_mask = cache[0].make_mask(h.shape[1])
        ssm_mask = None

    captured = {}
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        if getattr(layer, 'is_linear', False):
            h = layer(h, ssm_mask, cache=c)
        else:
            h = layer(h, fa_mask, cache=c)
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

    return logits, embed, hidden_states


def forward_with_hidden_states_and_rollback(
    model,
    inputs: mx.array,
    cache=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array], Dict[int, Dict[str, mx.array]]]:
    """Forward pass for Qwen3.5 that records rollback data for linear attention.

    Returns same as forward_with_hidden_states plus rollback_records dict
    mapping linear layer index to {initial_conv_state, initial_ssm_state,
    qkv, k, g, tape}.
    """
    if capture_layers is None:
        capture_layers = []

    capture_set = set(capture_layers)
    inner = _get_inner_model(model)
    h = inner.embed_tokens(inputs)
    embed = h

    if cache is None:
        from mlx_lm.models import cache as cache_module
        cache = cache_module.make_prompt_cache(model)

    fa_mask, ssm_mask = _make_masks(h, cache, inner)

    captured = {}
    rollback_records = {}

    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        if getattr(layer, 'is_linear', False):
            h, record = _forward_linear_layer_with_record(layer, h, ssm_mask, c)
            rollback_records[i] = record
        else:
            h = layer(h, fa_mask, cache=c)
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

    return logits, embed, hidden_states, rollback_records


def _forward_linear_layer_with_record(layer, hidden_states, mask, cache):
    """Forward through a Qwen3.5 linear attention layer recording rollback data.

    Replicates GatedDeltaNet.__call__ + DecoderLayer structure but saves
    initial cache states and intermediate (qkv, k, v, g, beta) so we can
    roll back the SSM state on partial block rejection.
    """
    linear = layer.linear_attn
    residual = hidden_states
    inputs = layer.input_layernorm(hidden_states)
    B, S, _ = inputs.shape

    qkv = linear.in_proj_qkv(inputs)
    z = linear.in_proj_z(inputs).reshape(B, S, linear.num_v_heads, linear.head_v_dim)
    b_raw = linear.in_proj_b(inputs)
    a_raw = linear.in_proj_a(inputs)

    initial_conv_state = cache[0] if cache[0] is not None else mx.zeros(
        (B, linear.conv_kernel_size - 1, linear.conv_dim),
        dtype=inputs.dtype,
    )
    initial_ssm_state = cache[1]

    if mask is not None:
        qkv = mx.where(mask[..., None], qkv, 0)
    conv_input = mx.concatenate([initial_conv_state, qkv], axis=1)
    n_keep = linear.conv_kernel_size - 1
    if cache.lengths is not None:
        ends = mx.clip(cache.lengths, 0, S)
        positions = (ends[:, None] + mx.arange(n_keep))[..., None]
        cache[0] = mx.take_along_axis(conv_input, positions, axis=1)
    else:
        cache[0] = mx.contiguous(conv_input[:, -n_keep:, :])
    conv_out = nn.silu(linear.conv1d(conv_input))

    queries, keys, values = [
        t.reshape(B, S, h, d)
        for t, h, d in zip(
            mx.split(conv_out, [linear.key_dim, 2 * linear.key_dim], -1),
            [linear.num_k_heads, linear.num_k_heads, linear.num_v_heads],
            [linear.head_k_dim, linear.head_k_dim, linear.head_v_dim],
        )
    ]

    inv_scale = keys.shape[-1] ** -0.5
    queries = (inv_scale**2) * mx.fast.rms_norm(queries, None, 1e-6)
    keys = inv_scale * mx.fast.rms_norm(keys, None, 1e-6)

    from mlx_lm.models.gated_delta import compute_g
    from .kernels import gated_delta_kernel_with_tape
    beta = mx.sigmoid(b_raw)
    g = compute_g(linear.A_log, a_raw, linear.dt_bias)

    out, state, tape = gated_delta_kernel_with_tape(
        queries, keys, values, g, beta, initial_ssm_state, mask,
    )

    cache[1] = state
    cache.advance(S)

    out = linear.norm(out, z)
    out = linear.out_proj(out.reshape(B, S, -1))
    hidden_states = residual + out

    residual = hidden_states
    hidden_states = layer.post_attention_layernorm(hidden_states)
    hidden_states = residual + layer.mlp(hidden_states)

    rollback_record = {
        'initial_conv_state': initial_conv_state,
        'initial_ssm_state': initial_ssm_state,
        'qkv': qkv,
        'k': keys,   # [B, T, Hk, Dk] — not GQA-repeated
        'g': g,      # [B, T, Hv]
        'tape': tape, # [B, T, Hv, Dv] float32 — precomputed innovation deltas
    }

    return hidden_states, rollback_record


def rollback_linear_caches(cache, rollback_records, accepted_inputs):
    """Roll back Qwen3.5 linear attention caches after partial block rejection.

    Restores conv_state and SSM state to what they would be after processing
    only the first accepted_inputs tokens from the block.

    Uses tape_replay_kernel: reads precomputed innovation deltas from the tape
    recorded during the verify forward pass, so no dot-product reduction is
    needed during rollback. Batches all layers into one kernel dispatch.
    """
    layer_indices = []
    initial_states = []
    all_tapes = []
    all_keys = []
    all_g = []

    for idx, record in rollback_records.items():
        layer_cache = cache[idx]

        initial_conv_state = record['initial_conv_state']
        qkv = record['qkv']
        n_keep = initial_conv_state.shape[1]
        conv_prefix = mx.concatenate(
            [initial_conv_state, qkv[:, :accepted_inputs, :]],
            axis=1,
        )
        layer_cache[0] = conv_prefix[:, -n_keep:, :]

        layer_indices.append(idx)
        initial_states.append(record['initial_ssm_state'])
        all_tapes.append(record['tape'][:, :accepted_inputs])
        all_keys.append(record['k'][:, :accepted_inputs])
        all_g.append(record['g'][:, :accepted_inputs])

    if not layer_indices:
        return

    from .kernels import tape_replay_kernel
    rebuilt_states = tape_replay_kernel(
        mx.concatenate(all_tapes, axis=0),
        mx.concatenate(all_keys, axis=0),
        mx.concatenate(all_g, axis=0),
        mx.concatenate(initial_states, axis=0),
    )

    for offset, idx in enumerate(layer_indices):
        cache[idx][1] = rebuilt_states[offset:offset + 1]


def forward_prefix(
    model,
    inputs: mx.array,
    cache=None,
    exit_layer: int = 19,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array], mx.array]:
    if capture_layers is None:
        capture_layers = []

    capture_set = set(capture_layers)
    inner = _get_inner_model(model)
    h = inner.embed_tokens(inputs)
    embed = h

    if cache is None:
        cache = [None] * len(inner.layers)

    if is_qwen35(model):
        fa_mask, ssm_mask = _make_masks(h, cache, inner)
    else:
        try:
            fa_mask = create_attention_mask(h, cache[0])
        except TypeError:
            fa_mask = cache[0].make_mask(h.shape[1])
        ssm_mask = None

    captured = {}
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        if getattr(layer, 'is_linear', False):
            h = layer(h, ssm_mask, cache=c)
        else:
            h = layer(h, fa_mask, cache=c)
        if i in capture_set:
            captured[i] = h
        if i == exit_layer:
            break

    hidden_states = [captured[i] for i in capture_layers if i <= exit_layer]

    return h, embed, hidden_states, fa_mask


def _apply_lm_head(model, h: mx.array) -> mx.array:
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head(h)
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        return model.language_model.lm_head(h)
    return _get_inner_model(model).embed_tokens.as_linear(h)


def forward_verifier_states(
    model,
    inputs: mx.array,
    cache=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array], Dict[int, Dict[str, mx.array]]]:
    """Forward pass that returns norm'd hidden states before lm_head.

    Returns (norm_hidden, embed, hidden_states, rollback_records).
    norm_hidden is the output of the final RMSNorm, ready for lm_head.
    This enables lazy-logits verification: compute lm_head in chunks
    only up to the rejection point, saving compute on early rejections.
    """
    if capture_layers is None:
        capture_layers = []

    capture_set = set(capture_layers)
    inner = _get_inner_model(model)
    h = inner.embed_tokens(inputs)
    embed = h

    if cache is None:
        from mlx_lm.models import cache as cache_module
        cache = cache_module.make_prompt_cache(model)

    q35 = is_qwen35(model)
    if q35:
        fa_mask, ssm_mask = _make_masks(h, cache, inner)
    else:
        try:
            fa_mask = create_attention_mask(h, cache[0])
        except TypeError:
            fa_mask = cache[0].make_mask(h.shape[1])
        ssm_mask = None

    captured = {}
    rollback_records = {}

    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        if q35 and getattr(layer, 'is_linear', False):
            h, record = _forward_linear_layer_with_record(layer, h, ssm_mask, c)
            rollback_records[i] = record
        else:
            h = layer(h, fa_mask, cache=c)
        if i in capture_set:
            captured[i] = h

    norm_hidden = inner.norm(h)
    hidden_states = [captured[i] for i in capture_layers]

    return norm_hidden, embed, hidden_states, rollback_records


def forward_suffix(
    model,
    h: mx.array,
    cache=None,
    start_layer: int = 20,
    mask=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, List[mx.array]]:
    if capture_layers is None:
        capture_layers = []

    capture_set = set(capture_layers)
    inner = _get_inner_model(model)

    if cache is None:
        cache = [None] * len(inner.layers)

    ssm_mask = None

    captured = {}
    for i in range(start_layer, len(inner.layers)):
        layer = inner.layers[i]
        if getattr(layer, 'is_linear', False):
            h = layer(h, ssm_mask, cache=cache[i])
        else:
            h = layer(h, mask, cache=cache[i])
        if i in capture_set:
            captured[i] = h

    h = inner.norm(h)

    if hasattr(model, 'lm_head') and model.lm_head is not None:
        logits = model.lm_head(h)
    else:
        logits = inner.embed_tokens.as_linear(h)

    hidden_states = [captured[i] for i in capture_layers if i >= start_layer]

    return logits, hidden_states
