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
- Per-layer mx.compile for linear attention verify
"""

from typing import Any, Dict, List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.base import create_attention_mask
from mlx_lm.models.cache import ArraysCache, KVCache

_COMPILED_LINEAR_VERIFY_FNS: Dict[int, Any] = {}
_COMPILED_FULL_ATTENTION_VERIFY_FNS: Dict[int, Any] = {}


def _is_qwen35_full_attn(layer):
    """Check if a full-attention layer uses the Qwen3.5 gate structure."""
    attn = getattr(layer, 'self_attn', None)
    if attn is None:
        return False
    q_proj = getattr(attn, 'q_proj', None)
    if q_proj is None:
        return False
    return hasattr(attn, 'q_norm') and hasattr(attn, 'k_norm')


def _get_attn_dims(attn):
    """Normalize attention head dimensions across Qwen3 / Qwen3.5."""
    n_heads = getattr(attn, 'num_attention_heads', None) or getattr(attn, 'n_heads', None)
    n_kv_heads = getattr(attn, 'num_key_value_heads', None) or getattr(attn, 'n_kv_heads', None)
    head_dim = getattr(attn, 'head_dim', None)
    if head_dim is None:
        head_dim = attn.q_proj.weight.shape[0] // n_heads
    return n_heads, n_kv_heads, head_dim


def get_compiled_full_attention_verify_fn(layer):
    """Get or create a compiled full-attention verify function for a layer.

    Takes explicit KV cache arrays (old_keys, old_values, offset) as inputs
    instead of using the KVCache object. This lets mx.compile trace the
    full computation graph and avoid per-iteration graph rebuilds.
    """
    key = id(layer)
    compiled = _COMPILED_FULL_ATTENTION_VERIFY_FNS.get(key)
    if compiled is not None:
        return compiled

    attn = layer.self_attn
    n_heads, n_kv_heads, head_dim = _get_attn_dims(attn)
    has_qk_norm = hasattr(attn, 'q_norm') and hasattr(attn, 'k_norm')
    q_out_dim = attn.q_proj.weight.shape[0]
    has_gate = (q_out_dim != n_heads * head_dim)

    if has_qk_norm:
        @mx.compile
        def compiled_full_attention_verify(
            hidden_states: mx.array,
            old_keys: mx.array,
            old_values: mx.array,
            offset: int,
        ) -> Tuple[mx.array, mx.array, mx.array]:
            residual = hidden_states
            inputs = layer.input_layernorm(hidden_states)
            B, L, _ = inputs.shape

            q_proj_out = attn.q_proj(inputs)
            if has_gate:
                queries, gate = mx.split(
                    q_proj_out.reshape(B, L, n_heads, -1),
                    2, axis=-1,
                )
                gate = gate.reshape(B, L, -1)
            else:
                queries = q_proj_out.reshape(B, L, n_heads, -1)

            new_keys = attn.k_proj(inputs)
            new_values = attn.v_proj(inputs)

            queries = attn.q_norm(queries).transpose(0, 2, 1, 3)
            new_keys = attn.k_norm(
                new_keys.reshape(B, L, n_kv_heads, -1)
            ).transpose(0, 2, 1, 3)
            new_values = new_values.reshape(
                B, L, n_kv_heads, -1
            ).transpose(0, 2, 1, 3)

            queries = attn.rope(queries, offset=offset)
            new_keys = attn.rope(new_keys, offset=offset)

            keys = mx.concatenate([old_keys[..., :offset, :], new_keys], axis=2)
            values = mx.concatenate([old_values[..., :offset, :], new_values], axis=2)
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=attn.scale, mask="causal",
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
            return hidden_states, new_keys, new_values
    else:
        @mx.compile
        def compiled_full_attention_verify(
            hidden_states: mx.array,
            old_keys: mx.array,
            old_values: mx.array,
            offset: int,
        ) -> Tuple[mx.array, mx.array, mx.array]:
            residual = hidden_states
            inputs = layer.input_layernorm(hidden_states)
            B, L, _ = inputs.shape

            queries = attn.q_proj(inputs).reshape(B, L, n_heads, -1).transpose(0, 2, 1, 3)
            new_keys = attn.k_proj(inputs).reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)
            new_values = attn.v_proj(inputs).reshape(B, L, n_kv_heads, -1).transpose(0, 2, 1, 3)

            queries = attn.rope(queries, offset=offset)
            new_keys = attn.rope(new_keys, offset=offset)

            keys = mx.concatenate([old_keys[..., :offset, :], new_keys], axis=2)
            values = mx.concatenate([old_values[..., :offset, :], new_values], axis=2)
            output = mx.fast.scaled_dot_product_attention(
                queries, keys, values, scale=attn.scale, mask="causal",
            )
            output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
            output = attn.o_proj(output)

            hidden_states = residual + output
            residual = hidden_states
            hidden_states = layer.post_attention_layernorm(hidden_states)
            hidden_states = residual + layer.mlp(hidden_states)
            return hidden_states, new_keys, new_values

    _COMPILED_FULL_ATTENTION_VERIFY_FNS[key] = compiled_full_attention_verify
    return compiled_full_attention_verify


def _forward_full_attention_layer_compiled(layer, hidden_states, cache):
    """Forward a full-attention layer using compiled function with explicit cache.

    Only used during verify (hidden_states.shape[1] > 1) when cache is
    populated. Falls back to standard layer call otherwise.
    """
    if (
        cache is not None
        and isinstance(cache, KVCache)
        and cache.keys is not None
        and cache.values is not None
        and hidden_states.shape[1] > 1
    ):
        compiled = get_compiled_full_attention_verify_fn(layer)
        new_hidden, new_keys, new_values = compiled(
            hidden_states, cache.keys, cache.values, cache.offset,
        )
        cache.update_and_fetch(new_keys, new_values)
        return new_hidden
    return layer(hidden_states, mask="causal", cache=cache)


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
    compile_full: bool = False,
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
        elif compile_full and inputs.shape[1] > 1:
            h = _forward_full_attention_layer_compiled(layer, h, c)
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
    compile_full: bool = False,
) -> Tuple[mx.array, mx.array, List[mx.array], Dict[int, Dict[str, mx.array]]]:
    """Forward pass for Qwen3.5 that records rollback data for linear attention.

    Returns same as forward_with_hidden_states plus rollback_records dict
    mapping linear layer index to {initial_conv_state, initial_ssm_state,
    qkv, k, v, g, beta, repeat_factor}.
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
        elif inputs.shape[1] > 1 and isinstance(c, KVCache) and c.keys is not None:
            h = _forward_full_attention_layer_compiled(layer, h, c)
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

    from mlx_lm.models.gated_delta import compute_g, gated_delta_update
    beta = mx.sigmoid(b_raw)
    g = compute_g(linear.A_log, a_raw, linear.dt_bias)

    out, state = gated_delta_update(
        queries, keys, values, a_raw, b_raw, linear.A_log, linear.dt_bias,
        initial_ssm_state, mask, use_kernel=True,
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
        'k': keys,
        'v': values,
        'g': g,
        'beta': beta,
        'repeat_factor': linear.num_v_heads // linear.num_k_heads,
    }

    return hidden_states, rollback_record


def rollback_linear_caches(cache, rollback_records, accepted_inputs):
    """Roll back Qwen3.5 linear attention caches after partial block rejection.

    Restores conv_state and SSM state to what they would be after processing
    only the first accepted_inputs tokens from the block.

    Args:
        cache: The model's cache list
        rollback_records: Dict mapping layer index to rollback record
        accepted_inputs: Number of accepted input tokens from the block
                        (including the anchor token)
    """
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

        record_keys = record['k'][:, :accepted_inputs]
        repeat_factor = int(record['repeat_factor'])
        if repeat_factor > 1:
            record_keys = mx.repeat(record_keys, repeat_factor, axis=2)
        record_values = record['v'][:, :accepted_inputs]
        record_g = record['g'][:, :accepted_inputs]
        record_beta = record['beta'][:, :accepted_inputs]

        layer_cache[1] = _advance_gated_delta_states(
            record['initial_ssm_state'],
            record_keys, record_values, record_g, record_beta,
        )


def _advance_gated_delta_states(initial_state, keys, values, g, beta):
    """Advance GatedDeltaNet SSM state for a subset of tokens.

    Starting from initial_state, applies the recurrent update for each
    token using its (k, v, g, beta) values. Used for rollback: advance
    through only the accepted tokens to reconstruct the correct SSM state.

    Uses custom Metal kernel when available, falls back to Python loop.
    """
    from .ssm_kernel import advance_gated_delta_states_metal
    return advance_gated_delta_states_metal(initial_state, keys, values, g, beta)


def get_compiled_linear_verify_fn(layer):
    """Get or create a compiled linear attention verify function for a layer.

    Compiled functions have fixed input shapes (determined by block_size),
    so they never need recompilation during the decode loop. Model weights
    are captured as constants via closure.
    """
    key = id(layer)
    compiled = _COMPILED_LINEAR_VERIFY_FNS.get(key)
    if compiled is not None:
        return compiled

    linear = layer.linear_attn

    @mx.compile
    def compiled_linear_verify(
        hidden_states: mx.array,
        initial_conv_state: mx.array,
        initial_ssm_state: mx.array,
    ):
        residual = hidden_states
        inputs = layer.input_layernorm(hidden_states)
        B, S, _ = inputs.shape

        qkv = linear.in_proj_qkv(inputs)
        z = linear.in_proj_z(inputs).reshape(B, S, linear.num_v_heads, linear.head_v_dim)
        b_raw = linear.in_proj_b(inputs)
        a_raw = linear.in_proj_a(inputs)

        conv_input = mx.concatenate([initial_conv_state, qkv], axis=1)
        n_keep = linear.conv_kernel_size - 1
        new_conv_state = mx.contiguous(conv_input[:, -n_keep:, :])
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
        beta = mx.sigmoid(b_raw)
        g = mx.exp(-mx.exp(linear.A_log.astype(mx.float32)) * nn.softplus(a_raw + linear.dt_bias))

        from mlx_lm.models.gated_delta import gated_delta_ops
        out, new_ssm_state = gated_delta_ops(queries, keys, values, g, beta, initial_ssm_state, None)

        out = linear.norm(out, z)
        out = linear.out_proj(out.reshape(B, S, -1))
        hidden_states = residual + out

        residual = hidden_states
        hidden_states = layer.post_attention_layernorm(hidden_states)
        hidden_states = residual + layer.mlp(hidden_states)

        return hidden_states, new_conv_state, new_ssm_state, qkv, keys, values, g, beta

    _COMPILED_LINEAR_VERIFY_FNS[key] = compiled_linear_verify
    return compiled_linear_verify


def forward_with_hidden_states_compiled(
    model,
    inputs: mx.array,
    cache=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array], Dict[int, Dict[str, mx.array]]]:
    """Compiled forward pass for Qwen3.5 verify with rollback recording.

    Uses per-layer mx.compile for both linear AND full attention layers.
    Linear layers get fixed-shape compiled functions with explicit cache.
    Full attention layers use explicit KV cache arrays to avoid graph
    rebuilds from growing cache shapes.
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
        if getattr(layer, 'is_linear', False) and ssm_mask is None and c[0] is not None and c[1] is not None:
            compiled_fn = get_compiled_linear_verify_fn(layer)
            initial_conv_state = c[0]
            initial_ssm_state = c[1]
            h, new_conv_state, new_ssm_state, qkv, keys, values, g, beta = compiled_fn(h, initial_conv_state, initial_ssm_state)
            c[0] = new_conv_state
            c[1] = new_ssm_state
            c.advance(inputs.shape[1])
            repeat_factor = layer.linear_attn.num_v_heads // layer.linear_attn.num_k_heads
            rollback_records[i] = {
                'initial_conv_state': initial_conv_state,
                'initial_ssm_state': initial_ssm_state,
                'qkv': qkv,
                'k': keys,
                'v': values,
                'g': g,
                'beta': beta,
                'repeat_factor': repeat_factor,
            }
        elif getattr(layer, 'is_linear', False):
            h, record = _forward_linear_layer_with_record(layer, h, ssm_mask, c)
            rollback_records[i] = record
        elif inputs.shape[1] > 1 and isinstance(c, KVCache) and c.keys is not None:
            h = _forward_full_attention_layer_compiled(layer, h, c)
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
    compile_full: bool = False,
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
        elif compile_full and inputs.shape[1] > 1 and isinstance(c, KVCache) and c.keys is not None:
            h = _forward_full_attention_layer_compiled(layer, h, c)
        else:
            h = layer(h, fa_mask, cache=c)
        if i in capture_set:
            captured[i] = h

    norm_hidden = inner.norm(h)
    hidden_states = [captured[i] for i in capture_layers]

    return norm_hidden, embed, hidden_states, rollback_records


def forward_verifier_states_compiled(
    model,
    inputs: mx.array,
    cache=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array], Dict[int, Dict[str, mx.array]]]:
    """Compiled variant of forward_verifier_states for Qwen3.5."""
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
        if getattr(layer, 'is_linear', False) and ssm_mask is None and c[0] is not None and c[1] is not None:
            compiled_fn = get_compiled_linear_verify_fn(layer)
            initial_conv_state = c[0]
            initial_ssm_state = c[1]
            h, new_conv_state, new_ssm_state, qkv, keys, values, g, beta = compiled_fn(h, initial_conv_state, initial_ssm_state)
            c[0] = new_conv_state
            c[1] = new_ssm_state
            c.advance(inputs.shape[1])
            repeat_factor = layer.linear_attn.num_v_heads // layer.linear_attn.num_k_heads
            rollback_records[i] = {
                'initial_conv_state': initial_conv_state,
                'initial_ssm_state': initial_ssm_state,
                'qkv': qkv,
                'k': keys,
                'v': values,
                'g': g,
                'beta': beta,
                'repeat_factor': repeat_factor,
            }
        elif getattr(layer, 'is_linear', False):
            h, record = _forward_linear_layer_with_record(layer, h, ssm_mask, c)
            rollback_records[i] = record
        elif inputs.shape[1] > 1 and isinstance(c, KVCache) and c.keys is not None:
            h = _forward_full_attention_layer_compiled(layer, h, c)
        else:
            h = layer(h, fa_mask, cache=c)
        if i in capture_set:
            captured[i] = h

    norm_hidden = inner.norm(h)
    hidden_states = [captured[i] for i in capture_layers]

    return norm_hidden, embed, hidden_states, rollback_records


def forward_accept_all_block(
    model,
    inputs: mx.array,
    cache=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array]]:
    """Forward pass optimized for the all-accepted case.

    Only computes lm_head on the last position instead of all positions,
    saving ~(block_size-1)/block_size of the lm_head compute.
    Returns (logits_last, embed, hidden_states) where logits_last has
    shape [1, 1, vocab_size].
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
    logits_last = _apply_lm_head(model, h[:, -1:, :])

    hidden_states = [captured[i] for i in capture_layers]

    return logits_last, embed, hidden_states


def forward_accept_all_block_compiled(
    model,
    inputs: mx.array,
    cache=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array], Dict[int, Dict[str, mx.array]]]:
    """Compiled all-accepted forward for Qwen3.5 (with rollback records).

    Only computes lm_head on the last position. Still records rollback data
    because even if all tokens are accepted in THIS iteration, we need the
    rollback data for the NEXT iteration's potential partial rejection.
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
        if getattr(layer, 'is_linear', False) and ssm_mask is None and c[0] is not None and c[1] is not None:
            compiled_fn = get_compiled_linear_verify_fn(layer)
            initial_conv_state = c[0]
            initial_ssm_state = c[1]
            h, new_conv_state, new_ssm_state, qkv, keys, values, g, beta = compiled_fn(h, initial_conv_state, initial_ssm_state)
            c[0] = new_conv_state
            c[1] = new_ssm_state
            c.advance(inputs.shape[1])
            repeat_factor = layer.linear_attn.num_v_heads // layer.linear_attn.num_k_heads
            rollback_records[i] = {
                'initial_conv_state': initial_conv_state,
                'initial_ssm_state': initial_ssm_state,
                'qkv': qkv,
                'k': keys,
                'v': values,
                'g': g,
                'beta': beta,
                'repeat_factor': repeat_factor,
            }
        elif getattr(layer, 'is_linear', False):
            h, record = _forward_linear_layer_with_record(layer, h, ssm_mask, c)
            rollback_records[i] = record
        elif inputs.shape[1] > 1 and isinstance(c, KVCache) and c.keys is not None:
            h = _forward_full_attention_layer_compiled(layer, h, c)
        else:
            h = layer(h, fa_mask, cache=c)
        if i in capture_set:
            captured[i] = h

    h = inner.norm(h)
    logits_last = _apply_lm_head(model, h[:, -1:, :])

    hidden_states = [captured[i] for i in capture_layers]

    return logits_last, embed, hidden_states, rollback_records


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
