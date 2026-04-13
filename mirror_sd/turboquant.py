"""TurboQuant KV cache integration for Mirror-SD.

Wraps mlx-vlm's TurboQuantKVCache to replace standard KVCache in
full-attention layers, reducing memory pressure at long context lengths.

Key insight: TurboQuantKVCache.update_and_fetch() quantizes keys/values
internally and returns _QuantizedStateProxy objects. We must then call
cache.quantized_attention() instead of scaled_dot_product_attention().
"""

from typing import Any, List, Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_lm.models.cache import ArraysCache, KVCache

from mlx_vlm.turboquant import TurboQuantKVCache


def make_turboquant_cache(model, bits: float = 2.5, seed: int = 0) -> List[Any]:
    """Create a prompt cache with TurboQuantKVCache for full-attention layers.

    For Qwen3.5 (mixed attention), replaces KVCache entries with TurboQuantKVCache
    while keeping ArraysCache entries for linear attention layers unchanged.
    For standard models (all full-attention), replaces all KVCache entries.
    Also patches each full-attention layer's Attention.__call__ to use
    quantized_attention when the cache is TurboQuantKVCache.

    Args:
        model: The language model.
        bits: Quantization bit-width for TurboQuant (e.g. 2.5, 3.5).
        seed: Random seed for TurboQuant rotation matrices.

    Returns:
        List of cache objects (mix of TurboQuantKVCache and ArraysCache).
    """
    from mlx_lm.models import cache as cache_module

    cache = cache_module.make_prompt_cache(model)

    inner = _get_inner_model(model)
    if not hasattr(inner, 'layers'):
        return cache

    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        if isinstance(c, KVCache) and not getattr(layer, 'is_linear', False):
            cache[i] = TurboQuantKVCache(bits=bits, seed=seed)
            _patch_attention_for_tq(layer.self_attn)

    return cache


def is_turboquant_cache(cache_list):
    """Check if any cache entry uses TurboQuantKVCache."""
    return any(isinstance(c, TurboQuantKVCache) for c in cache_list)


def _patch_attention_for_tq(attn):
    """Patch an Attention module to use TurboQuant quantized_attention.

    Replaces attn.__call__ with a version that detects TurboQuantKVCache
    and routes to quantized_attention() instead of sdpa().
    """
    original_call = attn.__class__.__call__

    if hasattr(attn, '_tq_patched'):
        return
    attn._tq_patched = True

    def patched_call(self, x, mask=None, cache=None):
        if isinstance(cache, TurboQuantKVCache):
            return _tq_attention_forward(self, x, mask, cache)
        return original_call(self, x, mask, cache)

    import types
    attn.__class__.__call__ = patched_call


def _tq_attention_forward(attn, x, mask, cache):
    """Attention forward using TurboQuant quantized_attention.

    Replicates the standard Attention.__call__ logic but uses
    cache.quantized_attention() instead of scaled_dot_product_attention().
    Handles both Qwen3 (no gate) and Qwen3.5 (q_proj split into queries+gate).
    """
    B, L, D = x.shape

    q_proj_output = attn.q_proj(x)
    keys = attn.k_proj(x)
    values = attn.v_proj(x)

    n_heads = getattr(attn, 'n_heads', None) or getattr(attn, 'num_attention_heads', None)
    n_kv_heads = getattr(attn, 'n_kv_heads', None) or getattr(attn, 'num_key_value_heads', None)
    head_dim = getattr(attn, 'head_dim', None)
    if head_dim is None:
        head_dim = attn.q_proj.weight.shape[0] // n_heads

    q_out_dim = attn.q_proj.weight.shape[0]
    has_gate = (q_out_dim != n_heads * head_dim)

    if has_gate:
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, n_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)
    else:
        queries = q_proj_output.reshape(B, L, n_heads, head_dim)

    keys = keys.reshape(B, L, n_kv_heads, head_dim)
    values = values.reshape(B, L, n_kv_heads, head_dim)

    has_qk_norm = hasattr(attn, 'q_norm') and hasattr(attn, 'k_norm')
    if has_qk_norm:
        queries = attn.q_norm(queries)
        keys = attn.k_norm(keys)

    queries = queries.transpose(0, 2, 1, 3)
    keys = keys.transpose(0, 2, 1, 3)
    values = values.transpose(0, 2, 1, 3)

    queries = attn.rope(queries, offset=cache.offset)
    keys = attn.rope(keys, offset=cache.offset)

    keys_state, values_state = cache.update_and_fetch(keys, values)

    scale = getattr(attn, 'scale', head_dim ** -0.5)

    mask_str = mask if isinstance(mask, str) else None
    output = cache.quantized_attention(
        queries,
        keys_state=keys_state,
        values_state=values_state,
        scale=scale,
        mask=mask_str,
    )

    output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)

    if has_gate:
        return attn.o_proj(output * mx.sigmoid(gate))
    return attn.o_proj(output)


def _get_inner_model(model):
    inner = getattr(model, 'model', None) or getattr(model, 'language_model', None)
    if inner is not None and not hasattr(inner, 'embed_tokens'):
        inner = getattr(inner, 'model', inner)
    return inner
