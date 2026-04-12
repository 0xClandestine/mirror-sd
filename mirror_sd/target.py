"""Target model utilities for Mirror-SD.

Provides forward_with_hidden_states() which runs a target model while
capturing intermediate hidden states at specified layer indices. These
hidden states are the key input to the DFlash draft model.

Provides forward_split() for Mirror-SD early-exit: runs prefix layers,
emits hidden states for the draft model, then continues suffix layers.
This enables parallel ANE draft || GPU suffix execution (Eq. 10).
"""

from typing import List, Optional, Tuple

import mlx.core as mx

from mlx_lm.models.base import create_attention_mask


def _get_inner_model(model):
    """Get the inner model that has embed_tokens, layers, and norm.

    Handles both Qwen3 (model.model) and Qwen3.5 (model.language_model.model).
    """
    inner = getattr(model, 'model', None) or getattr(model, 'language_model', None)
    if inner is not None and not hasattr(inner, 'embed_tokens'):
        inner = getattr(inner, 'model', inner)
    return inner


def get_embed_tokens(model):
    """Get the embed_tokens layer from any model architecture."""
    return _get_inner_model(model).embed_tokens


def get_lm_head(model):
    """Get the lm_head layer from any model architecture."""
    if hasattr(model, 'lm_head') and model.lm_head is not None:
        return model.lm_head
    if hasattr(model, 'language_model') and hasattr(model.language_model, 'lm_head'):
        return model.language_model.lm_head
    return _get_inner_model(model).embed_tokens.as_linear


def forward_with_hidden_states(
    model,
    inputs: mx.array,
    cache=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array]]:
    """Run the target model forward pass while capturing hidden states.

    Captures **post-layer** activations (the hidden state AFTER each
    specified layer processes it), matching the HuggingFace convention
    where output.hidden_states[layer_id + 1] = output of layer layer_id.
    This matches the DFlash training convention used in the PyTorch reference.

    Args:
        model: The mlx-lm model
        inputs: Token IDs [B, L]
        cache: KV cache list
        capture_layers: Which layer indices to capture post-layer hidden states from

    Returns:
        logits: [B, L, vocab_size]
        embed: Token embeddings [B, L, hidden_size]
        hidden_states: List of post-layer hidden state tensors, one per capture_layer
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

    try:
        mask = create_attention_mask(h, cache[0])
    except TypeError:
        mask = cache[0].make_mask(h.shape[1])

    captured = {}
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        h = layer(h, mask, cache=c)
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


def forward_prefix(
    model,
    inputs: mx.array,
    cache=None,
    exit_layer: int = 19,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, mx.array, List[mx.array], mx.array]:
    """Run target model prefix layers (0..exit_layer) for early-exit.

    Used by Mirror-SD to emit hidden states at an intermediate layer so
    the draft model can start while the target continues with suffix layers.

    Args:
        model: The mlx-lm model
        inputs: Token IDs [B, L]
        cache: KV cache list
        exit_layer: Layer index at which to stop (inclusive). Hidden state
                    AFTER this layer is the early-exit hidden state.
        capture_layers: Additional layers to capture post-layer hidden states from

    Returns:
        h: Hidden state after exit_layer [B, L, hidden_size]
        embed: Token embeddings [B, L, hidden_size]
        captured: List of post-layer hidden state tensors for capture_layers
        mask: Attention mask (needed for suffix continuation)
    """
    if capture_layers is None:
        capture_layers = []

    capture_set = set(capture_layers)
    inner = _get_inner_model(model)
    h = inner.embed_tokens(inputs)
    embed = h

    if cache is None:
        cache = [None] * len(inner.layers)

    try:
        mask = create_attention_mask(h, cache[0])
    except TypeError:
        mask = cache[0].make_mask(h.shape[1])

    captured = {}
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        h = layer(h, mask, cache=c)
        if i in capture_set:
            captured[i] = h
        if i == exit_layer:
            break

    hidden_states = [captured[i] for i in capture_layers if i <= exit_layer]

    return h, embed, hidden_states, mask


def forward_suffix(
    model,
    h: mx.array,
    cache=None,
    start_layer: int = 20,
    mask=None,
    capture_layers: Optional[List[int]] = None,
) -> Tuple[mx.array, List[mx.array]]:
    """Run target model suffix layers (start_layer..N) on a hidden state.

    Used by Mirror-SD to continue the target model after the draft has
    been started in parallel. The hidden state from forward_prefix is fed
    into this function.

    Args:
        model: The mlx-lm model
        h: Hidden state from forward_prefix [B, L, hidden_size]
        cache: KV cache list (must already have prefix layers cached)
        start_layer: First layer to run (exit_layer + 1)
        mask: Attention mask (from forward_prefix)
        capture_layers: Additional layers to capture post-layer hidden states from

    Returns:
        logits: [B, L, vocab_size]
        hidden_states: List of post-layer hidden state tensors for capture_layers
    """
    if capture_layers is None:
        capture_layers = []

    capture_set = set(capture_layers)
    inner = _get_inner_model(model)

    if cache is None:
        cache = [None] * len(inner.layers)

    captured = {}
    for i in range(start_layer, len(inner.layers)):
        h = inner.layers[i](h, mask, cache=cache[i])
        if i in capture_set:
            captured[i] = h

    h = inner.norm(h)

    if hasattr(model, 'lm_head') and model.lm_head is not None:
        logits = model.lm_head(h)
    else:
        logits = inner.embed_tokens.as_linear(h)

    hidden_states = [captured[i] for i in capture_layers if i >= start_layer]

    return logits, hidden_states
