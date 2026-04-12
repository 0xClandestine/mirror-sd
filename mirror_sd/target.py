"""Target model utilities for Mirror-SD.

Provides forward_with_hidden_states() which runs a target model while
capturing intermediate hidden states at specified layer indices. These
hidden states are the key input to the DFlash draft model.

Provides forward_split() for Mirror-SD early-exit: runs prefix layers,
emits hidden states for the draft model, then continues suffix layers.
This enables parallel ANE draft || GPU suffix execution (Eq. 10).

Also provides extract_context_feature() for fusing multi-layer hidden
states into a single tensor for the draft model's fc layer.
"""

from typing import List, Optional, Tuple

import mlx.core as mx
import mlx.nn as nn


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

    inner = model.model
    h = inner.embed_tokens(inputs)
    embed = h

    if cache is None:
        cache = [None] * len(inner.layers)

    from mlx_lm.models.base import create_attention_mask
    mask = create_attention_mask(h, cache[0])

    captured = {}
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        h = layer(h, mask, cache=c)
        if i in capture_layers:
            captured[i] = h

    h = inner.norm(h)

    if hasattr(model, 'lm_head') and model.lm_head is not None:
        logits = model.lm_head(h)
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

    inner = model.model
    h = inner.embed_tokens(inputs)
    embed = h

    if cache is None:
        cache = [None] * len(inner.layers)

    from mlx_lm.models.base import create_attention_mask
    mask = create_attention_mask(h, cache[0])

    captured = {}
    for i, (layer, c) in enumerate(zip(inner.layers, cache)):
        h = layer(h, mask, cache=c)
        if i in capture_layers:
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

    inner = model.model

    if cache is None:
        cache = [None] * len(inner.layers)

    captured = {}
    for i in range(start_layer, len(inner.layers)):
        h = inner.layers[i](h, mask, cache=cache[i])
        if i in capture_layers:
            captured[i] = h

    h = inner.norm(h)

    if hasattr(model, 'lm_head') and model.lm_head is not None:
        logits = model.lm_head(h)
    else:
        logits = inner.embed_tokens.as_linear(h)

    hidden_states = [captured[i] for i in capture_layers if i >= start_layer]

    return logits, hidden_states


def extract_context_feature(
    hidden_states: List[mx.array],
    layer_ids: List[int],
) -> mx.array:
    """Fuse multi-layer hidden states for the draft model.

    If hidden_states length matches layer_ids length (captured-only mode),
    concatenates all directly. Otherwise, uses layer_ids as indices into
    a full hidden_states list (HuggingFace convention with offset=1).
    """
    if len(hidden_states) == len(layer_ids):
        return mx.concatenate(hidden_states, axis=-1)
    offset = 1
    selected = [hidden_states[lid + offset] for lid in layer_ids]
    return mx.concatenate(selected, axis=-1)
