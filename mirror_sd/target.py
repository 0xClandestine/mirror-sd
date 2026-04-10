"""Target model utilities for Mirror-SD.

Provides forward_with_hidden_states() which runs a target model while
capturing intermediate hidden states at specified layer indices. These
hidden states are the key input to the DFlash draft model.

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

    Captures **pre-layer** activations (the hidden state BEFORE each
    specified layer processes it), matching the DFlash training convention.

    Works with any mlx-lm model that has:
      - model.model.embed_tokens
      - model.model.layers
      - model.model.norm
      - model.lm_head (or model.model.embed_tokens.as_linear for tied weights)

    Args:
        model: The mlx-lm model
        inputs: Token IDs [B, L]
        cache: KV cache list
        capture_layers: Which layer indices to capture pre-layer hidden states from

    Returns:
        logits: [B, L, vocab_size]
        embed: Token embeddings [B, L, hidden_size]
        hidden_states: List of pre-layer hidden state tensors, one per capture_layer
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
        if i in capture_layers:
            captured[i] = h
        h = layer(h, mask, cache=c)

    h = inner.norm(h)

    if hasattr(model, 'lm_head') and model.lm_head is not None:
        logits = model.lm_head(h)
    else:
        logits = inner.embed_tokens.as_linear(h)

    hidden_states = [captured[i] for i in capture_layers]

    return logits, embed, hidden_states


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
