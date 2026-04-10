"""DFlash draft model for MLX.

Implements the DFlash block-diffusion draft model architecture from
"DFlash: Block Diffusion for Flash Speculative Decoding" (arXiv:2602.06036).

Architecture (from z-lab/Qwen3-8B-DFlash-b16):
  fc:           Linear(num_target_features * hidden_size -> hidden_size) - fuses target hidden states
  hidden_norm:  RMSNorm(hidden_size)
  layers:       5 x Qwen3DFlashDecoderLayer (target-aware attention + MLP)
  norm:         RMSNorm(hidden_size) - final norm before target's lm_head

Key design (from Section 4.1 of the DFlash paper):
  - Target context features are extracted from 5 layers of the target model
  - Features are fused via fc + hidden_norm into a compact context vector
  - Context vector is injected into K/V projections of EVERY draft layer
  - K/V come from both target context (ctx) and draft positions (noise)
  - All masked positions decoded in parallel (block diffusion, non-causal)
  - Draft model shares embedding and lm_head with the target model
"""

import math
from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, List

import mlx.core as mx
import mlx.nn as nn


@dataclass
class DFlashConfig:
    hidden_size: int = 4096
    num_hidden_layers: int = 5
    intermediate_size: int = 12288
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    rope_theta: float = 1000000.0
    max_position_embeddings: int = 40960
    vocab_size: int = 151936
    block_size: int = 16
    mask_token_id: int = 151669
    target_layer_ids: List[int] = field(default_factory=lambda: [4, 12, 19, 26, 33])
    num_target_layers: int = 36
    rope_scaling: Optional[dict] = None

    @classmethod
    def qwen3_8b(cls, num_target_layers: int = 36) -> "DFlashConfig":
        cfg = cls(num_target_layers=num_target_layers)
        cfg.target_layer_ids = build_target_layer_ids(
            num_target_layers, cfg.num_hidden_layers
        )
        return cfg

    @classmethod
    def qwen3_4b(cls, num_target_layers: int = 36) -> "DFlashConfig":
        cfg = cls(
            hidden_size=2560,
            num_hidden_layers=5,
            intermediate_size=6912,
            num_attention_heads=20,
            num_key_value_heads=4,
            head_dim=128,
            vocab_size=151936,
        )
        cfg.num_target_layers = num_target_layers
        cfg.target_layer_ids = build_target_layer_ids(
            num_target_layers, cfg.num_hidden_layers
        )
        return cfg

    @classmethod
    def from_dict(cls, d: dict) -> "DFlashConfig":
        dflash_cfg = d.get("dflash_config", {})
        num_hidden_layers = d.get("num_hidden_layers", dflash_cfg.get("num_hidden_layers", 5))
        num_target_layers = dflash_cfg.get("num_target_layers", d.get("num_target_layers", 36))
        cfg = cls(
            hidden_size=d.get("hidden_size", 4096),
            num_hidden_layers=num_hidden_layers,
            intermediate_size=d.get("intermediate_size", 10944),
            num_attention_heads=d.get("num_attention_heads", 32),
            num_key_value_heads=d.get("num_key_value_heads", 2),
            head_dim=d.get("head_dim", 128),
            rms_norm_eps=d.get("rms_norm_eps", 1e-6),
            rope_theta=d.get("rope_theta", 10000.0),
            max_position_embeddings=d.get("max_position_embeddings", 4096),
            vocab_size=d.get("vocab_size", 151936),
            block_size=dflash_cfg.get("block_size", d.get("block_size", 16)),
            mask_token_id=dflash_cfg.get("mask_token_id", d.get("mask_token_id", 151667)),
            num_target_layers=num_target_layers,
        )
        if "target_layer_ids" in dflash_cfg:
            cfg.target_layer_ids = dflash_cfg["target_layer_ids"]
        else:
            cfg.target_layer_ids = build_target_layer_ids(
                cfg.num_target_layers, cfg.num_hidden_layers
            )
        if "rope_scaling" in d:
            cfg.rope_scaling = d["rope_scaling"]
        return cfg


def build_target_layer_ids(
    num_target_layers: int, num_draft_layers: int
) -> List[int]:
    if num_draft_layers == 1:
        return [num_target_layers // 2]
    start = 1
    end = num_target_layers - 3
    span = end - start
    return [
        int(round(start + (i * span) / (num_draft_layers - 1)))
        for i in range(num_draft_layers)
    ]


def extract_context_feature(
    hidden_states: List[mx.array],
    layer_ids: List[int],
) -> mx.array:
    if len(hidden_states) == len(layer_ids):
        return mx.concatenate(hidden_states, axis=-1)
    offset = 1
    selected = [hidden_states[lid + offset] for lid in layer_ids]
    return mx.concatenate(selected, axis=-1)


def sample(logits: mx.array, temperature: float = 0.0) -> mx.array:
    if temperature < 1e-5:
        return mx.argmax(logits, axis=-1)
    logits = logits / temperature
    probs = mx.softmax(logits, axis=-1)
    return mx.argmax(probs, axis=-1)


class Qwen3DFlashAttention(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx
        self.head_dim = config.head_dim
        self.n_heads = config.num_attention_heads
        self.n_kv_heads = config.num_key_value_heads
        self.scale = self.head_dim ** -0.5

        self.q_proj = nn.Linear(config.hidden_size, self.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.hidden_size, self.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, config.hidden_size, bias=False)

        self.q_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=config.rms_norm_eps)

        from mlx_lm.models.rope_utils import initialize_rope
        self.rope = initialize_rope(
            self.head_dim,
            base=config.rope_theta,
            traditional=False,
            scaling_config=config.rope_scaling,
            max_position_embeddings=config.max_position_embeddings,
        )

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> Tuple[mx.array, Optional[Any]]:
        B, q_len, _ = hidden_states.shape
        ctx_len = target_hidden.shape[1]

        q = self.q_proj(hidden_states)
        q = self.q_norm(q.reshape(B, q_len, self.n_heads, -1)).transpose(0, 2, 1, 3)

        k_ctx = self.k_proj(target_hidden)
        k_noise = self.k_proj(hidden_states)
        v_ctx = self.v_proj(target_hidden)
        v_noise = self.v_proj(hidden_states)

        k = mx.concatenate([k_ctx, k_noise], axis=1)
        v = mx.concatenate([v_ctx, v_noise], axis=1)

        k = self.k_norm(k.reshape(B, ctx_len + q_len, self.n_kv_heads, -1)).transpose(0, 2, 1, 3)
        v = v.reshape(B, ctx_len + q_len, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            q = self.rope(q, offset=cache.offset)
            k = self.rope(k, offset=cache.offset)
            k, v = cache.update_and_fetch(k, v)
        else:
            q = self.rope(q)
            k = self.rope(k)

        n_rep = self.n_heads // self.n_kv_heads
        if n_rep > 1:
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        output = mx.fast.scaled_dot_product_attention(
            q, k, v, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, q_len, -1)
        return self.o_proj(output)


class Qwen3MLP(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.gate_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.up_proj = nn.Linear(config.hidden_size, config.intermediate_size, bias=False)
        self.down_proj = nn.Linear(config.intermediate_size, config.hidden_size, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(nn.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3DFlashDecoderLayer(nn.Module):
    def __init__(self, config: DFlashConfig, layer_idx: int):
        super().__init__()
        self.self_attn = Qwen3DFlashAttention(config, layer_idx)
        self.mlp = Qwen3MLP(config)
        self.input_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)

    def __call__(
        self,
        hidden_states: mx.array,
        target_hidden: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states=hidden_states,
            target_hidden=target_hidden,
            mask=mask,
            cache=cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


class DFlashDraftModel(nn.Module):
    def __init__(self, config: DFlashConfig):
        super().__init__()
        self.config = config
        self.layers = [
            Qwen3DFlashDecoderLayer(config, layer_idx)
            for layer_idx in range(config.num_hidden_layers)
        ]
        self.fc = nn.Linear(
            len(config.target_layer_ids) * config.hidden_size,
            config.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.norm = nn.RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.block_size = config.block_size
        self.mask_token_id = config.mask_token_id

    def __call__(
        self,
        noise_embedding: mx.array,
        target_hidden: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[list] = None,
    ) -> mx.array:
        hidden_states = noise_embedding
        target_hidden = self.hidden_norm(self.fc(target_hidden))

        for layer in self.layers:
            hidden_states = layer(
                hidden_states=hidden_states,
                target_hidden=target_hidden,
                mask=mask,
                cache=None,
            )

        return self.norm(hidden_states)

    def sanitize(self, weights):
        return weights
