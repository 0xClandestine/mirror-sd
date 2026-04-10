"""Test attn_residual kernel in isolation."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS
from mirror_sd.loader import load_dflash_model

SEQ_Q = 16
CTX_LEN = 64

def cosine_sim(a, b):
    a_f = a.astype(mx.float32).flatten()
    b_f = b.astype(mx.float32).flatten()
    dot = float(mx.sum(a_f * b_f))
    na = float(mx.sqrt(mx.sum(a_f * a_f)))
    nb = float(mx.sqrt(mx.sum(b_f * b_f)))
    return dot / (na * nb + 1e-8)

def mlx_rmsnorm(x, weight, eps=1e-6):
    ms = x.mean(axis=-1, keepdims=True)
    diff = x - ms
    sq = diff * diff
    mean_sq = sq.mean(axis=-1, keepdims=True)
    inv_std = mx.rsqrt(mean_sq + eps)
    return x * inv_std * weight

def mlx_rope(x, rope_theta=1000000.0, offset=0):
    """Apply RoPE to x of shape [1, seq, n_heads, head_dim]."""
    seq = x.shape[1]
    n_heads = x.shape[2]
    head_dim = x.shape[3]
    half_dim = head_dim // 2
    cos_vals = mx.zeros((1, 1, seq, half_dim))
    sin_vals = mx.zeros((1, 1, seq, half_dim))
    import math
    for pos in range(seq):
        for d in range(half_dim):
            freq = 1.0 / (rope_theta ** (2.0 * d / head_dim))
            angle = (offset + pos) * freq
            cos_vals[0, 0, pos, d] = math.cos(angle)
            sin_vals[0, 0, pos, d] = math.sin(angle)
    # Extend to full head_dim
    cos_full = mx.concatenate([cos_vals, mx.ones((1, 1, seq, half_dim))], axis=-1)
    sin_full = mx.concatenate([sin_vals, mx.zeros((1, 1, seq, half_dim))], axis=-1)
    x_pairs = x.reshape(1, seq, n_heads, half_dim, 2)
    x_even = x_pairs[..., 0]
    x_odd = x_pairs[..., 1]
    rotated = mx.concatenate([-x_odd, x_even], axis=-1).reshape(1, seq, n_heads, head_dim)
    return x * cos_full + rotated * sin_full

print("Loading...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)

layer = draft_model.layers[0]

mx.random.seed(42)
noise = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
target = mx.random.normal(shape=(1, CTX_LEN, 5 * HIDDEN), dtype=mx.bfloat16)

# Compute context (fc + hidden_norm)
ane_model._write_mlx_2d(ane_model.b_target, target)
ane_model.kernels['fc_norm'].run(
    [ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm],
    [ane_model.b_context],
)
context = ane_model._read_mlx_2d(ane_model.b_context, CTX_LEN, HIDDEN)

# GPU reference for attn_residual inputs
in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
q_proj_w = layer.self_attn.q_proj.weight.astype(mx.float32)
k_proj_w = layer.self_attn.k_proj.weight.astype(mx.float32)
v_proj_w = layer.self_attn.v_proj.weight.astype(mx.float32)
o_proj_w = layer.self_attn.o_proj.weight.astype(mx.float32)
q_norm_w = layer.self_attn.q_norm.weight.astype(mx.float32)
k_norm_w = layer.self_attn.k_norm.weight.astype(mx.float32)

noise_f = noise.astype(mx.float32)
ctx_f = context.astype(mx.float32)
normed = mlx_rmsnorm(noise_f, in_norm_w)

# Q: normed @ q_proj_w.T, then per-head rmsnorm
q = normed @ q_proj_w.T  # [1, SEQ_Q, N_HEADS*HEAD_DIM]
q_4d = q.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
ms = q_4d.mean(axis=-1, keepdims=True)
q_normed = q_4d * mx.rsqrt(((q_4d - ms)**2).mean(axis=-1, keepdims=True) + 1e-6) * q_norm_w.astype(mx.float32)

# K: concat ctx and noise K projections
k_ctx = ctx_f @ k_proj_w.T  # [1, CTX_LEN, N_KV_HEADS*HEAD_DIM]
k_noise = normed @ k_proj_w.T  # [1, SEQ_Q, N_KV_HEADS*HEAD_DIM]
k_all = mx.concatenate([k_ctx, k_noise], axis=1)  # [1, CTX_LEN+SEQ_Q, N_KV_HEADS*HEAD_DIM]
k_4d = k_all.reshape(1, CTX_LEN + SEQ_Q, N_KV_HEADS, HEAD_DIM)
ms_k = k_4d.mean(axis=-1, keepdims=True)
k_normed = k_4d * mx.rsqrt(((k_4d - ms_k)**2).mean(axis=-1, keepdims=True) + 1e-6) * k_norm_w.astype(mx.float32)

# V: concat ctx and noise V projections
v_ctx = ctx_f @ v_proj_w.T
v_noise = normed @ v_proj_w.T
v_all = mx.concatenate([v_ctx, v_noise], axis=1)

# RoPE
q_rope = mlx_rope(q_normed, offset=0)
k_rope = mlx_rope(k_normed, offset=0)

# GQA: tile K and V heads
gqa_ratio = N_HEADS // N_KV_HEADS
k_tiled = mx.repeat(k_rope, gqa_ratio, axis=2)  # [1, seq_k, N_HEADS, HEAD_DIM]
v_tiled = mx.repeat(v_all.reshape(1, -1, N_KV_HEADS, HEAD_DIM), gqa_ratio, axis=2)

# Attention
q_t = q_rope.transpose(0, 2, 1, 3)  # [1, N_HEADS, SEQ_Q, HEAD_DIM]
k_t = k_tiled.transpose(0, 2, 1, 3)  # [1, N_HEADS, CTX_LEN+SEQ_Q, HEAD_DIM]
v_t = v_tiled.transpose(0, 2, 1, 3)

scores = q_t @ k_t.transpose(0, 1, 3, 2) / mx.sqrt(mx.array(HEAD_DIM, dtype=mx.float32))
# No causal mask for block diffusion
probs = mx.softmax(scores, axis=-1)
attn_out = probs @ v_t  # [1, N_HEADS, SEQ_Q, HEAD_DIM]
attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(1, SEQ_Q, N_HEADS * HEAD_DIM)

# O projection + residual
o_proj = attn_flat @ o_proj_w.T
attn_res_expected = noise_f + o_proj

print(f"Expected attn_res sample: {attn_res_expected[0, 0, :4]}")
print(f"Expected o_proj sample: {o_proj[0, 0, :4]}")
print(f"Expected noise sample: {noise_f[0, 0, :4]}")
