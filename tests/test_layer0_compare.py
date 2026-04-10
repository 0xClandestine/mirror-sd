"""Compare Layer 0 output between ANE pipeline and GPU reference."""

import sys, math
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
import mlx.nn as nn
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, INTERMEDIATE, TARGET_HIDDEN
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
    mean_sq = (diff * diff).mean(axis=-1, keepdims=True)
    return x * mx.rsqrt(mean_sq + eps) * weight

print("Loading...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)

mx.random.seed(42)
noise = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
target = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16)

# --- GPU reference for full Layer 0 ---
# Step 1: fc_norm to get context
fc_w = draft_model.fc.weight.astype(mx.float32)
hidden_norm_w = draft_model.hidden_norm.weight.astype(mx.float32)
ctx_f = target.astype(mx.float32) @ fc_w.T  # [1, CTX, HIDDEN]
context_gpu = mlx_rmsnorm(ctx_f, hidden_norm_w)

# Step 2: Layer 0
layer = draft_model.layers[0]
in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
q_proj_w = layer.self_attn.q_proj.weight.astype(mx.float32)
k_proj_w = layer.self_attn.k_proj.weight.astype(mx.float32)
v_proj_w = layer.self_attn.v_proj.weight.astype(mx.float32)
o_proj_w = layer.self_attn.o_proj.weight.astype(mx.float32)
q_norm_w = layer.self_attn.q_norm.weight.astype(mx.float32)
k_norm_w = layer.self_attn.k_norm.weight.astype(mx.float32)
post_norm_w = layer.post_attention_layernorm.weight.astype(mx.float32)
gate_w = layer.mlp.gate_proj.weight.astype(mx.float32)
up_w = layer.mlp.up_proj.weight.astype(mx.float32)
down_w = layer.mlp.down_proj.weight.astype(mx.float32)

noise_f = noise.astype(mx.float32)
normed = mlx_rmsnorm(noise_f, in_norm_w)

# Q projection + per-head rmsnorm
q = normed @ q_proj_w.T
q_4d = q.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
q_normed = mlx_rmsnorm(q_4d, q_norm_w)  # per-head

# K: ctx + noise, then per-head rmsnorm
k_ctx = context_gpu @ k_proj_w.T
k_noise = normed @ k_proj_w.T
k_all = mx.concatenate([k_ctx, k_noise], axis=1)
k_4d = k_all.reshape(1, CTX_LEN + SEQ_Q, N_KV_HEADS, HEAD_DIM)
k_normed = mlx_rmsnorm(k_4d, k_norm_w)

# V: ctx + noise
v_ctx = context_gpu @ v_proj_w.T
v_noise = normed @ v_proj_w.T
v_all = mx.concatenate([v_ctx, v_noise], axis=1)

# RoPE
rope_theta = 1000000.0
def apply_rope_mlx(x, rope_theta=1000000.0, offset=0):
    seq = x.shape[1]
    n_heads = x.shape[2]
    hd = x.shape[-1]
    half = hd // 2
    cos_data = []
    sin_data = []
    for p in range(seq):
        for d in range(half):
            freq = 1.0 / (rope_theta ** (2.0 * d / hd))
            angle = (offset + p) * freq
            cos_data.append(math.cos(angle))
            sin_data.append(math.sin(angle))
        for d in range(half):
            cos_data.append(1.0)
            sin_data.append(0.0)
    cos_v = mx.array(cos_data, dtype=mx.float32).reshape(1, seq, 1, hd)
    sin_v = mx.array(sin_data, dtype=mx.float32).reshape(1, seq, 1, hd)
    x_even = x[..., :half]
    x_odd = x[..., half:]
    x_rot = mx.concatenate([-x_odd, x_even], axis=-1)
    return x * cos_v + x_rot * sin_v

q_rope = apply_rope_mlx(q_normed, offset=0)
k_rope = apply_rope_mlx(k_normed, offset=0)

# GQA tile
gqa_ratio = N_HEADS // N_KV_HEADS
k_tiled = mx.repeat(k_rope, gqa_ratio, axis=2)
v_4d = v_all.reshape(1, CTX_LEN + SEQ_Q, N_KV_HEADS, HEAD_DIM)
v_tiled = mx.repeat(v_4d, gqa_ratio, axis=2)

# Attention (non-causal)
q_t = q_rope.transpose(0, 2, 1, 3)
k_t = k_tiled.transpose(0, 2, 1, 3)
v_t = v_tiled.transpose(0, 2, 1, 3)
scores = q_t @ k_t.transpose(0, 1, 3, 2) / mx.sqrt(mx.array(HEAD_DIM, dtype=mx.float32))
probs = mx.softmax(scores, axis=-1)
attn_out = probs @ v_t
attn_flat = attn_out.transpose(0, 2, 1, 3).reshape(1, SEQ_Q, N_HEADS * HEAD_DIM)
o_proj = attn_flat @ o_proj_w.T
attn_res_gpu = noise_f + o_proj

# FFN
ffn_normed = mlx_rmsnorm(attn_res_gpu, post_norm_w)
gate_out = ffn_normed @ gate_w.T
up_out = ffn_normed @ up_w.T
silu = gate_out * (1.0 / (1.0 + mx.exp(-gate_out)))
gate_mul = silu * up_out
down_out = gate_mul @ down_w.T
hidden_l0_gpu = attn_res_gpu + down_out

print(f"GPU Layer 0 output sample: {hidden_l0_gpu[0, 0, :4]}")
print(f"GPU attn_res sample:       {attn_res_gpu[0, 0, :4]}")
print(f"GPU o_proj sample:         {o_proj[0, 0, :4]}")

# --- ANE Layer 0 ---
ane_model._write_mlx_2d(ane_model.b_noise, noise)
ane_model._write_mlx_2d(ane_model.b_target, target)
ane_model.b_hidden.write_f32(ane_model.b_noise.read_f32())
ane_model._compute_rope(0)

ane_model.kernels['fc_norm'].run_uncached(
    [ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm],
    [ane_model.b_context],
)

# Run Layer 0 step by step
k = ane_model.kernels
k['q_kernel'].run_uncached(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj, ane_model.w_l0_q_norm],
    [ane_model.b_q_out],
)
k['k_proj_ctx'].run_uncached(
    [ane_model.b_context, ane_model.w_l0_k_proj],
    [ane_model.b_k_ctx],
)
k['k_proj_noise'].run_uncached(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_k_proj],
    [ane_model.b_k_noise],
)
k['k_concat'].run_uncached([ane_model.b_k_ctx, ane_model.b_k_noise], [ane_model.b_k_out])
k['k_norm'].run_uncached(
    [ane_model.b_k_out, ane_model.w_l0_k_norm],
    [ane_model.b_k_normed],
)
k['v_proj_ctx'].run_uncached(
    [ane_model.b_context, ane_model.w_l0_v_proj],
    [ane_model.b_v_ctx],
)
k['v_proj_noise'].run_uncached(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_v_proj],
    [ane_model.b_v_noise],
)
k['v_concat'].run_uncached([ane_model.b_v_ctx, ane_model.b_v_noise], [ane_model.b_v_out])
k['rope_q'].run_uncached(
    [ane_model.b_q_out, ane_model.b_cos_q, ane_model.b_sin_q],
    [ane_model.b_q_rope],
)
k['rope_k'].run_uncached(
    [ane_model.b_k_normed, ane_model.b_cos_k, ane_model.b_sin_k],
    [ane_model.b_k_rope],
)
k['gqa_tile'].run_uncached(
    [ane_model.b_k_rope, ane_model.b_v_out],
    [ane_model.b_kv_tiled],
)
k['attn_residual'].run_uncached(
    [ane_model.b_q_rope, ane_model.b_kv_tiled, ane_model.w_l0_o_proj, ane_model.b_hidden],
    [ane_model.b_attn_res],
)

# Compare attn_res
attn_res_ane = ane_model._read_mlx_2d(ane_model.b_attn_res, SEQ_Q, HIDDEN)
cos_attn = cosine_sim(attn_res_gpu, attn_res_ane)
print(f"\nattn_res cosine: {cos_attn:.6f}")
print(f"  ANE attn_res sample: {attn_res_ane[0, 0, :4]}")

k['ffn_residual'].run_uncached(
    [ane_model.b_attn_res, ane_model.w_l0_post_norm,
     ane_model.w_l0_gate, ane_model.w_l0_up, ane_model.w_l0_down],
    [ane_model.b_hidden],
)

hidden_l0_ane = ane_model._read_mlx_2d(ane_model.b_hidden, SEQ_Q, HIDDEN)
cos_l0 = cosine_sim(hidden_l0_gpu, hidden_l0_ane)
print(f"\nLayer 0 hidden cosine: {cos_l0:.6f}")
print(f"  GPU sample: {hidden_l0_gpu[0, 0, :4]}")
print(f"  ANE sample: {hidden_l0_ane[0, 0, :4]}")
