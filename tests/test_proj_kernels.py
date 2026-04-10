"""Test k_proj_ctx, k_proj_noise, v_proj_ctx, v_proj_noise in isolation."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, TARGET_HIDDEN
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

print("Loading...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)

mx.random.seed(42)
noise = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
target = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16)

# Get context from fc_norm
ane_model._write_mlx_2d(ane_model.b_target, target)
ane_model.kernels['fc_norm'].run(
    [ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm],
    [ane_model.b_context],
)
context = ane_model._read_mlx_2d(ane_model.b_context, CTX_LEN, HIDDEN)

layer = draft_model.layers[0]

# --- k_proj_ctx ---
k_proj_w = layer.self_attn.k_proj.weight.astype(mx.float32)
k_ctx_expected = context.astype(mx.float32) @ k_proj_w.T
ane_model.kernels['k_proj_ctx'].run(
    [ane_model.b_context, ane_model.w_l0_k_proj],
    [ane_model.b_k_ctx],
)
k_ctx_ane = ane_model._read_mlx_2d(ane_model.b_k_ctx, CTX_LEN, N_KV_HEADS * HEAD_DIM)
print(f"k_proj_ctx: cosine={cosine_sim(k_ctx_expected, k_ctx_ane):.6f}")

# --- k_proj_noise ---
in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
normed = mlx_rmsnorm(noise.astype(mx.float32), in_norm_w)
k_noise_expected = normed @ k_proj_w.T
ane_model._write_mlx_2d(ane_model.b_noise, noise)
ane_model.b_hidden.write_f32(ane_model.b_noise.read_f32())
ane_model.kernels['k_proj_noise'].run(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_k_proj],
    [ane_model.b_k_noise],
)
k_noise_ane = ane_model._read_mlx_2d(ane_model.b_k_noise, SEQ_Q, N_KV_HEADS * HEAD_DIM)
print(f"k_proj_noise: cosine={cosine_sim(k_noise_expected, k_noise_ane):.6f}")

# --- v_proj_ctx ---
v_proj_w = layer.self_attn.v_proj.weight.astype(mx.float32)
v_ctx_expected = context.astype(mx.float32) @ v_proj_w.T
ane_model.kernels['v_proj_ctx'].run(
    [ane_model.b_context, ane_model.w_l0_v_proj],
    [ane_model.b_v_ctx],
)
v_ctx_ane = ane_model._read_mlx_2d(ane_model.b_v_ctx, CTX_LEN, N_KV_HEADS * HEAD_DIM)
print(f"v_proj_ctx: cosine={cosine_sim(v_ctx_expected, v_ctx_ane):.6f}")

# --- v_proj_noise ---
v_noise_expected = normed @ v_proj_w.T
ane_model._write_mlx_2d(ane_model.b_noise, noise)
ane_model.b_hidden.write_f32(ane_model.b_noise.read_f32())
ane_model.kernels['v_proj_noise'].run(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_v_proj],
    [ane_model.b_v_noise],
)
v_noise_ane = ane_model._read_mlx_2d(ane_model.b_v_noise, SEQ_Q, N_KV_HEADS * HEAD_DIM)
print(f"v_proj_noise: cosine={cosine_sim(v_noise_expected, v_noise_ane):.6f}")
