"""Compare ANE attention output against GPU for layer 0."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
import mlx.nn as nn
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, TARGET_HIDDEN
from mirror_sd.dflash import DFlashDraftModel
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

print("Loading draft model...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")

mx.random.seed(42)
noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
target_hidden = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16)

# GPU reference: manually compute layer 0 attention
layer = draft_model.layers[0]
hidden_f = noise_embedding.astype(mx.float32)
target_f = target_hidden.astype(mx.float32)

def rmsnorm(x, weight, eps=1e-6):
    ms = x.mean(axis=-1, keepdims=True)
    diff = x - ms
    return diff * mx.rsqrt(mx.mean(diff**2, axis=-1, keepdims=True) + eps) * weight

# fc + hidden_norm
context = rmsnorm(draft_model.fc(target_f), draft_model.hidden_norm.weight)

# input_layernorm
normed = rmsnorm(hidden_f, layer.input_layernorm.weight)

# Q, K, V projections
q_gpu = layer.self_attn.q_proj(normed)  # [1, SEQ_Q, N_HEADS*HEAD_DIM]
q_gpu = layer.self_attn.q_norm(q_gpu.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)).transpose(0, 2, 1, 3)

k_ctx = layer.self_attn.k_proj(context)
k_noise = layer.self_attn.k_proj(normed)
k_gpu = mx.concatenate([k_ctx, k_noise], axis=1)
k_gpu = layer.self_attn.k_norm(k_gpu.reshape(1, CTX_LEN + SEQ_Q, N_KV_HEADS, HEAD_DIM)).transpose(0, 2, 1, 3)

v_ctx = layer.self_attn.v_proj(context)
v_noise = layer.self_attn.v_proj(normed)
v_gpu = mx.concatenate([v_ctx, v_noise], axis=1)
v_gpu = v_gpu.reshape(1, CTX_LEN + SEQ_Q, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

# RoPE
q_gpu = mx.fast.rope(q_gpu, HEAD_DIM, traditional=False, base=1000000.0, scale=1.0, offset=CTX_LEN)
k_ctx_rope = mx.fast.rope(k_gpu[:, :, :CTX_LEN, :], HEAD_DIM, traditional=False, base=1000000.0, scale=1.0, offset=0)
k_noise_rope = mx.fast.rope(k_gpu[:, :, CTX_LEN:, :], HEAD_DIM, traditional=False, base=1000000.0, scale=1.0, offset=CTX_LEN)
k_gpu = mx.concatenate([k_ctx_rope, k_noise_rope], axis=2)

# GQA tile
k_tiled = mx.repeat(k_gpu, N_HEADS // N_KV_HEADS, axis=1)
v_tiled = mx.repeat(v_gpu, N_HEADS // N_KV_HEADS, axis=1)

# SDPA
attn_out_gpu = mx.fast.scaled_dot_product_attention(q_gpu, k_tiled, v_tiled, scale=HEAD_DIM**-0.5)
attn_flat_gpu = attn_out_gpu.transpose(0, 2, 1, 3).reshape(1, SEQ_Q, N_HEADS * HEAD_DIM)

print(f"GPU attn_out range: [{float(attn_out_gpu.min()):.2f}, {float(attn_out_gpu.max()):.2f}]")
print(f"GPU attn_flat range: [{float(attn_flat_gpu.min()):.2f}, {float(attn_flat_gpu.max()):.2f}]")

# ANE model
print("\nBuilding ANE model...")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)

ane_model._write_mlx_2d(ane_model.b_noise, noise_embedding)
ane_model._write_mlx_2d(ane_model.b_target, target_hidden)
noise_data = ane_model.b_noise.read_f32()
ane_model.b_hidden.write_f32(noise_data)
ane_model._compute_rope(0)

k = ane_model.kernels
k['fc_norm'].run_uncached([ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm], [ane_model.b_context])

p = "l0_"
k['q_proj'].run_uncached([ane_model.b_hidden, getattr(ane_model, f"w_{p}in_norm"), getattr(ane_model, f"w_{p}q_proj")], [ane_model.b_q_out])
ane_model._flat_to_4d_norm(ane_model.b_q_out, N_HEADS, ane_model.w_sq, ane_model.b_q_norm_4d)
k['q_norm_4d'].run_uncached([ane_model.b_q_norm_4d, getattr(ane_model, f"w_{p}q_norm_4d")], [ane_model.b_q_norm_4d])
ane_model._4d_norm_to_4d_heads(ane_model.b_q_norm_4d, N_HEADS, ane_model.w_sq, ane_model.b_q_4d)

k['k_proj_ctx'].run_uncached([ane_model.b_context, getattr(ane_model, f"w_{p}k_proj")], [ane_model.b_k_ctx])
k['k_proj_noise'].run_uncached([ane_model.b_hidden, getattr(ane_model, f"w_{p}in_norm"), getattr(ane_model, f"w_{p}k_proj")], [ane_model.b_k_noise])
k['k_concat'].run_uncached([ane_model.b_k_ctx, ane_model.b_k_noise], [ane_model.b_k_out])
ane_model._flat_to_4d_norm(ane_model.b_k_out, N_KV_HEADS, ane_model.w_kv, ane_model.b_k_norm_4d)
k['k_norm_4d'].run_uncached([ane_model.b_k_norm_4d, getattr(ane_model, f"w_{p}k_norm_4d")], [ane_model.b_k_norm_4d])
ane_model._4d_norm_to_4d_heads(ane_model.b_k_norm_4d, N_KV_HEADS, ane_model.w_kv, ane_model.b_k_4d)

k['rope_q'].run_uncached([ane_model.b_q_4d, ane_model.b_cos_q, ane_model.b_sin_q], [ane_model.b_q_4d])
k['rope_k'].run_uncached([ane_model.b_k_4d, ane_model.b_cos_k, ane_model.b_sin_k], [ane_model.b_k_rope_4d])

k['v_proj_ctx'].run_uncached([ane_model.b_context, getattr(ane_model, f"w_{p}v_proj")], [ane_model.b_v_ctx])
k['v_proj_noise'].run_uncached([ane_model.b_hidden, getattr(ane_model, f"w_{p}in_norm"), getattr(ane_model, f"w_{p}v_proj")], [ane_model.b_v_noise])
k['v_concat'].run_uncached([ane_model.b_v_ctx, ane_model.b_v_noise], [ane_model.b_v_out])
ane_model._flat_to_4d_heads(ane_model.b_v_out, N_KV_HEADS, ane_model.w_kv, ane_model.b_v_4d)

# Compare Q after rope
q_ane_data = ane_model.b_q_4d.read_f32()
q_ane_4d = mx.array(q_ane_data, dtype=mx.float32).reshape(1, N_HEADS, ane_model.w_sq, HEAD_DIM)[:, :, :SEQ_Q, :]
# De-interleave
half = HEAD_DIM // 2
q_ane_std = mx.zeros_like(q_ane_4d)
for kk in range(half):
    q_ane_std[:, :, :, kk] = q_ane_4d[:, :, :, 2*kk]
    q_ane_std[:, :, :, kk+half] = q_ane_4d[:, :, :, 2*kk+1]
cos_q = cosine_sim(q_ane_std, q_gpu)
print(f"\nQ after rope: cos={cos_q:.6f}")

# Compare K after rope
k_ane_data = ane_model.b_k_rope_4d.read_f32()
k_ane_4d = mx.array(k_ane_data, dtype=mx.float32).reshape(1, N_KV_HEADS, ane_model.w_kv, HEAD_DIM)[:, :, :CTX_LEN+SEQ_Q, :]
k_ane_std = mx.zeros_like(k_ane_4d)
for kk in range(half):
    k_ane_std[:, :, :, kk] = k_ane_4d[:, :, :, 2*kk]
    k_ane_std[:, :, :, kk+half] = k_ane_4d[:, :, :, 2*kk+1]
cos_k = cosine_sim(k_ane_std, k_gpu)
print(f"K after rope: cos={cos_k:.6f}")

# Compare V
v_ane_data = ane_model.b_v_4d.read_f32()
v_ane_4d = mx.array(v_ane_data, dtype=mx.float32).reshape(1, N_KV_HEADS, ane_model.w_kv, HEAD_DIM)[:, :, :CTX_LEN+SEQ_Q, :]
cos_v = cosine_sim(v_ane_4d, v_gpu)
print(f"V (before gqa_tile): cos={cos_v:.6f}")

# GQA tile + SDPA
k['gqa_tile'].run_uncached([ane_model.b_k_rope_4d, ane_model.b_v_4d], [ane_model.b_kv_tiled])
k['attn_out'].run_uncached([ane_model.b_q_4d, ane_model.b_kv_tiled], [ane_model.b_attn_out])

# Read and compare attn_out
attn_ane_data = ane_model.b_attn_out.read_f32()
attn_ane_4d = mx.array(attn_ane_data, dtype=mx.float32).reshape(1, N_HEADS, ane_model.w_sq, HEAD_DIM)[:, :, :SEQ_Q, :]
# attn_out is NOT interleaved (V is original format, Q@K^T is invariant)
cos_attn = cosine_sim(attn_ane_4d, attn_out_gpu)
print(f"\nattn_out: cos={cos_attn:.6f}")
print(f"ANE attn range: [{float(attn_ane_4d.min()):.2f}, {float(attn_ane_4d.max()):.2f}]")
print(f"GPU attn range: [{float(attn_out_gpu.min()):.2f}, {float(attn_out_gpu.max()):.2f}]")

# Per-head comparison
for h in range(min(4, N_HEADS)):
    a_h = attn_ane_4d[:, h, :, :]
    b_h = attn_out_gpu[:, h, :, :]
    print(f"  Head {h}: cos={cosine_sim(a_h, b_h):.6f}")
