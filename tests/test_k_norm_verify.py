"""Verify K norm_4d and identify remaining error sources."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import math
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

def rmsnorm(x, weight, eps=1e-6):
    ms = x.mean(axis=-1, keepdims=True)
    diff = x - ms
    return diff * mx.rsqrt(mx.mean(diff**2, axis=-1, keepdims=True) + eps) * weight

print("Loading draft model...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")

mx.random.seed(42)
noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
target_hidden = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16)

layer = draft_model.layers[0]
hidden_f = noise_embedding.astype(mx.float32)
target_f = target_hidden.astype(mx.float32)
context = rmsnorm(draft_model.fc(target_f), draft_model.hidden_norm.weight)
normed = rmsnorm(hidden_f, layer.input_layernorm.weight)

# GPU K before rope
k_ctx_gpu = layer.self_attn.k_proj(context)
k_noise_gpu = layer.self_attn.k_proj(normed)
k_gpu_concat = mx.concatenate([k_ctx_gpu, k_noise_gpu], axis=1)  # [1, CTX+SEQ, KV_HEADS*HEAD_DIM]
k_gpu_before_rope = layer.self_attn.k_norm(
    k_gpu_concat.reshape(1, CTX_LEN+SEQ_Q, N_KV_HEADS, HEAD_DIM)
).transpose(0, 2, 1, 3)  # [1, N_KV_HEADS, CTX+SEQ, HEAD_DIM]

print(f"GPU K before rope: range=[{float(k_gpu_before_rope.min()):.2f}, {float(k_gpu_before_rope.max()):.2f}]")

# ANE
print("Building ANE model...")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)
ane_model._write_mlx_2d(ane_model.b_noise, noise_embedding)
ane_model._write_mlx_2d(ane_model.b_target, target_hidden)
noise_data = ane_model.b_noise.read_f32()
ane_model.b_hidden.write_f32(noise_data)
ane_model._compute_rope(0)

k = ane_model.kernels
k['fc_norm'].run_uncached([ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm], [ane_model.b_context])
k['q_proj'].run_uncached([ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj], [ane_model.b_q_out])
k['k_proj_ctx'].run_uncached([ane_model.b_context, ane_model.w_l0_k_proj], [ane_model.b_k_ctx])
k['k_proj_noise'].run_uncached([ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_k_proj], [ane_model.b_k_noise])
k['k_concat'].run_uncached([ane_model.b_k_ctx, ane_model.b_k_noise], [ane_model.b_k_out])

# Check K before norm (flat format, interleaved)
k_ane_before_norm = ane_model._read_mlx_2d(ane_model.b_k_out, CTX_LEN + SEQ_Q, N_KV_HEADS * HEAD_DIM).astype(mx.float32)
# De-interleave for comparison
half = HEAD_DIM // 2
k_ane_4d = k_ane_before_norm.reshape(1, CTX_LEN+SEQ_Q, N_KV_HEADS, HEAD_DIM)
k_ane_std = mx.zeros_like(k_ane_4d)
for kk in range(half):
    k_ane_std[:, :, :, kk] = k_ane_4d[:, :, :, 2*kk]
    k_ane_std[:, :, :, kk+half] = k_ane_4d[:, :, :, 2*kk+1]

cos_k_before_norm = cosine_sim(k_ane_std.transpose(0, 2, 1, 3), k_gpu_before_rope)
print(f"ANE K before norm (de-interleaved) vs GPU: cos={cos_k_before_norm:.6f}")

# K norm
ane_model._flat_to_4d_norm(ane_model.b_k_out, N_KV_HEADS, ane_model.w_kv, ane_model.b_k_norm_4d)
k['k_norm_4d'].run_uncached([ane_model.b_k_norm_4d, ane_model.w_l0_k_norm_4d], [ane_model.b_k_norm_4d])
ane_model._4d_norm_to_4d_heads(ane_model.b_k_norm_4d, N_KV_HEADS, ane_model.w_kv, ane_model.b_k_4d)

# Read K after norm (4D format, interleaved)
k_ane_after_norm_data = ane_model.b_k_4d.read_f32()
k_ane_after_norm = mx.array(k_ane_after_norm_data, dtype=mx.float32).reshape(1, N_KV_HEADS, ane_model.w_kv, HEAD_DIM)[:, :, :CTX_LEN+SEQ_Q, :]
k_ane_norm_std = mx.zeros_like(k_ane_after_norm)
for kk in range(half):
    k_ane_norm_std[:, :, :, kk] = k_ane_after_norm[:, :, :, 2*kk]
    k_ane_norm_std[:, :, :, kk+half] = k_ane_after_norm[:, :, :, 2*kk+1]

cos_k_after_norm = cosine_sim(k_ane_norm_std, k_gpu_before_rope)
print(f"ANE K after norm (de-interleaved) vs GPU: cos={cos_k_after_norm:.6f}")

# Apply rope
k['rope_k'].run_uncached([ane_model.b_k_4d, ane_model.b_cos_k, ane_model.b_sin_k], [ane_model.b_k_rope_4d])
k_ane_rope_data = ane_model.b_k_rope_4d.read_f32()
k_ane_rope = mx.array(k_ane_rope_data, dtype=mx.float32).reshape(1, N_KV_HEADS, ane_model.w_kv, HEAD_DIM)[:, :, :CTX_LEN+SEQ_Q, :]
k_ane_rope_std = mx.zeros_like(k_ane_rope)
for kk in range(half):
    k_ane_rope_std[:, :, :, kk] = k_ane_rope[:, :, :, 2*kk]
    k_ane_rope_std[:, :, :, kk+half] = k_ane_rope[:, :, :, 2*kk+1]

# GPU K after rope
k_ctx_rope = mx.fast.rope(k_gpu_before_rope[:, :, :CTX_LEN, :], HEAD_DIM, traditional=False, base=1000000.0, scale=1.0, offset=0)
k_noise_rope = mx.fast.rope(k_gpu_before_rope[:, :, CTX_LEN:, :], HEAD_DIM, traditional=False, base=1000000.0, scale=1.0, offset=CTX_LEN)
k_gpu_rope = mx.concatenate([k_ctx_rope, k_noise_rope], axis=2)

cos_k_after_rope = cosine_sim(k_ane_rope_std, k_gpu_rope)
print(f"ANE K after rope vs GPU: cos={cos_k_after_rope:.6f}")

# Per-KV-head comparison
for h in range(min(4, N_KV_HEADS)):
    a_h = k_ane_rope_std[:, h, :, :]
    b_h = k_gpu_rope[:, h, :, :]
    print(f"  KV Head {h}: cos={cosine_sim(a_h, b_h):.6f}")
