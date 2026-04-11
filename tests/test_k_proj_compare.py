"""Check k_proj_ctx and k_proj_noise separately against GPU."""

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

def rmsnorm(x, weight, eps=1e-6):
    ms = x.mean(axis=-1, keepdims=True)
    diff = x - ms
    return diff * mx.rsqrt(mx.mean(diff**2, axis=-1, keepdims=True) + eps) * weight

def interleave_heads(x, n_heads, head_dim, seq_len):
    x_4d = x.reshape(1, seq_len, n_heads, head_dim)
    half = head_dim // 2
    x_new = mx.zeros_like(x_4d)
    for k in range(half):
        x_new[:, :, :, 2*k] = x_4d[:, :, :, k]
        x_new[:, :, :, 2*k+1] = x_4d[:, :, :, k+half]
    return x_new.reshape(1, seq_len, n_heads * head_dim)

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

# GPU K projections
k_ctx_gpu = layer.self_attn.k_proj(context)  # [1, CTX_LEN, N_KV_HEADS*HEAD_DIM]
k_noise_gpu = layer.self_attn.k_proj(normed)  # [1, SEQ_Q, N_KV_HEADS*HEAD_DIM]

# Interleave for comparison with ANE
k_ctx_gpu_il = interleave_heads(k_ctx_gpu, N_KV_HEADS, HEAD_DIM, CTX_LEN)
k_noise_gpu_il = interleave_heads(k_noise_gpu, N_KV_HEADS, HEAD_DIM, SEQ_Q)

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

# k_proj_ctx
k['k_proj_ctx'].run_uncached([ane_model.b_context, ane_model.w_l0_k_proj], [ane_model.b_k_ctx])
k_ctx_ane = ane_model._read_mlx_2d(ane_model.b_k_ctx, CTX_LEN, N_KV_HEADS * HEAD_DIM).astype(mx.float32)
cos_k_ctx = cosine_sim(k_ctx_ane, k_ctx_gpu_il)
print(f"k_proj_ctx (ANE vs GPU interleaved): cos={cos_k_ctx:.6f}")

# k_proj_noise
k['k_proj_noise'].run_uncached([ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_k_proj], [ane_model.b_k_noise])
k_noise_ane = ane_model._read_mlx_2d(ane_model.b_k_noise, SEQ_Q, N_KV_HEADS * HEAD_DIM).astype(mx.float32)
cos_k_noise = cosine_sim(k_noise_ane, k_noise_gpu_il)
print(f"k_proj_noise (ANE vs GPU interleaved): cos={cos_k_noise:.6f}")

# Also check q_proj
k['q_proj'].run_uncached([ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj], [ane_model.b_q_out])
q_gpu = layer.self_attn.q_proj(normed)
q_gpu_il = interleave_heads(q_gpu, N_HEADS, HEAD_DIM, SEQ_Q)
q_ane = ane_model._read_mlx_2d(ane_model.b_q_out, SEQ_Q, N_HEADS * HEAD_DIM).astype(mx.float32)
cos_q = cosine_sim(q_ane, q_gpu_il)
print(f"q_proj (ANE vs GPU interleaved): cos={cos_q:.6f}")

# Check context (fc + hidden_norm)
context_ane = ane_model._read_mlx_2d(ane_model.b_context, CTX_LEN, HIDDEN).astype(mx.float32)
cos_ctx = cosine_sim(context_ane, context)
print(f"context/fc_norm (ANE vs GPU): cos={cos_ctx:.6f}")

# Check normed hidden
normed_ane = ane_model._read_mlx_2d(ane_model.b_hidden, SEQ_Q, HIDDEN).astype(mx.float32)
# ANE b_hidden has NOT been normed — it's the raw noise_embedding
# So we can't compare directly. Let's check the in_norm output separately.
