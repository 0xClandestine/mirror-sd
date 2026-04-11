"""Check k_concat and the flat→4d_norm→norm→4d chain for K."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, TARGET_HIDDEN, align_width
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

# GPU reference: k_concat + k_norm
k_ctx_gpu = layer.self_attn.k_proj(context)
k_noise_gpu = layer.self_attn.k_proj(normed)
k_concat_gpu = mx.concatenate([k_ctx_gpu, k_noise_gpu], axis=1)  # [1, CTX+SEQ, KV*HD]
k_4d_gpu = k_concat_gpu.reshape(1, CTX_LEN+SEQ_Q, N_KV_HEADS, HEAD_DIM)
k_normed_gpu = layer.self_attn.k_norm(k_4d_gpu)  # per-head norm
k_normed_gpu_t = k_normed_gpu.transpose(0, 2, 1, 3)  # [1, N_KV_HEADS, CTX+SEQ, HEAD_DIM]

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
k['k_proj_ctx'].run_uncached([ane_model.b_context, ane_model.w_l0_k_proj], [ane_model.b_k_ctx])
k['k_proj_noise'].run_uncached([ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_k_proj], [ane_model.b_k_noise])

# k_concat
k['k_concat'].run_uncached([ane_model.b_k_ctx, ane_model.b_k_noise], [ane_model.b_k_out])
k_concat_ane = ane_model._read_mlx_2d(ane_model.b_k_out, CTX_LEN + SEQ_Q, N_KV_HEADS * HEAD_DIM).astype(mx.float32)

# Interleave GPU for comparison
k_concat_gpu_il = interleave_heads(k_concat_gpu, N_KV_HEADS, HEAD_DIM, CTX_LEN + SEQ_Q)
cos_concat = cosine_sim(k_concat_ane, k_concat_gpu_il)
print(f"k_concat (ANE vs GPU interleaved): cos={cos_concat:.6f}")

# Now test the _flat_to_4d_norm + k_norm_4d + _4d_norm_to_4d_heads chain
# First, let's do it in Python for comparison
k_ane_deil = k_concat_ane.reshape(1, CTX_LEN+SEQ_Q, N_KV_HEADS, HEAD_DIM)
half = HEAD_DIM // 2
k_ane_std = mx.zeros_like(k_ane_deil)
for kk in range(half):
    k_ane_std[:, :, :, kk] = k_ane_deil[:, :, :, 2*kk]
    k_ane_std[:, :, :, kk+half] = k_ane_deil[:, :, :, 2*kk+1]

# Per-head rmsnorm in Python
k_ane_normed_py = rmsnorm(k_ane_std, layer.self_attn.k_norm.weight.astype(mx.float32))
cos_py_norm = cosine_sim(k_ane_normed_py.transpose(0, 2, 1, 3), k_normed_gpu_t)
print(f"Python per-head norm on ANE data vs GPU: cos={cos_py_norm:.6f}")

# ANE k_norm_4d chain
ane_model._flat_to_4d_norm(ane_model.b_k_out, N_KV_HEADS, ane_model.w_kv, ane_model.b_k_norm_4d)
k['k_norm_4d'].run_uncached([ane_model.b_k_norm_4d, ane_model.w_l0_k_norm_4d], [ane_model.b_k_norm_4d])
ane_model._4d_norm_to_4d_heads(ane_model.b_k_norm_4d, N_KV_HEADS, ane_model.w_kv, ane_model.b_k_4d)

# Read ANE K after norm
k_ane_norm_data = ane_model.b_k_4d.read_f32()
k_ane_norm_4d = mx.array(k_ane_norm_data, dtype=mx.float32).reshape(1, N_KV_HEADS, ane_model.w_kv, HEAD_DIM)[:, :, :CTX_LEN+SEQ_Q, :]
k_ane_norm_std = mx.zeros_like(k_ane_norm_4d)
for kk in range(half):
    k_ane_norm_std[:, :, :, kk] = k_ane_norm_4d[:, :, :, 2*kk]
    k_ane_norm_std[:, :, :, kk+half] = k_ane_norm_4d[:, :, :, 2*kk+1]

cos_ane_norm = cosine_sim(k_ane_norm_std, k_normed_gpu_t)
print(f"ANE k_norm_4d chain vs GPU: cos={cos_ane_norm:.6f}")

# Also compare ANE norm vs Python norm on same data
cos_ane_vs_py = cosine_sim(k_ane_norm_std, k_ane_normed_py.transpose(0, 2, 1, 3))
print(f"ANE k_norm_4d vs Python per-head norm (same input): cos={cos_ane_vs_py:.6f}")
