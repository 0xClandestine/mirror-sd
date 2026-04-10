"""Compare intermediate tensors between ANE and GPU for Layer 0 attention."""

import sys, math
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS
from mirror_sd.loader import load_dflash_model

SEQ_Q = 16
CTX_LEN = 64
TARGET_HIDDEN = 5 * HIDDEN

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

print("Loading...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)

mx.random.seed(42)
noise = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
target = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16)

# Run ANE up to rope_q/rope_k/gqa_tile
ane_model._write_mlx_2d(ane_model.b_noise, noise)
ane_model._write_mlx_2d(ane_model.b_target, target)
ane_model.b_hidden.write_f32(ane_model.b_noise.read_f32())
ane_model._compute_rope(0)

k = ane_model.kernels
k['fc_norm'].run_uncached([ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm], [ane_model.b_context])
k['q_kernel'].run_uncached([ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj, ane_model.w_l0_q_norm], [ane_model.b_q_out])
k['k_proj_ctx'].run_uncached([ane_model.b_context, ane_model.w_l0_k_proj], [ane_model.b_k_ctx])
k['k_proj_noise'].run_uncached([ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_k_proj], [ane_model.b_k_noise])
k['k_concat'].run_uncached([ane_model.b_k_ctx, ane_model.b_k_noise], [ane_model.b_k_out])
k['k_norm'].run_uncached([ane_model.b_k_out, ane_model.w_l0_k_norm], [ane_model.b_k_normed])
k['v_proj_ctx'].run_uncached([ane_model.b_context, ane_model.w_l0_v_proj], [ane_model.b_v_ctx])
k['v_proj_noise'].run_uncached([ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_v_proj], [ane_model.b_v_noise])
k['v_concat'].run_uncached([ane_model.b_v_ctx, ane_model.b_v_noise], [ane_model.b_v_out])
k['rope_q'].run_uncached([ane_model.b_q_out, ane_model.b_cos_q, ane_model.b_sin_q], [ane_model.b_q_rope])
k['rope_k'].run_uncached([ane_model.b_k_normed, ane_model.b_cos_k, ane_model.b_sin_k], [ane_model.b_k_rope])
k['gqa_tile'].run_uncached([ane_model.b_k_rope, ane_model.b_v_out], [ane_model.b_kv_tiled])

# Read ANE intermediates
q_rope_data = mx.array(ane_model.b_q_rope.read_f32(), dtype=mx.float32).reshape(1, N_HEADS, 64, HEAD_DIM)
q_rope_ane = q_rope_data[:, :, :SEQ_Q, :]  # [1, N_HEADS, SEQ_Q, HEAD_DIM]

k_rope_ane = ane_model._read_mlx_2d(ane_model.b_k_rope, CTX_LEN + SEQ_Q, N_KV_HEADS * HEAD_DIM)
v_out_ane = ane_model._read_mlx_2d(ane_model.b_v_out, CTX_LEN + SEQ_Q, N_KV_HEADS * HEAD_DIM)

# GPU reference
layer = draft_model.layers[0]
in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
q_proj_w = layer.self_attn.q_proj.weight.astype(mx.float32)
k_proj_w = layer.self_attn.k_proj.weight.astype(mx.float32)
v_proj_w = layer.self_attn.v_proj.weight.astype(mx.float32)
q_norm_w = layer.self_attn.q_norm.weight.astype(mx.float32)
k_norm_w = layer.self_attn.k_norm.weight.astype(mx.float32)

fc_w = draft_model.fc.weight.astype(mx.float32)
hidden_norm_w = draft_model.hidden_norm.weight.astype(mx.float32)
ctx_f = target.astype(mx.float32) @ fc_w.T
context_gpu = mlx_rmsnorm(ctx_f, hidden_norm_w)

noise_f = noise.astype(mx.float32)
normed = mlx_rmsnorm(noise_f, in_norm_w)
q = normed @ q_proj_w.T
q_4d = q.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
q_normed_gpu = mlx_rmsnorm(q_4d, q_norm_w)
k_ctx = context_gpu @ k_proj_w.T
k_noise = normed @ k_proj_w.T
k_all = mx.concatenate([k_ctx, k_noise], axis=1)
k_4d = k_all.reshape(1, CTX_LEN + SEQ_Q, N_KV_HEADS, HEAD_DIM)
k_normed_gpu = mlx_rmsnorm(k_4d, k_norm_w)
v_ctx = context_gpu @ v_proj_w.T
v_noise = normed @ v_proj_w.T
v_all_gpu = mx.concatenate([v_ctx, v_noise], axis=1)

q_rope_gpu = apply_rope_mlx(q_normed_gpu, offset=0)
k_rope_gpu = apply_rope_mlx(k_normed_gpu, offset=0)

print(f"\nQ after RoPE:")
print(f"  GPU shape: {q_rope_gpu.shape}, ANE shape: {q_rope_ane.shape}")
# GPU Q: [1, SEQ_Q, N_HEADS, HEAD_DIM], ANE Q: [1, N_HEADS, SEQ_Q, HEAD_DIM]
q_rope_ane_t = q_rope_ane.transpose(0, 2, 1, 3)  # [1, SEQ_Q, N_HEADS, HEAD_DIM]
cos_q = cosine_sim(q_rope_gpu, q_rope_ane_t)
print(f"  Cosine sim: {cos_q:.6f}")
print(f"  GPU sample [0,0,0,:4]: {q_rope_gpu[0, 0, 0, :4]}")
print(f"  ANE sample [0,0,0,:4]: {q_rope_ane_t[0, 0, 0, :4]}")

print(f"\nK after RoPE:")
cos_k = cosine_sim(k_rope_gpu, k_rope_ane.reshape(1, -1, N_KV_HEADS, HEAD_DIM))
print(f"  Cosine sim: {cos_k:.6f}")

print(f"\nV (concat):")
cos_v = cosine_sim(v_all_gpu, v_out_ane.reshape(1, -1, N_KV_HEADS * HEAD_DIM))
print(f"  Cosine sim: {cos_v:.6f}")
print(f"  GPU sample [0,0,:4]: {v_all_gpu[0, 0, :4]}")
print(f"  ANE sample [0,0,:4]: {v_out_ane[0, 0, :4]}")
