"""Debug RoPE: compare Q before and after RoPE between ANE and GPU."""

import sys, math
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
    mean_sq = (diff * diff).mean(axis=-1, keepdims=True)
    return x * mx.rsqrt(mean_sq + eps) * weight

print("Loading...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)

mx.random.seed(42)
noise = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)

# Run ANE q_kernel
ane_model._write_mlx_2d(ane_model.b_noise, noise)
ane_model.b_hidden.write_f32(ane_model.b_noise.read_f32())
ane_model.kernels['q_kernel'].run_uncached(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj, ane_model.w_l0_q_norm],
    [ane_model.b_q_out],
)

# Read Q before RoPE: b_q_out is [1, N_HEADS*HEAD_DIM, 1, w_sq]
q_before_rope_ane = ane_model._read_mlx_2d(ane_model.b_q_out, SEQ_Q, N_HEADS * HEAD_DIM)
print(f"ANE Q before RoPE shape: {q_before_rope_ane.shape}")
print(f"ANE Q before RoPE sample [0,0,:4]: {q_before_rope_ane[0, 0, :4]}")

# GPU reference: Q before RoPE
layer = draft_model.layers[0]
in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
q_proj_w = layer.self_attn.q_proj.weight.astype(mx.float32)
q_norm_w = layer.self_attn.q_norm.weight.astype(mx.float32)

noise_f = noise.astype(mx.float32)
normed = mlx_rmsnorm(noise_f, in_norm_w)
q = normed @ q_proj_w.T  # [1, SEQ_Q, N_HEADS*HEAD_DIM]
q_4d = q.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
ms = q_4d.mean(axis=-1, keepdims=True)
q_normed = q_4d * mx.rsqrt(((q_4d - ms)**2).mean(axis=-1, keepdims=True) + 1e-6) * q_norm_w

print(f"GPU Q before RoPE shape: {q_normed.shape}")
print(f"GPU Q before RoPE sample [0,0,0,:4]: {q_normed[0, 0, 0, :4]}")

cos_q = cosine_sim(q_normed.reshape(1, SEQ_Q, -1), q_before_rope_ane)
print(f"Q before RoPE cosine: {cos_q:.6f}")

# Now run ANE RoPE
ane_model._compute_rope(0)
ane_model.kernels['rope_q'].run_uncached(
    [ane_model.b_q_out, ane_model.b_cos_q, ane_model.b_sin_q],
    [ane_model.b_q_rope],
)

# Read Q after RoPE: b_q_rope is [1, N_HEADS, w_sq, HEAD_DIM]
q_after_data = mx.array(ane_model.b_q_rope.read_f32(), dtype=mx.float32).reshape(1, N_HEADS, 64, HEAD_DIM)
q_after_ane = q_after_data[:, :, :SEQ_Q, :]  # [1, N_HEADS, SEQ_Q, HEAD_DIM]
print(f"\nANE Q after RoPE shape: {q_after_ane.shape}")
print(f"ANE Q after RoPE sample [0,0,:4]: {q_after_ane[0, 0, :4]}")

# GPU RoPE
def apply_rope_mlx(x, rope_theta=1000000.0, offset=0):
    seq = x.shape[1]
    hd = x.shape[-1]
    half = hd // 2
    cos_data, sin_data = [], []
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

q_after_gpu = apply_rope_mlx(q_normed, offset=CTX_LEN)
# GPU: [1, SEQ_Q, N_HEADS, HEAD_DIM], ANE: [1, N_HEADS, SEQ_Q, HEAD_DIM]
q_after_ane_t = q_after_ane.transpose(0, 2, 1, 3)
cos_q_rope = cosine_sim(q_after_gpu, q_after_ane_t)
print(f"Q after RoPE cosine: {cos_q_rope:.6f}")
print(f"GPU Q after RoPE sample [0,0,0,:4]: {q_after_gpu[0, 0, 0, :4]}")
print(f"ANE Q after RoPE sample [0,0,0,:4]: {q_after_ane_t[0, 0, 0, :4]}")

# Check if the issue is the RoPE cos/sin values themselves
# Read ANE cos/sin
cos_q_data = ane_model.b_cos_q.read_f32()
sin_q_data = ane_model.b_sin_q.read_f32()
print(f"\nANE cos_q shape: {ane_model.b_cos_q.shape}")
print(f"ANE cos_q first 8: {cos_q_data[:8]}")
print(f"ANE sin_q first 8: {sin_q_data[:8]}")

# GPU cos/sin for position 0
gpu_cos_p0 = [math.cos(0 * 1.0 / (1000000.0 ** (2.0 * d / 128))) for d in range(8)]
gpu_sin_p0 = [math.sin(0 * 1.0 / (1000000.0 ** (2.0 * d / 128))) for d in range(8)]
print(f"GPU cos for pos=0 first 8: {gpu_cos_p0}")
print(f"GPU sin for pos=0 first 8: {gpu_sin_p0}")
