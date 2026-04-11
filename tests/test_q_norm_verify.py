"""Verify q_proj + q_norm_4d produces correct per-head rmsnorm matching GPU."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, align_width
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
print("Building ANE model...")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)

mx.random.seed(42)
noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)

# GPU reference: full q_proj + per-head q_norm
layer = draft_model.layers[0]
hidden_f = noise_embedding.astype(mx.float32)
in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
ms = hidden_f.mean(axis=-1, keepdims=True)
normed = (hidden_f - ms) * mx.rsqrt(mx.mean((hidden_f - ms)**2, axis=-1, keepdims=True) + 1e-6)
normed = normed * in_norm_w
q_proj_w = layer.self_attn.q_proj.weight.astype(mx.float32)
q_after_proj = normed @ q_proj_w.T  # [1, SEQ_Q, N_HEADS*HEAD_DIM]

q_4d = q_after_proj.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
ms2 = q_4d.mean(axis=-1, keepdims=True)
normed2 = (q_4d - ms2) * mx.rsqrt(mx.mean((q_4d - ms2)**2, axis=-1, keepdims=True) + 1e-6)
q_norm_w = layer.self_attn.q_norm.weight.astype(mx.float32)
q_gpu = (normed2 * q_norm_w).reshape(1, SEQ_Q, N_HEADS * HEAD_DIM)

# ANE: q_proj + q_norm_4d
ane_model._write_mlx_2d(ane_model.b_noise, noise_embedding)
noise_data = ane_model.b_noise.read_f32()
ane_model.b_hidden.write_f32(noise_data)

ane_model.kernels['q_proj'].run_uncached(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj],
    [ane_model.b_q_out],
)
ane_model._flat_to_4d_norm(ane_model.b_q_out, N_HEADS, ane_model.w_sq, ane_model.b_q_norm_4d)
ane_model.kernels['q_norm_4d'].run_uncached(
    [ane_model.b_q_norm_4d, ane_model.w_l0_q_norm_4d],
    [ane_model.b_q_norm_4d],
)
ane_model._4d_norm_to_4d_heads(ane_model.b_q_norm_4d, N_HEADS, ane_model.w_sq, ane_model.b_q_4d)

# Read ANE output as 4D and convert to standard (non-interleaved) order
data_4d = ane_model.b_q_4d.read_f32()
w = ane_model.w_sq
q_ane_4d = mx.array(data_4d, dtype=mx.float32).reshape(1, N_HEADS, w, HEAD_DIM)
q_ane_4d = q_ane_4d[:, :, :SEQ_Q, :]  # [1, N_HEADS, SEQ_Q, HEAD_DIM] in interleaved order

# De-interleave: [d0,d64,d1,d65,...] -> [d0,d1,...,d63,d64,...,d127]
half = HEAD_DIM // 2
q_ane_std = mx.zeros_like(q_ane_4d)
for k in range(half):
    q_ane_std[:, :, :, k] = q_ane_4d[:, :, :, 2*k]
    q_ane_std[:, :, :, k+half] = q_ane_4d[:, :, :, 2*k+1]

# Transpose to [1, SEQ_Q, N_HEADS, HEAD_DIM] then flatten
q_ane_flat = q_ane_std.transpose(0, 2, 1, 3).reshape(1, SEQ_Q, N_HEADS * HEAD_DIM)

cos = cosine_sim(q_ane_flat, q_gpu)
print(f"\nq_proj + q_norm_4d (ANE) vs GPU per-head rmsnorm: cosine = {cos:.6f}")
if cos > 0.99:
    print("✓ PASS")
elif cos > 0.95:
    print("~ CLOSE (likely fp16 precision)")
else:
    print("✗ FAIL")
    for h in range(min(4, N_HEADS)):
        a_h = q_ane_flat[:, :, h*HEAD_DIM:(h+1)*HEAD_DIM]
        b_h = q_gpu[:, :, h*HEAD_DIM:(h+1)*HEAD_DIM]
        print(f"  Head {h}: cos={cosine_sim(a_h, b_h):.6f}")
