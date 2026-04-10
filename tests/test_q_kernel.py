"""Quick test: just q_kernel in isolation."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM
from mirror_sd.loader import load_dflash_model

SEQ_Q = 16
CTX_LEN = 64

print("Loading draft model...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")
print("Building ANE model...")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)

mx.random.seed(42)
noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)

# Write input
ane_model._write_mlx_2d(ane_model.b_noise, noise_embedding)
noise_data = ane_model.b_noise.read_f32()
ane_model.b_hidden.write_f32(noise_data)

# Run q_kernel
print("Running q_kernel...")
import time
t0 = time.time()
ane_model.kernels['q_kernel'].run(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj, ane_model.w_l0_q_norm],
    [ane_model.b_q_out],
)
t1 = time.time()
print(f"q_kernel done in {t1-t0:.3f}s")

# Read output
q_out = ane_model._read_mlx_2d(ane_model.b_q_out, SEQ_Q, N_HEADS * HEAD_DIM)
print(f"Output shape: {q_out.shape}")
print(f"Output sample: {q_out[0, 0, :8]}")

# GPU reference
layer = draft_model.layers[0]
in_norm_w = layer.input_layernorm.weight
hidden_f = noise_embedding.astype(mx.float32)
ms = hidden_f.mean(axis=-1, keepdims=True)
diff = hidden_f - ms
sq = diff * diff
mean_sq = sq.mean(axis=-1, keepdims=True)
inv_std = mx.rsqrt(mean_sq + 1e-6)
normed = hidden_f * inv_std * in_norm_w.astype(mx.float32)
q_proj_w = layer.self_attn.q_proj.weight.astype(mx.float32)
q_expected = normed @ q_proj_w.T
q_norm_w = layer.self_attn.q_norm.weight.astype(mx.float32)
# per-head rmsnorm
q_4d = q_expected.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
ms2 = q_4d.mean(axis=-1, keepdims=True)
diff2 = q_4d - ms2
sq2 = diff2 * diff2
mean_sq2 = sq2.mean(axis=-1, keepdims=True)
inv_std2 = mx.rsqrt(mean_sq2 + 1e-6)
q_normed = q_4d * inv_std2 * q_norm_w.astype(mx.float32)
q_normed = q_normed.reshape(1, SEQ_Q, N_HEADS * HEAD_DIM)

# Compare
a_f = q_normed.flatten()
b_f = q_out.astype(mx.float32).flatten()
dot = float(mx.sum(a_f * b_f))
na = float(mx.sqrt(mx.sum(a_f * a_f)))
nb = float(mx.sqrt(mx.sum(b_f * b_f)))
cos = dot / (na * nb + 1e-8)
print(f"Cosine similarity: {cos:.6f}")
if cos > 0.99:
    print("✓ PASS")
else:
    print(f"✗ FAIL")
    print(f"Expected sample: {q_normed[0, 0, :8]}")
