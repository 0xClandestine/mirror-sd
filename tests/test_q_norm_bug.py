"""Test: Verify per-head q_norm_4d kernel matches GPU reference.

The GPU applies nn.RMSNorm(head_dim=128) per-head.
The new q_norm_4d kernel uses [1, HEAD_DIM, 1, N_HEADS*w_sq] format
so rmsnorm reduces over channels (HEAD_DIM) per spatial position.
"""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, align_width
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

# Run q_proj (in_norm + q_proj, no q_norm)
print("Running q_proj...")
ane_model.kernels['q_proj'].run_uncached(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj],
    [ane_model.b_q_out],
)

# Rearrange to norm_4d format
ane_model._flat_to_4d_norm(ane_model.b_q_out, N_HEADS, ane_model.w_sq, ane_model.b_q_norm_4d)

# Run q_norm_4d
print("Running q_norm_4d...")
ane_model.kernels['q_norm_4d'].run_uncached(
    [ane_model.b_q_norm_4d, ane_model.w_l0_q_norm_4d],
    [ane_model.b_q_norm_4d],
)

# Read ANE output and convert back to flat for comparison
norm_data = ane_model.b_q_norm_4d.read_f32()
w = ane_model.w_sq
# Shape is [1, HEAD_DIM, 1, N_HEADS*w] — but read_f32 gives all elements including padding
# Total elements: HEAD_DIM * N_HEADS * w = 128 * 32 * 64 = 262144
arr = mx.array(norm_data, dtype=mx.float32).reshape(1, HEAD_DIM, N_HEADS, w)
arr_t = arr.transpose(0, 2, 3, 1)  # [1, N_HEADS, w, HEAD_DIM]
# Take only actual seq positions (first SEQ_Q of each head)
q_ane = arr_t[0, :, :SEQ_Q, :].reshape(1, SEQ_Q, N_HEADS * HEAD_DIM)

# GPU reference: per-head rmsnorm
layer = draft_model.layers[0]
in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
hidden_f = noise_embedding.astype(mx.float32)
ms = hidden_f.mean(axis=-1, keepdims=True)
diff = hidden_f - ms
sq = diff * diff
mean_sq = sq.mean(axis=-1, keepdims=True)
inv_std = mx.rsqrt(mean_sq + 1e-6)
normed = hidden_f * inv_std * in_norm_w
q_proj_w = layer.self_attn.q_proj.weight.astype(mx.float32)
q_after_proj = normed @ q_proj_w.T  # [1, SEQ_Q, N_HEADS*HEAD_DIM]

# Per-head rmsnorm (correct GPU reference)
q_4d = q_after_proj.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
ms2 = q_4d.mean(axis=-1, keepdims=True)
diff2 = q_4d - ms2
sq2 = diff2 * diff2
mean_sq2 = sq2.mean(axis=-1, keepdims=True)
inv_std2 = mx.rsqrt(mean_sq2 + 1e-6)
q_norm_w = layer.self_attn.q_norm.weight.astype(mx.float32)
q_per_head = q_4d * inv_std2 * q_norm_w
q_per_head_flat = q_per_head.reshape(1, SEQ_Q, N_HEADS * HEAD_DIM)

def cosine_sim(a, b):
    a_f = a.flatten()
    b_f = b.flatten()
    dot = float(mx.sum(a_f * b_f))
    na = float(mx.sqrt(mx.sum(a_f * a_f)))
    nb = float(mx.sqrt(mx.sum(b_f * b_f)))
    return dot / (na * nb + 1e-8)

cos = cosine_sim(q_ane[:, :, :N_HEADS * HEAD_DIM], q_per_head_flat)

print(f"\nANE q_norm_4d vs GPU per-head rmsnorm: cosine = {cos:.6f}")
if cos > 0.99:
    print("✓ PASS: q_norm_4d matches GPU per-head rmsnorm")
elif cos > 0.95:
    print("~ CLOSE")
else:
    print("✗ FAIL")

# Also compare per-head
for h in range(min(4, N_HEADS)):
    a_h = q_ane[:, :, h*HEAD_DIM:(h+1)*HEAD_DIM]
    b_h = q_per_head_flat[:, :, h*HEAD_DIM:(h+1)*HEAD_DIM]
    cos_h = cosine_sim(a_h, b_h)
    print(f"  Head {h}: cos={cos_h:.6f}")

