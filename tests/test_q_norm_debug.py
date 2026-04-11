"""Debug: verify _flat_to_4d_norm rearrangement and q_norm_4d step by step."""

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

# GPU reference: compute q_after_proj (before norm)
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

# Run q_proj (ANE)
ane_model.kernels['q_proj'].run_uncached(
    [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj],
    [ane_model.b_q_out],
)

# Read ANE q_proj output
q_ane_flat = ane_model._read_mlx_2d(ane_model.b_q_out, SEQ_Q, N_HEADS * HEAD_DIM).astype(mx.float32)

def cosine_sim(a, b):
    a_f = a.astype(mx.float32).flatten()
    b_f = b.astype(mx.float32).flatten()
    dot = float(mx.sum(a_f * b_f))
    na = float(mx.sqrt(mx.sum(a_f * a_f)))
    nb = float(mx.sqrt(mx.sum(b_f * b_f)))
    return dot / (na * nb + 1e-8)

# Compare q_proj output (should match but be in interleaved order)
# The ANE q_proj uses interleaved weights, so output is interleaved
# GPU output is in standard order. Let's interleave the GPU output for comparison
def interleave_heads(x, n_heads, head_dim, seq_len=None):
    """Interleave head dims: [d0..d63,d64..d127] -> [d0,d64,d1,d65,...] per head"""
    if seq_len is None:
        seq_len = x.shape[1]
    x_4d = x.reshape(1, seq_len, n_heads, head_dim)
    half = head_dim // 2
    x_new = mx.zeros_like(x_4d)
    for k in range(half):
        x_new[:, :, :, 2*k] = x_4d[:, :, :, k]
        x_new[:, :, :, 2*k+1] = x_4d[:, :, :, k+half]
    return x_new.reshape(1, seq_len, n_heads * head_dim)

q_gpu_interleaved = interleave_heads(q_after_proj, N_HEADS, HEAD_DIM)
cos_proj = cosine_sim(q_ane_flat, q_gpu_interleaved)
print(f"q_proj output (ANE vs GPU interleaved): cosine = {cos_proj:.6f}")

# Now test _flat_to_4d_norm rearrangement
ane_model._flat_to_4d_norm(ane_model.b_q_out, N_HEADS, ane_model.w_sq, ane_model.b_q_norm_4d)

# Read the rearranged data
norm_4d_data = ane_model.b_q_norm_4d.read_f32()
w = ane_model.w_sq
arr_norm = mx.array(norm_4d_data, dtype=mx.float32).reshape(1, HEAD_DIM, N_HEADS, w)

# Expected: for head h, seq pos s, dim d_interleaved:
# arr_norm[0, d, h, s] should equal q_ane_flat[0, s, h*HEAD_DIM + d]
# (where d is in interleaved order)

# Let's check a few values
print(f"\nVerifying _flat_to_4d_norm rearrangement:")
for h in range(min(2, N_HEADS)):
    for s in range(min(3, SEQ_Q)):
        for d in [0, 1, 2, 63, 64, 127]:
            ane_val = float(arr_norm[0, d, h, s])
            flat_idx = h * HEAD_DIM + d
            expected = float(q_ane_flat[0, s, flat_idx])
            if abs(ane_val - expected) > 0.01:
                print(f"  MISMATCH: h={h}, s={s}, d={d}: ane={ane_val:.4f}, expected={expected:.4f}")
                break
        else:
            continue
        break
    else:
        continue
    break
else:
    print("  All spot checks passed!")

# Now do Python-only per-head rmsnorm on the norm_4d data to verify the norm kernel should work
q_4d_from_norm = arr_norm.transpose(0, 2, 3, 1)[:, :, :SEQ_Q, :]  # [1, N_HEADS, SEQ_Q, HEAD_DIM]
# Apply per-head rmsnorm manually
ms2 = q_4d_from_norm.mean(axis=-1, keepdims=True)
diff2 = q_4d_from_norm - ms2
sq2 = diff2 * diff2
mean_sq2 = sq2.mean(axis=-1, keepdims=True)
inv_std2 = mx.rsqrt(mean_sq2 + 1e-6)
q_norm_w = layer.self_attn.q_norm.weight.astype(mx.float32)
q_normed_python = q_4d_from_norm * inv_std2 * q_norm_w

# Also apply rmsnorm the ANE way (over channels of norm_4d format)
arr_for_ane_norm = arr_norm[:, :, :, :SEQ_Q]  # truncate to actual seq len
# rmsnorm over channels: for each (h, s), reduce over d
ms_ane = arr_for_ane_norm.mean(axis=1, keepdims=True)  # [1, 1, N_HEADS, SEQ_Q]
diff_ane = arr_for_ane_norm - ms_ane
sq_ane = diff_ane * diff_ane
mean_sq_ane = sq_ane.mean(axis=1, keepdims=True)
inv_std_ane = mx.rsqrt(mean_sq_ane + 1e-6)
# Weight: [HEAD_DIM] repeated for each (h, s)
q_norm_w_il = interleave_heads(q_norm_w.reshape(1, 1, HEAD_DIM), 1, HEAD_DIM, seq_len=1).reshape(HEAD_DIM)
q_norm_w_expanded = q_norm_w_il.reshape(1, HEAD_DIM, 1, 1)
q_normed_ane_style = arr_for_ane_norm * inv_std_ane * q_norm_w_expanded

# Compare the two
cos_norm_methods = cosine_sim(
    q_normed_python.reshape(1, -1),
    q_normed_ane_style.transpose(0, 2, 3, 1).reshape(1, -1)
)
print(f"\nPython per-head vs ANE-style norm (both on rearranged data): cosine = {cos_norm_methods:.6f}")

# Now run the actual ANE q_norm_4d kernel and compare
ane_model.kernels['q_norm_4d'].run_uncached(
    [ane_model.b_q_norm_4d, ane_model.w_l0_q_norm_4d],
    [ane_model.b_q_norm_4d],
)

norm_out_data = ane_model.b_q_norm_4d.read_f32()
arr_out = mx.array(norm_out_data, dtype=mx.float32).reshape(1, HEAD_DIM, N_HEADS, w)
q_ane_normed = arr_out.transpose(0, 2, 3, 1)[:, :, :SEQ_Q, :]  # [1, N_HEADS, SEQ_Q, HEAD_DIM]

# Compare with Python per-head norm (both on same interleaved data)
cos_ane_vs_python = cosine_sim(
    q_ane_normed.reshape(1, -1),
    q_normed_python.reshape(1, -1)
)
print(f"ANE q_norm_4d vs Python per-head norm: cosine = {cos_ane_vs_python:.6f}")

# Also compare with GPU reference (in standard, non-interleaved order)
# De-interleave both
def deinterleave_heads(x, n_heads, head_dim):
    half = head_dim // 2
    x_4d = x.reshape(1, n_heads, -1, head_dim)
    x_new = mx.zeros_like(x_4d)
    for k in range(half):
        x_new[:, :, :, k] = x_4d[:, :, :, 2*k]
        x_new[:, :, :, k+half] = x_4d[:, :, :, 2*k+1]
    return x_new.reshape(1, -1, n_heads * head_dim)

q_ane_deinterleaved = deinterleave_heads(q_ane_normed, N_HEADS, HEAD_DIM)

# GPU reference with per-head rmsnorm
q_4d_gpu = q_after_proj.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
ms_gpu = q_4d_gpu.mean(axis=-1, keepdims=True)
diff_gpu = q_4d_gpu - ms_gpu
sq_gpu = diff_gpu * diff_gpu
mean_sq_gpu = sq_gpu.mean(axis=-1, keepdims=True)
inv_std_gpu = mx.rsqrt(mean_sq_gpu + 1e-6)
q_gpu_normed = q_4d_gpu * inv_std_gpu * q_norm_w
q_gpu_normed_flat = q_gpu_normed.reshape(1, SEQ_Q, N_HEADS * HEAD_DIM)

cos_final = cosine_sim(q_ane_deinterleaved, q_gpu_normed_flat)
print(f"ANE (de-interleaved) vs GPU per-head norm: cosine = {cos_final:.6f}")
