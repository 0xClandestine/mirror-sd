"""Compare Q and K before and after RoPE in detail."""

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

# GPU Q before rope (standard order, no interleaving)
q_gpu_before_rope = layer.self_attn.q_norm(
    layer.self_attn.q_proj(normed).reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
).transpose(0, 2, 1, 3)  # [1, N_HEADS, SEQ_Q, HEAD_DIM]

print(f"GPU Q before rope: shape={q_gpu_before_rope.shape}, range=[{float(q_gpu_before_rope.min()):.2f}, {float(q_gpu_before_rope.max()):.2f}]")

# Build ANE
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
ane_model._flat_to_4d_norm(ane_model.b_q_out, N_HEADS, ane_model.w_sq, ane_model.b_q_norm_4d)
k['q_norm_4d'].run_uncached([ane_model.b_q_norm_4d, ane_model.w_l0_q_norm_4d], [ane_model.b_q_norm_4d])
ane_model._4d_norm_to_4d_heads(ane_model.b_q_norm_4d, N_HEADS, ane_model.w_sq, ane_model.b_q_4d)

# Read ANE Q before rope
q_ane_data = ane_model.b_q_4d.read_f32()
q_ane_4d = mx.array(q_ane_data, dtype=mx.float32).reshape(1, N_HEADS, ane_model.w_sq, HEAD_DIM)[:, :, :SEQ_Q, :]
# De-interleave
half = HEAD_DIM // 2
q_ane_std = mx.zeros_like(q_ane_4d)
for kk in range(half):
    q_ane_std[:, :, :, kk] = q_ane_4d[:, :, :, 2*kk]
    q_ane_std[:, :, :, kk+half] = q_ane_4d[:, :, :, 2*kk+1]

cos_q_before = cosine_sim(q_ane_std, q_gpu_before_rope)
print(f"ANE Q before rope (de-interleaved) vs GPU: cos={cos_q_before:.6f}")

# Now apply RoPE manually in Python on the de-interleaved data (matching MLX behavior)
# and compare with ANE's interleaved RoPE result
k['rope_q'].run_uncached([ane_model.b_q_4d, ane_model.b_cos_q, ane_model.b_sin_q], [ane_model.b_q_rope_4d])

q_ane_rope_data = ane_model.b_q_rope_4d.read_f32()
q_ane_rope_4d = mx.array(q_ane_rope_data, dtype=mx.float32).reshape(1, N_HEADS, ane_model.w_sq, HEAD_DIM)[:, :, :SEQ_Q, :]
# De-interleave
q_ane_rope_std = mx.zeros_like(q_ane_rope_4d)
for kk in range(half):
    q_ane_rope_std[:, :, :, kk] = q_ane_rope_4d[:, :, :, 2*kk]
    q_ane_rope_std[:, :, :, kk+half] = q_ane_rope_4d[:, :, :, 2*kk+1]

# GPU RoPE
q_gpu_rope = mx.fast.rope(q_gpu_before_rope, HEAD_DIM, traditional=False, base=1000000.0, scale=1.0, offset=CTX_LEN)

cos_q_after = cosine_sim(q_ane_rope_std, q_gpu_rope)
print(f"ANE Q after rope (de-interleaved) vs GPU: cos={cos_q_after:.6f}")

# Test: apply interleaved RoPE manually on the interleaved ANE data
# and compare with the ANE rope_q output
# Interleaved RoPE: pairs (2k, 2k+1) → x_even = x[..., 2k], x_odd = x[..., 2k+1]
# rotated_even = x_even * cos - x_odd * sin
# rotated_odd = x_even * sin + x_odd * cos
# But wait — the cos/sin are generated as [cos(θ_k), 1.0, sin(θ_k), 0.0] per pair
# So: x[..., 2k] * cos_k + (-x[..., 2k+1]) * sin_k = x[..., 2k]*cos_k - x[..., 2k+1]*sin_k
#     x[..., 2k+1] * 1.0 + x[..., 2k] * 0.0 = x[..., 2k+1]  ... WAIT

# Let me re-read the apply_rope implementation:
# reshape to [1, NH, pairs, 2] where pairs = seq * hd / 2
# x_e = slice(pairs, 0) → even elements [2k]
# x_o = slice(pairs, 1) → odd elements [2k+1]
# neg_xo = -x_o
# rotated = concat([neg_xo, x_e], axis=3) → [-x_odd, x_even]
# reshape to [1, NH, seq, hd]
# result = x * cos + rotated * sin
# So: result[2k] = x[2k] * cos[2k] + (-x[2k+1]) * sin[2k]
#     result[2k+1] = x[2k+1] * cos[2k+1] + x[2k] * sin[2k+1]
# With cos[2k] = cos(θ_k), cos[2k+1] = 1.0, sin[2k] = sin(θ_k), sin[2k+1] = 0.0:
# result[2k] = x[2k] * cos(θ_k) - x[2k+1] * sin(θ_k)
# result[2k+1] = x[2k+1]

# So the interleaved RoPE only modifies the EVEN indices (2k) using the odd (2k+1) as the paired value!
# The ODD indices remain unchanged (cos=1, sin=0)!
# This means: interleaved pair (2k, 2k+1) → rotate(x[2k], x[2k+1], θ_k)
#   where x[2k]_rotated = x[2k]*cos - x[2k+1]*sin
#   and x[2k+1] stays the same

# Meanwhile, half-rotation pairs (k, k+64) → rotate(x[k], x[k+64], θ_k)
#   x[k]_rotated = x[k]*cos - x[k+64]*sin
#   x[k+64]_rotated = x[k]*sin + x[k+64]*cos

# With interleaving: interleaved[2k] = original[k], interleaved[2k+1] = original[k+64]
# So: interleaved_rope[2k] = interleaved[2k]*cos - interleaved[2k+1]*sin = original[k]*cos - original[k+64]*sin ✓
# But: interleaved_rope[2k+1] = interleaved[2k+1] (UNCHANGED!) = original[k+64] ✗
# Should be: original[k+64]_rotated = original[k]*sin + original[k+64]*cos

# **THE BUG**: interleaved RoPE only rotates the FIRST element of each pair!
# The second element (2k+1) is left unchanged because cos=1, sin=0.
# But half-rotation rotates BOTH elements of each pair!

print("\n*** BUG IDENTIFIED ***")
print("Interleaved RoPE only rotates EVEN indices (2k). ODD indices (2k+1) are unchanged.")
print("Half-rotation rotates BOTH elements of each pair (k, k+64).")
print("The interleaving trick makes Q@K^T invariant ONLY if both elements are rotated the same way.")
print("With only one element rotated, Q@K^T is NOT invariant.")

# Let me verify: manually apply half-rotation RoPE on the interleaved data
# For each pair (2k, 2k+1) = (original[k], original[k+64]):
# We need: rotated[2k] = x[2k]*cos - x[2k+1]*sin  (this is what ANE does ✓)
#          rotated[2k+1] = x[2k]*sin + x[2k+1]*cos  (this is what ANE MISSES ✗)

# To fix: the cos/sin data for odd indices should be:
# cos[2k+1] = cos(θ_k)  (NOT 1.0)
# sin[2k+1] = sin(θ_k)  (NOT 0.0)
# And the apply_rope should use a DIFFERENT rotation for odd indices

# But the ANE apply_rope implementation can't do this because it uses the same
# cos/sin for both even and odd. The concat pattern only creates ONE rotation.

# Actually wait — let me re-read apply_rope more carefully:
# rotated = concat([neg_xo, x_e], axis=3) → for each pair, [-x_odd, x_even]
# reshape to [1, NH, seq, hd]
# result = x * cos + rotated * sin
# For element 2k: x[2k]*cos[2k] + (-x[2k+1])*sin[2k]
# For element 2k+1: x[2k+1]*cos[2k+1] + x[2k]*sin[2k+1]

# So if we set cos[2k+1] = cos(θ_k) and sin[2k+1] = sin(θ_k):
# result[2k+1] = x[2k+1]*cos(θ_k) + x[2k]*sin(θ_k)  ✓ THIS IS CORRECT!

# The fix is to change the cos/sin data from:
#   [cos(θ_0), 1.0, cos(θ_1), 1.0, ...] and [sin(θ_0), 0.0, sin(θ_1), 0.0, ...]
# to:
#   [cos(θ_0), cos(θ_0), cos(θ_1), cos(θ_1), ...] and [sin(θ_0), sin(θ_0), sin(θ_1), sin(θ_1), ...]

# But wait, that would give:
# result[2k+1] = x[2k+1]*cos(θ_k) + x[2k]*sin(θ_k)
# This is the SECOND half of the rotation: x2*cos + x1*sin
# For standard rotation: [x1*cos - x2*sin, x1*sin + x2*cos]
# With interleaving: x1=interleaved[2k]=original[k], x2=interleaved[2k+1]=original[k+64]
# result[2k] = original[k]*cos - original[k+64]*sin ✓
# result[2k+1] = original[k+64]*cos + original[k]*sin ✓ ← THIS IS CORRECT!

# BUT WAIT: the rotated tensor is [-x_odd, x_even] = [-x[2k+1], x[2k]]
# So rotated[2k] = -x[2k+1] and rotated[2k+1] = x[2k]
# result[2k+1] = x[2k+1]*cos[2k+1] + rotated[2k+1]*sin[2k+1]
#              = x[2k+1]*cos(θ_k) + x[2k]*sin(θ_k)  ✓

# So the fix IS to use cos/sin repeated for both even and odd indices!
# cos_data = [cos(θ_0), cos(θ_0), cos(θ_1), cos(θ_1), ...]
# sin_data = [sin(θ_0), sin(θ_0), sin(θ_1), sin(θ_1), ...]

print("\nFix: change cos/sin data from [cos(θ),1.0,...] to [cos(θ),cos(θ),...]")
print("     and from [sin(θ),0.0,...] to [sin(θ),sin(θ),...]")
