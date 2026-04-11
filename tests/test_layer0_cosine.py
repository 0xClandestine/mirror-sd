"""Layer 0 correctness test: ANE vs GPU with scaled inputs to avoid fp16 overflow."""

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

print("Loading draft model...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")

# Use small inputs to stay in fp16 range
mx.random.seed(42)
noise_embedding = (mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16) * 0.01)
target_hidden = (mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16) * 0.01)

# GPU: run just layer 0
layer = draft_model.layers[0]
hidden_f = noise_embedding.astype(mx.float32)
context = rmsnorm(draft_model.fc(target_hidden.astype(mx.float32)), draft_model.hidden_norm.weight)
gpu_l0 = layer(hidden_f, context)
print(f"GPU layer 0 output range: [{float(gpu_l0.min()):.2f}, {float(gpu_l0.max()):.2f}]")

# ANE: run up to layer 0
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
ane_model._run_layer(k, 0)
ane_l0 = ane_model._read_mlx_2d(ane_model.b_hidden, SEQ_Q, HIDDEN).astype(mx.float32)

print(f"ANE layer 0 output range: [{float(ane_l0.min()):.2f}, {float(ane_l0.max()):.2f}]")
cos = cosine_sim(ane_l0, gpu_l0)
print(f"\nLayer 0 cosine: {cos:.6f}")

if cos > 0.99:
    print("✓ PASS")
elif cos > 0.95:
    print("~ CLOSE (fp16 precision)")
else:
    print("✗ FAIL")
