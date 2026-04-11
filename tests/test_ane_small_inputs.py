"""Compare ANE vs GPU for layer 0 only, and full 5-layer output."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, TARGET_HIDDEN, N_DFLASH_LAYERS
from mirror_sd.dflash import DFlashDraftModel
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

# Use smaller inputs to avoid fp16 overflow
mx.random.seed(42)
noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16) * 0.1
target_hidden = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16) * 0.1

# GPU
gpu_out = draft_model(noise_embedding, target_hidden)
print(f"GPU output range: [{float(gpu_out.min()):.2f}, {float(gpu_out.max()):.2f}]")

# ANE
print("Building ANE model...")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)
ane_out = ane_model.forward(noise_embedding, target_hidden, rope_offset=0)
print(f"ANE output range: [{float(ane_out.min()):.2f}, {float(ane_out.max()):.2f}]")

cos = cosine_sim(ane_out, gpu_out)
print(f"\nFull output cosine: {cos:.6f}")
if cos > 0.99:
    print("✓ PASS")
elif cos > 0.95:
    print("~ CLOSE")
else:
    print("✗ FAIL")

# Per-token
for t in range(min(4, SEQ_Q)):
    cos_t = cosine_sim(gpu_out[:, t:t+1, :], ane_out[:, t:t+1, :])
    print(f"  Token {t}: cos={cos_t:.6f}")
