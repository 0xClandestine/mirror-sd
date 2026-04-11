"""Full 5-layer pipeline test with scaled inputs to avoid fp16 overflow.
The ANE uses fp16 internally (max ~65504) while GPU uses bf16 (max ~3.4e38).
With standard random inputs, intermediate values can overflow fp16.
With scaled inputs (0.01x), we stay within fp16 range and test correctness."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, TARGET_HIDDEN
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

mx.random.seed(42)
noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16) * 0.01
target_hidden = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16) * 0.01

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
    print("~ CLOSE (fp16 precision)")
elif cos > 0.90:
    print("DECENT (fp16 accumulation error)")
else:
    print("✗ FAIL")

for t in range(min(4, SEQ_Q)):
    cos_t = cosine_sim(gpu_out[:, t:t+1, :], ane_out[:, t:t+1, :])
    print(f"  Token {t}: cos={cos_t:.6f}")

# Check for NaN/Inf
n_nan = int(mx.sum(mx.isnan(ane_out)))
n_inf = int(mx.sum(mx.isinf(ane_out)))
if n_nan > 0 or n_inf > 0:
    print(f"\n  NaN count: {n_nan}, Inf count: {n_inf}")
