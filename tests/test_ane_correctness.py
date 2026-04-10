"""End-to-end correctness test: ANE draft model vs GPU draft model.

Compares the output of the full ANE forward pass with the MLX GPU draft model.
Target: cosine similarity > 0.99 at fp16 precision.
"""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel
from mirror_sd.dflash import DFlashDraftModel
from mirror_sd.loader import load_dflash_model

SEQ_Q = 16
CTX_LEN = 64

DRAFT_PATH = "z-lab/Qwen3-8B-DFlash-b16"


def cosine_sim(a: mx.array, b: mx.array) -> float:
    a_f = a.astype(mx.float32).flatten()
    b_f = b.astype(mx.float32).flatten()
    dot = float(mx.sum(a_f * b_f))
    na = float(mx.sqrt(mx.sum(a_f * a_f)))
    nb = float(mx.sqrt(mx.sum(b_f * b_f)))
    return dot / (na * nb + 1e-8)


def main():
    print("Loading draft model...")
    draft_model, config = load_dflash_model(DRAFT_PATH)

    # Create random inputs
    mx.random.seed(42)
    noise_embedding = mx.random.normal(shape=(1, SEQ_Q, 4096), dtype=mx.bfloat16)
    target_hidden = mx.random.normal(shape=(1, CTX_LEN, 5 * 4096), dtype=mx.bfloat16)

    # GPU reference
    print("Running GPU draft forward...")
    gpu_out = draft_model(noise_embedding, target_hidden)

    # ANE model
    print(f"Building ANE model (seq_q={SEQ_Q}, ctx_len={CTX_LEN})...")
    ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
    print("Loading ANE weights...")
    ane_model.load_weights(draft_model)

    print("Running ANE forward...")
    ane_out = ane_model.forward(noise_embedding, target_hidden, rope_offset=0)

    # Compare
    cos = cosine_sim(gpu_out, ane_out)
    max_err = float(mx.max(mx.abs(gpu_out.astype(mx.float32) - ane_out.astype(mx.float32))))
    mean_err = float(mx.mean(mx.abs(gpu_out.astype(mx.float32) - ane_out.astype(mx.float32))))

    print(f"\nResults:")
    print(f"  Cosine similarity: {cos:.6f}")
    print(f"  Max absolute error: {max_err:.6f}")
    print(f"  Mean absolute error: {mean_err:.6f}")
    print(f"  GPU output shape: {gpu_out.shape}, dtype: {gpu_out.dtype}")
    print(f"  ANE output shape: {ane_out.shape}, dtype: {ane_out.dtype}")

    if cos > 0.99:
        print("\n✓ PASS: ANE output matches GPU (cos > 0.99)")
    elif cos > 0.95:
        print("\n~ CLOSE: ANE output is close but not perfect (0.95 < cos < 0.99)")
    else:
        print(f"\n✗ FAIL: ANE output does not match GPU (cos = {cos:.6f})")

    # Per-token comparison
    for t in range(min(4, SEQ_Q)):
        cos_t = cosine_sim(gpu_out[:, t:t+1, :], ane_out[:, t:t+1, :])
        print(f"  Token {t}: cos={cos_t:.6f}")


if __name__ == "__main__":
    main()