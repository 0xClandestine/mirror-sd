"""End-to-end cosine test: ANE draft model full forward pass vs GPU reference.

Uses CORRECT RoPE (traditional=False, neox/half-rotation style) and proper
de-interleaving of Q/K outputs for comparison.
"""

import time
import mlx.core as mx
import mlx.nn as nn

from mirror_sd.dflash import DFlashDraftModel, DFlashConfig
from mirror_sd.loader import load_dflash_model
from mirror_sd.ane_model import ANEDraftModel

SEQ_Q = 16
CTX_LEN = 64


def cosine_sim(a, b):
    a = a.flatten().astype(mx.float32)
    b = b.flatten().astype(mx.float32)
    return float((a * b).sum() / (mx.sqrt((a * a).sum()) * mx.sqrt((b * b).sum()) + 1e-12))


def main():
    print("=== ANE End-to-End Forward Pass Cosine Test ===")
    print(f"SEQ_Q={SEQ_Q}, CTX_LEN={CTX_LEN}")
    print()

    # Load draft model
    print("Loading draft model...")
    draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")
    print("Draft model loaded.")

    # Create synthetic input data
    mx.random.seed(42)
    HIDDEN = 4096
    noise_emb = mx.random.normal((1, SEQ_Q, HIDDEN), dtype=mx.float32) * 0.02
    target_hid = mx.random.normal((1, CTX_LEN, 5 * HIDDEN), dtype=mx.float32) * 0.02
    mx.eval(noise_emb, target_hid)

    # GPU reference forward pass
    print("Running GPU reference...")
    gpu_output = draft_model(noise_emb, target_hid)
    mx.eval(gpu_output)
    print(f"GPU output: shape={gpu_output.shape}, rms={float(mx.sqrt(mx.mean(gpu_output**2))):.6f}")

    # ANE forward pass
    print("\nInitializing ANE model...")
    ane_model = ANEDraftModel(SEQ_Q, CTX_LEN)
    ane_model.load_weights(draft_model)
    print("ANE model initialized.")

    print("Running ANE forward pass...")
    ane_output = ane_model(noise_emb, target_hid, rope_offset=0, ctx_len=CTX_LEN)
    mx.eval(ane_output)
    print(f"ANE output: shape={ane_output.shape}, rms={float(mx.sqrt(mx.mean(ane_output**2))):.6f}")

    # Compare
    cos = cosine_sim(ane_output, gpu_output)
    print(f"\nCosine similarity (ANE vs GPU): {cos:.6f}")

    # Per-position comparison
    for pos in range(SEQ_Q):
        a = ane_output[0, pos, :]
        g = gpu_output[0, pos, :]
        c = cosine_sim(a, g)
        print(f"  Position {pos:2d}: cosine={c:.6f}")

    if cos > 0.99:
        print("\n✓ ANE forward pass matches GPU with high fidelity!")
    elif cos > 0.95:
        print("\n~ ANE forward pass is close but not perfect. May need investigation.")
    else:
        print("\n✗ ANE forward pass has significant divergence from GPU.")


if __name__ == "__main__":
    main()
