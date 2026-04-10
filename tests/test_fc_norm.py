"""Test: rmsnorm + conv1x1_proj chain on ANE vs MLX.

Tests whether the concat-slice pattern works correctly when the activation
is an intermediate result (output of rmsnorm), not a direct placeholder.
"""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import numpy as np
import mlx.core as mx
from mirror_sd.dflash import DFlashDraftModel
from mirror_sd.loader import load_dflash_model
from mirror_sd.ane_model import ANEDraftModel

SEQ_Q = 16
CTX_LEN = 64
DRAFT_PATH = "z-lab/Qwen3-8B-DFlash-b16"


def test_fc_norm():
    """Test fc_norm kernel: conv1x1_proj(target_hid, fc_w) → rmsnorm(fc_out, norm_w).
    
    This is the simplest kernel — just one conv1x1_proj + rmsnorm.
    """
    print("Loading draft model...")
    draft_model, config = load_dflash_model(DRAFT_PATH)
    
    print("Building ANE model...")
    ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
    ane_model.load_weights(draft_model)
    
    mx.random.seed(42)
    target_hidden = mx.random.normal(shape=(1, CTX_LEN, 5 * 4096), dtype=mx.bfloat16)
    
    # GPU reference: fc + hidden_norm
    fc_w = draft_model.fc.weight.astype(mx.float32)  # [HIDDEN, TARGET_HIDDEN]
    hidden_norm_w = draft_model.hidden_norm.weight.astype(mx.float32)
    
    # target_hidden is [1, CTX_LEN, TARGET_HIDDEN]
    target_f32 = target_hidden.astype(mx.float32)
    fc_out = target_f32 @ fc_w.T  # [1, CTX_LEN, HIDDEN]
    
    # rmsnorm
    ms = fc_out.mean(axis=-1, keepdims=True)
    diff = fc_out - ms
    sq = diff * diff
    mean_sq = sq.mean(axis=-1, keepdims=True)
    inv_std = mx.rsqrt(mean_sq + 1e-6)
    normed = fc_out * inv_std
    context_expected = normed * hidden_norm_w
    
    # ANE
    ane_model._write_mlx_2d(ane_model.b_target, target_hidden)
    ane_model.kernels['fc_norm'].run(
        [ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm],
        [ane_model.b_context],
    )
    context_ane = ane_model._read_mlx_2d(ane_model.b_context, CTX_LEN, 4096)
    
    # Compare
    context_expected_f = context_expected.flatten()
    context_ane_f = context_ane.astype(mx.float32).flatten()
    dot = float(mx.sum(context_expected_f * context_ane_f))
    na = float(mx.sqrt(mx.sum(context_expected_f * context_expected_f)))
    nb = float(mx.sqrt(mx.sum(context_ane_f * context_ane_f)))
    cos = dot / (na * nb + 1e-8)
    
    print(f"fc_norm cosine similarity: {cos:.6f}")
    print(f"  Expected sample: {context_expected[0, 0, :4]}")
    print(f"  ANE sample:      {context_ane[0, 0, :4]}")
    
    return cos


if __name__ == "__main__":
    cos = test_fc_norm()
    if cos > 0.99:
        print("\n✓ PASS")
    else:
        print(f"\n✗ FAIL (cos={cos:.6f})")