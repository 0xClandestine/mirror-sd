"""Test individual ANE kernels against MLX ground truth."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import math
import mlx.core as mx
from mirror_sd.dflash import DFlashDraftModel
from mirror_sd.loader import load_dflash_model
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, INTERMEDIATE

SEQ_Q = 16
CTX_LEN = 64
DRAFT_PATH = "z-lab/Qwen3-8B-DFlash-b16"


def cosine_sim(a, b):
    a_f = a.astype(mx.float32).flatten()
    b_f = b.astype(mx.float32).flatten()
    dot = float(mx.sum(a_f * b_f))
    na = float(mx.sqrt(mx.sum(a_f * a_f)))
    nb = float(mx.sqrt(mx.sum(b_f * b_f)))
    return dot / (na * nb + 1e-8)


def mlx_rmsnorm(x, weight, eps=1e-6):
    ms = x.mean(axis=-1, keepdims=True)
    diff = x - ms
    sq = diff * diff
    mean_sq = sq.mean(axis=-1, keepdims=True)
    inv_std = mx.rsqrt(mean_sq + eps)
    return x * inv_std * weight


def mlx_per_head_rmsnorm(x, weight, n_heads, eps=1e-6):
    """RMSNorm with per-head weight: x is [1, seq, n_heads*head_dim], weight is [head_dim]."""
    seq = x.shape[1]
    head_dim = weight.shape[0]
    x_4d = x.reshape(1, seq, n_heads, head_dim)
    ms = x_4d.mean(axis=-1, keepdims=True)
    diff = x_4d - ms
    sq = diff * diff
    mean_sq = sq.mean(axis=-1, keepdims=True)
    inv_std = mx.rsqrt(mean_sq + eps)
    normed = x_4d * inv_std * weight
    return normed.reshape(1, seq, n_heads * head_dim)


def test_q_kernel(draft_model, ane_model, noise_embedding):
    """Test q_kernel: rmsnorm(hidden) → conv1x1(Q) → rmsnorm(q_norm)"""
    layer = draft_model.layers[0]
    in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
    q_proj_w = layer.self_attn.q_proj.weight.astype(mx.float32)
    q_norm_w = layer.self_attn.q_norm.weight.astype(mx.float32)
    
    hidden_f = noise_embedding.astype(mx.float32)
    normed = mlx_rmsnorm(hidden_f, in_norm_w)
    q_out = normed @ q_proj_w.T  # [1, SEQ_Q, N_HEADS*HEAD_DIM]
    q_normed = mlx_per_head_rmsnorm(q_out, q_norm_w, N_HEADS)
    
    # ANE
    ane_model._write_mlx_2d(ane_model.b_noise, noise_embedding)
    noise_data = ane_model.b_noise.read_f32()
    ane_model.b_hidden.write_f32(noise_data)
    ane_model.kernels['q_kernel'].run(
        [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_q_proj, ane_model.w_l0_q_norm],
        [ane_model.b_q_out],
    )
    q_ane = ane_model._read_mlx_2d(ane_model.b_q_out, SEQ_Q, N_HEADS * HEAD_DIM)
    
    cos = cosine_sim(q_normed, q_ane)
    print(f"q_kernel cosine: {cos:.6f}")
    return cos


def test_k_proj_ctx(draft_model, ane_model, context_mx):
    """Test k_proj_ctx: conv1x1(context, K_weight)"""
    layer = draft_model.layers[0]
    k_proj_w = layer.self_attn.k_proj.weight.astype(mx.float32)
    
    ctx_f = context_mx.astype(mx.float32)
    k_expected = ctx_f @ k_proj_w.T  # [1, CTX_LEN, N_KV_HEADS*HEAD_DIM]
    
    # ANE
    ane_model._write_mlx_2d(ane_model.b_context, context_mx)
    ane_model.kernels['k_proj_ctx'].run(
        [ane_model.b_context, ane_model.w_l0_k_proj],
        [ane_model.b_k_ctx],
    )
    k_ane = ane_model._read_mlx_2d(ane_model.b_k_ctx, CTX_LEN, N_KV_HEADS * HEAD_DIM)
    
    cos = cosine_sim(k_expected, k_ane)
    print(f"k_proj_ctx cosine: {cos:.6f}")
    return cos


def test_k_proj_noise(draft_model, ane_model, hidden_mx):
    """Test k_proj_noise: rmsnorm(hidden) → conv1x1(K_weight)"""
    layer = draft_model.layers[0]
    in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
    k_proj_w = layer.self_attn.k_proj.weight.astype(mx.float32)
    
    hidden_f = hidden_mx.astype(mx.float32)
    normed = mlx_rmsnorm(hidden_f, in_norm_w)
    k_expected = normed @ k_proj_w.T
    
    # ANE
    ane_model._write_mlx_2d(ane_model.b_hidden, hidden_mx)
    ane_model.kernels['k_proj_noise'].run(
        [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_k_proj],
        [ane_model.b_k_noise],
    )
    k_ane = ane_model._read_mlx_2d(ane_model.b_k_noise, SEQ_Q, N_KV_HEADS * HEAD_DIM)
    
    cos = cosine_sim(k_expected, k_ane)
    print(f"k_proj_noise cosine: {cos:.6f}")
    return cos


def test_v_proj_ctx(draft_model, ane_model, context_mx):
    """Test v_proj_ctx: conv1x1(context, V_weight)"""
    layer = draft_model.layers[0]
    v_proj_w = layer.self_attn.v_proj.weight.astype(mx.float32)
    
    ctx_f = context_mx.astype(mx.float32)
    v_expected = ctx_f @ v_proj_w.T
    
    ane_model._write_mlx_2d(ane_model.b_context, context_mx)
    ane_model.kernels['v_proj_ctx'].run(
        [ane_model.b_context, ane_model.w_l0_v_proj],
        [ane_model.b_v_ctx],
    )
    v_ane = ane_model._read_mlx_2d(ane_model.b_v_ctx, CTX_LEN, N_KV_HEADS * HEAD_DIM)
    
    cos = cosine_sim(v_expected, v_ane)
    print(f"v_proj_ctx cosine: {cos:.6f}")
    return cos


def test_v_proj_noise(draft_model, ane_model, hidden_mx):
    """Test v_proj_noise: rmsnorm(hidden) → conv1x1(V_weight)"""
    layer = draft_model.layers[0]
    in_norm_w = layer.input_layernorm.weight.astype(mx.float32)
    v_proj_w = layer.self_attn.v_proj.weight.astype(mx.float32)
    
    hidden_f = hidden_mx.astype(mx.float32)
    normed = mlx_rmsnorm(hidden_f, in_norm_w)
    v_expected = normed @ v_proj_w.T
    
    ane_model._write_mlx_2d(ane_model.b_hidden, hidden_mx)
    ane_model.kernels['v_proj_noise'].run(
        [ane_model.b_hidden, ane_model.w_l0_in_norm, ane_model.w_l0_v_proj],
        [ane_model.b_v_noise],
    )
    v_ane = ane_model._read_mlx_2d(ane_model.b_v_noise, SEQ_Q, N_KV_HEADS * HEAD_DIM)
    
    cos = cosine_sim(v_expected, v_ane)
    print(f"v_proj_noise cosine: {cos:.6f}")
    return cos


def main():
    print("Loading models...")
    draft_model, config = load_dflash_model(DRAFT_PATH)
    ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
    ane_model.load_weights(draft_model)
    
    mx.random.seed(42)
    noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
    target_hidden = mx.random.normal(shape=(1, CTX_LEN, 5 * HIDDEN), dtype=mx.bfloat16)
    
    # First run fc_norm to get context
    ane_model._write_mlx_2d(ane_model.b_target, target_hidden)
    ane_model.kernels['fc_norm'].run(
        [ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm],
        [ane_model.b_context],
    )
    context_mx = ane_model._read_mlx_2d(ane_model.b_context, CTX_LEN, HIDDEN)
    
    print("\n--- Testing individual kernels ---")
    
    cos_q = test_q_kernel(draft_model, ane_model, noise_embedding)
    cos_k_ctx = test_k_proj_ctx(draft_model, ane_model, context_mx)
    cos_k_noise = test_k_proj_noise(draft_model, ane_model, noise_embedding)
    cos_v_ctx = test_v_proj_ctx(draft_model, ane_model, context_mx)
    cos_v_noise = test_v_proj_noise(draft_model, ane_model, noise_embedding)
    
    print(f"\n--- Summary ---")
    print(f"q_kernel:     {cos_q:.6f}")
    print(f"k_proj_ctx:   {cos_k_ctx:.6f}")
    print(f"k_proj_noise: {cos_k_noise:.6f}")
    print(f"v_proj_ctx:   {cos_v_ctx:.6f}")
    print(f"v_proj_noise: {cos_v_noise:.6f}")
    
    all_pass = all(c > 0.99 for c in [cos_q, cos_k_ctx, cos_k_noise, cos_v_ctx, cos_v_noise])
    if all_pass:
        print("\n✓ ALL PASS")
    else:
        print("\n✗ SOME FAIL")


if __name__ == "__main__":
    main()