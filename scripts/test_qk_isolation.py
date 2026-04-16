"""Isolation test: pinpoint where Q/K cosine collapses inside mega_qkv.

Strategy:
1. Load draft model, create real input data
2. Run GPU reference (MLX) forward pass for layer 0 → gold Q/K/V at each step
3. Run mega_proj on ANE → raw Q/K/V → cosine vs GPU
4. Run mega_proj_qknorm on ANE → Q_normed/K_normed → cosine vs GPU
5. Run mega_qkv on ANE → Q_rope/K_rope → cosine vs GPU

This isolates whether the per-head norm or RoPE is the breaking point.
"""

import sys
import time

import mlx.core as mx
import mlx.nn as nn

from mirror_sd.dflash import DFlashConfig, DFlashDraftModel
from mirror_sd.loader import load_dflash_model
from mirror_sd.ane_model import (
    ANEDraftModel, HIDDEN, HEAD_DIM, N_HEADS, N_KV_HEADS, MIN_SPATIAL_WIDTH,
    align_width, _interleave_head_dims_mx,
)

SEQ_Q = 16
CTX_LEN = 64


def cosine_sim(a, b):
    a = a.flatten().astype(mx.float32)
    b = b.flatten().astype(mx.float32)
    return float((a * b).sum() / (mx.sqrt((a * a).sum()) * mx.sqrt((b * b).sum()) + 1e-12))


def deinterleave_heads(w_flat, n_heads, head_dim):
    half = head_dim // 2
    w = mx.array(w_flat, dtype=mx.float32).reshape(n_heads, head_dim, -1)
    w_first = w[:, :half, :]
    w_second = w[:, half:, :]
    stacked = mx.stack([w_first, w_second], axis=2)
    return stacked.reshape(n_heads, head_dim, -1).reshape(-1)


def interleave_heads(w_2d, n_heads, head_dim):
    half = head_dim // 2
    w_4d = w_2d.reshape(n_heads, head_dim, -1)
    w_first = w_4d[:, :half, :]
    w_second = w_4d[:, half:, :]
    stacked = mx.stack([w_first, w_second], axis=2)
    return stacked.reshape(n_heads, head_dim, -1).reshape(-1)


def deinterleave_output(q_4d, n_heads, head_dim):
    """De-interleave ANE Q/K output back to standard format.
    Input: [B, n_heads, seq, head_dim] in interleaved format
    Output: [B, n_heads, seq, head_dim] in standard format
    
    Interleaved: [d0, d_half, d1, d_half+1, ...] 
    Standard: [d0, d1, ..., d_half-1, d_half, ..., d_dim-1]
    """
    half = head_dim // 2
    B, nh, seq, hd = q_4d.shape
    q_flat = q_4d.reshape(B, nh, seq, half, 2)
    q_deil = mx.concatenate([q_flat[..., 0], q_flat[..., 1]], axis=-1)
    return q_deil


def python_rmsnorm(x, weight, eps=1e-6):
    rms = mx.sqrt(mx.mean(x.astype(mx.float32) ** 2, axis=-1, keepdims=True) + eps)
    return (x / rms) * weight


def python_perhead_norm(q_4d, norm_weight, n_heads):
    """Apply per-head RMSNorm.
    q_4d: [B, n_heads, seq, head_dim]
    norm_weight: [head_dim]
    Returns: [B, n_heads, seq, head_dim]
    """
    B, nh, seq, hd = q_4d.shape
    q_flat = q_4d.transpose(0, 2, 1, 3).reshape(B * seq, nh, hd)
    normed = python_rmsnorm(q_flat, norm_weight)
    return normed.reshape(B, seq, nh, hd).transpose(0, 2, 1, 3)


def python_rope(x_4d, offset, rope_theta=1000000.0):
    """Apply RoPE (neox/half-rotation style, traditional=False) to [B, n_heads, seq, head_dim]."""
    B, nh, seq, hd = x_4d.shape
    half = hd // 2
    positions = mx.array([offset + p for p in range(seq)], dtype=mx.float32)
    freqs = mx.array([1.0 / (rope_theta ** (2.0 * d / hd)) for d in range(half)], dtype=mx.float32)
    angles = positions[:, None] * freqs[None, :]
    cos_vals = mx.cos(angles)[None, None]  # [1, 1, seq, half]
    sin_vals = mx.sin(angles)[None, None]

    x_first = x_4d[..., :half]
    x_second = x_4d[..., half:]
    out_first = x_first * cos_vals - x_second * sin_vals
    out_second = x_first * sin_vals + x_second * cos_vals
    return mx.concatenate([out_first, out_second], axis=-1)


def read_ane_tensor_4d(buf, shape):
    data = buf.read_f32()
    arr = mx.array(data, dtype=mx.float32).reshape(*shape)
    return arr


def main():
    print("=== ANE Q/K Isolation Test ===")
    print(f"SEQ_Q={SEQ_Q}, CTX_LEN={CTX_LEN}")
    print()

    # Load draft model
    print("Loading draft model...")
    draft_model, _ = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")
    print("Draft model loaded.")

    layer = draft_model.layers[0]
    attn = layer.self_attn

    # Create synthetic input data
    mx.random.seed(42)
    noise_emb = mx.random.normal((1, SEQ_Q, HIDDEN), dtype=mx.float32) * 0.02
    target_hid = mx.random.normal((1, CTX_LEN, 5 * HIDDEN), dtype=mx.float32) * 0.02
    mx.eval(noise_emb, target_hid)

    # Compute context (same as ANE's _compute_context)
    fc_w = draft_model.fc.weight.astype(mx.float32)
    hidden_norm_w = draft_model.hidden_norm.weight.astype(mx.float32)
    fc_out = target_hid @ fc_w.T
    rms = mx.sqrt(mx.mean(fc_out.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
    context = (fc_out / rms) * hidden_norm_w
    mx.eval(context)

    # --- GPU reference: layer 0 step-by-step ---
    print("\n--- GPU Reference (Layer 0) ---")
    normed_hidden = python_rmsnorm(noise_emb, layer.input_layernorm.weight, eps=1e-6)
    mx.eval(normed_hidden)

    kv_input = mx.concatenate([context, normed_hidden], axis=1)
    mx.eval(kv_input)
    ctx_len_actual = context.shape[1]

    # Q projection + norm + RoPE
    gpu_q = attn.q_proj(normed_hidden)
    gpu_q_4d = gpu_q.reshape(1, SEQ_Q, N_HEADS, HEAD_DIM)
    gpu_q_normed = python_perhead_norm(gpu_q_4d.transpose(0, 2, 1, 3), attn.q_norm.weight, N_HEADS)
    gpu_q_rope = python_rope(gpu_q_normed, offset=ctx_len_actual)

    # K projection + norm + RoPE
    gpu_k = attn.k_proj(kv_input)
    gpu_k_4d = gpu_k.reshape(1, ctx_len_actual + SEQ_Q, N_KV_HEADS, HEAD_DIM)
    gpu_k_normed = python_perhead_norm(gpu_k_4d.transpose(0, 2, 1, 3), attn.k_norm.weight, N_KV_HEADS)
    gpu_k_rope = python_rope(gpu_k_normed, offset=0)

    # V projection
    gpu_v = attn.v_proj(kv_input)
    gpu_v_4d = gpu_v.reshape(1, ctx_len_actual + SEQ_Q, N_KV_HEADS, HEAD_DIM).transpose(0, 2, 1, 3)

    mx.eval(gpu_q_4d, gpu_q_normed, gpu_q_rope, gpu_k_4d, gpu_k_normed, gpu_k_rope, gpu_v_4d)

    print(f"GPU Q raw:     shape={gpu_q_4d.shape}, rms={float(mx.sqrt(mx.mean(gpu_q_4d**2))):.6f}")
    print(f"GPU Q normed:  shape={gpu_q_normed.shape}, rms={float(mx.sqrt(mx.mean(gpu_q_normed**2))):.6f}")
    print(f"GPU Q rope:    shape={gpu_q_rope.shape}, rms={float(mx.sqrt(mx.mean(gpu_q_rope**2))):.6f}")
    print(f"GPU K raw:     shape={gpu_k_4d.shape}, rms={float(mx.sqrt(mx.mean(gpu_k_4d**2))):.6f}")
    print(f"GPU K normed:  shape={gpu_k_normed.shape}, rms={float(mx.sqrt(mx.mean(gpu_k_normed**2))):.6f}")
    print(f"GPU K rope:    shape={gpu_k_rope.shape}, rms={float(mx.sqrt(mx.mean(gpu_k_rope**2))):.6f}")
    print(f"GPU V:         shape={gpu_v_4d.shape}, rms={float(mx.sqrt(mx.mean(gpu_v_4d**2))):.6f}")

    # --- Initialize ANE ---
    print("\n--- ANE Kernels ---")
    import mirror_sd_ane as ane_lib

    w_sq = align_width(SEQ_Q)
    w_ctx = align_width(CTX_LEN)
    w_kv = w_ctx + w_sq

    kernels_list = ane_lib.compile_dflash_kernels(SEQ_Q, CTX_LEN, 60000.0)
    kernels = {k.name: k for k in kernels_list}
    print(f"Compiled {len(kernels)} kernels: {list(kernels.keys())}")

    # Prepare ANE weights (same as ANEDraftModel._load_layer_weights)
    in_norm_w = layer.input_layernorm.weight.astype(mx.float32).flatten().tolist()
    w_in_norm = ane_lib.ANETensor.from_buffer(
        1, HIDDEN, 1, w_sq,
        memoryview(mx.broadcast_to(
            mx.array(in_norm_w, dtype=mx.float32).reshape(HIDDEN, 1),
            (HIDDEN, w_sq)
        ))
    )

    q_proj_w_il = _interleave_head_dims_mx(attn.q_proj.weight.astype(mx.float32), N_HEADS, HEAD_DIM)
    w_q_proj = ane_lib.ANETensor.from_buffer(
        1, HIDDEN, 1, N_HEADS * HEAD_DIM,
        memoryview((lambda: (
            d := mx.array(q_proj_w_il, dtype=mx.float32).reshape(N_HEADS * HEAD_DIM, HIDDEN),
            p := mx.zeros((HIDDEN, align_width(N_HEADS * HEAD_DIM)), dtype=mx.float32),
            p.__setitem__((slice(None), slice(N_HEADS * HEAD_DIM)), d.T),
            p
        )[-1])())
    )

    k_proj_w_il = _interleave_head_dims_mx(attn.k_proj.weight.astype(mx.float32), N_KV_HEADS, HEAD_DIM)
    w_k_proj = ane_lib.ANETensor.from_buffer(
        1, HIDDEN, 1, N_KV_HEADS * HEAD_DIM,
        memoryview((lambda: (
            d := mx.array(k_proj_w_il, dtype=mx.float32).reshape(N_KV_HEADS * HEAD_DIM, HIDDEN),
            p := mx.zeros((HIDDEN, align_width(N_KV_HEADS * HEAD_DIM)), dtype=mx.float32),
            p.__setitem__((slice(None), slice(N_KV_HEADS * HEAD_DIM)), d.T),
            p
        )[-1])())
    )

    v_proj_w = attn.v_proj.weight.astype(mx.float32)
    w_v_proj = ane_lib.ANETensor.from_buffer(
        1, HIDDEN, 1, N_KV_HEADS * HEAD_DIM,
        memoryview((lambda: (
            d := v_proj_w.reshape(N_KV_HEADS * HEAD_DIM, HIDDEN),
            p := mx.zeros((HIDDEN, align_width(N_KV_HEADS * HEAD_DIM)), dtype=mx.float32),
            p.__setitem__((slice(None), slice(N_KV_HEADS * HEAD_DIM)), d.T),
            p
        )[-1])())
    )

    # Per-head Q/K norm weights (4D interleaved, broadcasted)
    def make_4d_norm_weight(head_weight_list, n_heads, width):
        w = align_width(width)
        head_dim = len(head_weight_list)
        half = head_dim // 2
        il = []
        for k_idx in range(half):
            il.append(head_weight_list[k_idx])
            il.append(head_weight_list[k_idx + half])
        il_arr = mx.array(il, dtype=mx.float32).reshape(head_dim, 1)
        arr = mx.broadcast_to(il_arr, (head_dim, n_heads * w))
        return ane_lib.ANETensor.from_buffer(1, head_dim, 1, n_heads * w, memoryview(arr))

    q_norm_w_list = attn.q_norm.weight.astype(mx.float32).flatten().tolist()
    k_norm_w_list = attn.k_norm.weight.astype(mx.float32).flatten().tolist()
    w_q_norm_4d = make_4d_norm_weight(q_norm_w_list, N_HEADS, w_sq)
    w_k_norm_4d = make_4d_norm_weight(k_norm_w_list, N_KV_HEADS, w_kv)

    # Compute RoPE buffers
    rope_theta = 1000000.0
    half = HEAD_DIM // 2

    q_positions = mx.array([ctx_len_actual + p for p in range(w_sq)], dtype=mx.float32)
    q_freqs = mx.array([1.0 / (rope_theta ** (2.0 * d / HEAD_DIM)) for d in range(half)], dtype=mx.float32)
    q_angles = q_positions[:, None] * q_freqs[None, :]
    q_cos = mx.repeat(mx.cos(q_angles), 2, axis=1)
    q_sin = mx.repeat(mx.sin(q_angles), 2, axis=1)
    b_cos_q = ane_lib.ANETensor(1, 1, w_sq, HEAD_DIM)
    b_sin_q = ane_lib.ANETensor(1, 1, w_sq, HEAD_DIM)
    b_cos_q.write_buffer(memoryview(q_cos.flatten().astype(mx.float32)))
    b_sin_q.write_buffer(memoryview(q_sin.flatten().astype(mx.float32)))

    # K RoPE: context [0:ctx_len], noise [w_ctx:w_ctx+seq_q]
    k_ctx_pos = mx.array([p for p in range(ctx_len_actual)], dtype=mx.float32)
    k_noise_pos = mx.array([ctx_len_actual + p for p in range(SEQ_Q)], dtype=mx.float32)
    k_ctx_angles = k_ctx_pos[:, None] * q_freqs[None, :]
    k_noise_angles = k_noise_pos[:, None] * q_freqs[None, :]
    k_ctx_cos = mx.repeat(mx.cos(k_ctx_angles), 2, axis=1)
    k_ctx_sin = mx.repeat(mx.sin(k_ctx_angles), 2, axis=1)
    k_noise_cos = mx.repeat(mx.cos(k_noise_angles), 2, axis=1)
    k_noise_sin = mx.repeat(mx.sin(k_noise_angles), 2, axis=1)
    k_cos_full = mx.zeros((1, 1, w_kv, HEAD_DIM), dtype=mx.float32)
    k_sin_full = mx.zeros((1, 1, w_kv, HEAD_DIM), dtype=mx.float32)
    k_cos_full[0, 0, :ctx_len_actual, :] = k_ctx_cos
    k_cos_full[0, 0, w_ctx:w_ctx + SEQ_Q, :] = k_noise_cos
    k_sin_full[0, 0, :ctx_len_actual, :] = k_ctx_sin
    k_sin_full[0, 0, w_ctx:w_ctx + SEQ_Q, :] = k_noise_sin
    mx.eval(k_cos_full, k_sin_full)
    b_cos_k = ane_lib.ANETensor(1, 1, w_kv, HEAD_DIM)
    b_sin_k = ane_lib.ANETensor(1, 1, w_kv, HEAD_DIM)
    b_cos_k.write_buffer(memoryview(k_cos_full.flatten().astype(mx.float32)))
    b_sin_k.write_buffer(memoryview(k_sin_full.flatten().astype(mx.float32)))

    # Write input buffers
    b_hidden = ane_lib.ANETensor(1, HIDDEN, 1, w_sq)
    b_context = ane_lib.ANETensor(1, HIDDEN, 1, w_ctx)

    noise_f32 = noise_emb.astype(mx.float32).transpose(0, 2, 1)
    padded_h = mx.zeros((1, HIDDEN, w_sq), dtype=mx.float32)
    padded_h[:, :, :SEQ_Q] = noise_f32
    mx.eval(padded_h)
    b_hidden.write_buffer(memoryview(padded_h))

    ctx_f32 = context.astype(mx.float32).transpose(0, 2, 1)
    padded_c = mx.zeros((1, HIDDEN, w_ctx), dtype=mx.float32)
    padded_c[:, :, :ctx_len_actual] = ctx_f32
    mx.eval(padded_c)
    b_context.write_buffer(memoryview(padded_c))

    # --- Test 1: mega_proj (projections only) ---
    print("\n=== Test 1: mega_proj (projections only) ===")
    b_k_4d_proj = ane_lib.ANETensor(1, N_KV_HEADS, HEAD_DIM, w_kv)
    b_v_4d_t_proj = ane_lib.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)
    b_q_4d_t_proj = ane_lib.ANETensor(1, N_HEADS, w_sq, HEAD_DIM)

    kernels['mega_proj'].run_uncached(
        [b_hidden, w_in_norm, b_context, w_k_proj, w_v_proj, w_q_proj],
        [b_k_4d_proj, b_v_4d_t_proj, b_q_4d_t_proj],
    )

    # Read back raw Q from mega_proj
    # mega_proj K output: [1, N_KV_HEADS, HEAD_DIM, w_kv], transposed to [1, N_KV_HEADS, w_kv, HEAD_DIM]
    # mega_proj V output: [1, N_KV_HEADS, w_kv, HEAD_DIM] (already transposed)
    # mega_proj Q output: [1, N_HEADS, w_sq, HEAD_DIM] (transposed)

    ane_q_raw = read_ane_tensor_4d(b_q_4d_t_proj, (1, N_HEADS, w_sq, HEAD_DIM))[:, :, :SEQ_Q, :]
    ane_q_raw_deil = deinterleave_output(ane_q_raw, N_HEADS, HEAD_DIM)
    # GPU Q raw is [1, SEQ_Q, N_HEADS, HEAD_DIM], need to transpose to match
    gpu_q_4d_t = gpu_q_4d.transpose(0, 2, 1, 3)  # [1, N_HEADS, SEQ_Q, HEAD_DIM]
    cos_q_raw = cosine_sim(ane_q_raw_deil, gpu_q_4d_t)
    rms_q_raw = float(mx.sqrt(mx.mean(ane_q_raw_deil ** 2)))
    print(f"Q raw:  cosine={cos_q_raw:.6f}, rms_ane={rms_q_raw:.6f}, rms_gpu={float(mx.sqrt(mx.mean(gpu_q_4d_t**2))):.6f}")

    ane_k_raw = read_ane_tensor_4d(b_k_4d_proj, (1, N_KV_HEADS, HEAD_DIM, w_kv))
    # mega_proj K is transposed [0,1,3,2] so output is [1, N_KV_HEADS, w_kv, HEAD_DIM]
    # But let me check the actual buffer layout...
    # Actually mega_proj builds k_4d as [1, N_KV_HEADS, HEAD_DIM, w_kv] then transposes to [1, N_KV_HEADS, w_kv, HEAD_DIM]
    # Wait, looking at the code: let _k_4d_t = g.transpose(k_4d, [0, 1, 3, 2])
    # k_4d is [1, N_KV_HEADS, HEAD_DIM, w_kv], transpose [0,1,3,2] gives [1, N_KV_HEADS, w_kv, HEAD_DIM]
    # But the output buffer is declared as b_k_4d_proj with shape (1, N_KV_HEADS, HEAD_DIM, w_kv)
    # Hmm, this doesn't match. Let me check the ANE output binding...
    # The Rust graph defines outputs as the last N tensors. For mega_proj, the outputs are
    # _k_4d_t, _v_4d_t, _q_4d_t. But in the Rust code, those are bound to the output tensors
    # in the order they appear in the graph. The ANE graph's output order matters.
    # Actually, looking at build_mega_proj_kernel: the graph has _k_4d_t, _v_4d_t, _q_4d_t as outputs
    # The ANE runtime reads outputs in the order they were created as the "last" operations.

    # Let me just try both shapes and see which one has better cosine
    ane_k_try1 = read_ane_tensor_4d(b_k_4d_proj, (1, N_KV_HEADS, HEAD_DIM, w_kv))
    ane_k_try2 = ane_k_try1.transpose(0, 1, 3, 2)  # swap to [1, N_KV_HEADS, w_kv, HEAD_DIM]

    # Try de-interleaving both layouts
    ane_k_try1_deil = deinterleave_output(ane_k_try1, N_KV_HEADS, HEAD_DIM)  # [1, NKV, HEAD_DIM, w_kv] de-interleaved on dim 2
    ane_k_try2_deil = deinterleave_output(ane_k_try2, N_KV_HEADS, HEAD_DIM)  # [1, NKV, w_kv, HEAD_DIM] de-interleaved on dim 3

    gpu_k_4d_t = gpu_k_4d.transpose(0, 2, 1, 3)  # [1, N_KV_HEADS, ctx_len+SEQ_Q, HEAD_DIM]
    # GPU K has ctx_len+SEQ_Q positions, but ANE has w_kv padded positions
    gpu_k_ctx = gpu_k_4d_t[:, :, :ctx_len_actual, :]
    gpu_k_noise = gpu_k_4d_t[:, :, ctx_len_actual:, :]

    # try2_deil: [1, NKV, w_kv, HD] → slice ctx and noise
    cos_k_ctx_t2 = cosine_sim(ane_k_try2_deil[:, :, :ctx_len_actual, :], gpu_k_ctx)
    cos_k_noise_t2 = cosine_sim(ane_k_try2_deil[:, :, w_ctx:w_ctx+SEQ_Q, :], gpu_k_noise)

    # try1_deil: [1, NKV, HD, w_kv] → need to transpose for comparison
    ane_k_try1_deil_t = ane_k_try1_deil.transpose(0, 1, 3, 2)  # [1, NKV, w_kv, HD]
    cos_k_ctx_t1 = cosine_sim(ane_k_try1_deil_t[:, :, :ctx_len_actual, :], gpu_k_ctx)
    cos_k_noise_t1 = cosine_sim(ane_k_try1_deil_t[:, :, w_ctx:w_ctx+SEQ_Q, :], gpu_k_noise)

    best_k_layout = "try2" if cos_k_ctx_t2 > cos_k_ctx_t1 else "try1"
    cos_k_ctx_best = max(cos_k_ctx_t2, cos_k_ctx_t1)
    cos_k_noise_best = max(cos_k_noise_t2, cos_k_noise_t1)
    print(f"K raw ({best_k_layout}): ctx cosine={cos_k_ctx_best:.6f}, noise cosine={cos_k_noise_best:.6f}")

    ane_v_raw = read_ane_tensor_4d(b_v_4d_t_proj, (1, N_KV_HEADS, w_kv, HEAD_DIM))
    ane_v_ctx = ane_v_raw[:, :, :ctx_len_actual, :]
    ane_v_noise = ane_v_raw[:, :, w_ctx:w_ctx+SEQ_Q, :]
    gpu_v_4d_t = gpu_v_4d
    cos_v_ctx = cosine_sim(ane_v_ctx, gpu_v_4d_t[:, :, :ctx_len_actual, :])
    cos_v_noise = cosine_sim(ane_v_noise, gpu_v_4d_t[:, :, ctx_len_actual:, :])
    print(f"V raw: ctx cosine={cos_v_ctx:.6f}, noise cosine={cos_v_noise:.6f}")

    # --- Test 2: mega_proj_qknorm (projections + per-head Q/K norm, NO RoPE) ---
    print("\n=== Test 2: mega_proj_qknorm (proj + norm, NO RoPE) ===")
    b_k_norm_4d = ane_lib.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)
    b_v_4d_t_norm = ane_lib.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)
    b_q_norm_4d = ane_lib.ANETensor(1, N_HEADS, w_sq, HEAD_DIM)

    kernels['mega_proj_qknorm'].run_uncached(
        [b_hidden, w_in_norm, b_context,
         w_k_proj, w_k_norm_4d,
         w_v_proj,
         w_q_proj, w_q_norm_4d],
        [b_k_norm_4d, b_v_4d_t_norm, b_q_norm_4d],
    )

    ane_q_normed = read_ane_tensor_4d(b_q_norm_4d, (1, N_HEADS, w_sq, HEAD_DIM))[:, :, :SEQ_Q, :]
    ane_q_normed_deil = deinterleave_output(ane_q_normed, N_HEADS, HEAD_DIM)
    cos_q_normed = cosine_sim(ane_q_normed_deil, gpu_q_normed)
    rms_q_normed = float(mx.sqrt(mx.mean(ane_q_normed_deil ** 2)))
    print(f"Q normed: cosine={cos_q_normed:.6f}, rms_ane={rms_q_normed:.6f}, rms_gpu={float(mx.sqrt(mx.mean(gpu_q_normed**2))):.6f}")

    # For K normed, try both layouts
    ane_k_normed = read_ane_tensor_4d(b_k_norm_4d, (1, N_KV_HEADS, w_kv, HEAD_DIM))
    ane_k_normed_deil = deinterleave_output(ane_k_normed, N_KV_HEADS, HEAD_DIM)
    ane_k_normed_deil_ctx = ane_k_normed_deil[:, :, :ctx_len_actual, :]
    ane_k_normed_deil_noise = ane_k_normed_deil[:, :, w_ctx:w_ctx+SEQ_Q, :]
    cos_k_normed_ctx = cosine_sim(ane_k_normed_deil_ctx, gpu_k_normed[:, :, :ctx_len_actual, :])
    cos_k_normed_noise = cosine_sim(ane_k_normed_deil_noise, gpu_k_normed[:, :, ctx_len_actual:, :])
    print(f"K normed: ctx cosine={cos_k_normed_ctx:.6f}, noise cosine={cos_k_normed_noise:.6f}")

    # Also try the other layout for K normed (in case ANE layout differs from expected)
    ane_k_normed_alt = read_ane_tensor_4d(b_k_norm_4d, (1, N_KV_HEADS, HEAD_DIM, w_kv)).transpose(0, 1, 3, 2)
    ane_k_normed_alt_deil = deinterleave_output(ane_k_normed_alt, N_KV_HEADS, HEAD_DIM)
    ane_k_normed_alt_deil_ctx = ane_k_normed_alt_deil[:, :, :ctx_len_actual, :]
    ane_k_normed_alt_deil_noise = ane_k_normed_alt_deil[:, :, w_ctx:w_ctx+SEQ_Q, :]
    cos_k_normed_alt_ctx = cosine_sim(ane_k_normed_alt_deil_ctx, gpu_k_normed[:, :, :ctx_len_actual, :])
    cos_k_normed_alt_noise = cosine_sim(ane_k_normed_alt_deil_noise, gpu_k_normed[:, :, ctx_len_actual:, :])
    if cos_k_normed_alt_ctx > cos_k_normed_ctx:
        print(f"K normed (alt layout): ctx cosine={cos_k_normed_alt_ctx:.6f}, noise cosine={cos_k_normed_alt_noise:.6f}")
        cos_k_normed_ctx = cos_k_normed_alt_ctx
        cos_k_normed_noise = cos_k_normed_alt_noise

    ane_v_normed = read_ane_tensor_4d(b_v_4d_t_norm, (1, N_KV_HEADS, w_kv, HEAD_DIM))
    cos_v_normed_ctx = cosine_sim(ane_v_normed[:, :, :ctx_len_actual, :], gpu_v_4d[:, :, :ctx_len_actual, :])
    cos_v_normed_noise = cosine_sim(ane_v_normed[:, :, w_ctx:w_ctx+SEQ_Q, :], gpu_v_4d[:, :, ctx_len_actual:, :])
    print(f"V normed: ctx cosine={cos_v_normed_ctx:.6f}, noise cosine={cos_v_normed_noise:.6f}")

    # --- Test 3: mega_qkv (projections + per-head Q/K norm + RoPE) ---
    print("\n=== Test 3: mega_qkv (proj + norm + RoPE) ===")
    b_k_rope_4d = ane_lib.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)
    b_v_4d_t_full = ane_lib.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)
    b_q_rope_4d = ane_lib.ANETensor(1, N_HEADS, w_sq, HEAD_DIM)

    kernels['mega_qkv'].run_uncached(
        [b_hidden, w_in_norm, b_context,
         w_k_proj, w_k_norm_4d, b_cos_k, b_sin_k,
         w_v_proj,
         w_q_proj, w_q_norm_4d, b_cos_q, b_sin_q],
        [b_k_rope_4d, b_v_4d_t_full, b_q_rope_4d],
    )

    ane_q_rope = read_ane_tensor_4d(b_q_rope_4d, (1, N_HEADS, w_sq, HEAD_DIM))[:, :, :SEQ_Q, :]
    ane_q_rope_deil = deinterleave_output(ane_q_rope, N_HEADS, HEAD_DIM)
    cos_q_rope = cosine_sim(ane_q_rope_deil, gpu_q_rope)
    print(f"Q rope: cosine={cos_q_rope:.6f}")

    ane_k_rope = read_ane_tensor_4d(b_k_rope_4d, (1, N_KV_HEADS, w_kv, HEAD_DIM))
    ane_k_rope_deil = deinterleave_output(ane_k_rope, N_KV_HEADS, HEAD_DIM)
    ane_k_rope_deil_ctx = ane_k_rope_deil[:, :, :ctx_len_actual, :]
    ane_k_rope_deil_noise = ane_k_rope_deil[:, :, w_ctx:w_ctx+SEQ_Q, :]
    cos_k_rope_ctx = cosine_sim(ane_k_rope_deil_ctx, gpu_k_rope[:, :, :ctx_len_actual, :])
    cos_k_rope_noise = cosine_sim(ane_k_rope_deil_noise, gpu_k_rope[:, :, ctx_len_actual:, :])
    print(f"K rope: ctx cosine={cos_k_rope_ctx:.6f}, noise cosine={cos_k_rope_noise:.6f}")

    ane_v_full = read_ane_tensor_4d(b_v_4d_t_full, (1, N_KV_HEADS, w_kv, HEAD_DIM))
    cos_v_full_ctx = cosine_sim(ane_v_full[:, :, :ctx_len_actual, :], gpu_v_4d[:, :, :ctx_len_actual, :])
    cos_v_full_noise = cosine_sim(ane_v_full[:, :, w_ctx:w_ctx+SEQ_Q, :], gpu_v_4d[:, :, ctx_len_actual:, :])
    print(f"V full: ctx cosine={cos_v_full_ctx:.6f}, noise cosine={cos_v_full_noise:.6f}")

    # --- Summary ---
    print("\n" + "=" * 70)
    print("SUMMARY: Cosine similarity vs GPU reference")
    print("=" * 70)
    print(f"{'Step':<30} {'Q cosine':>10} {'K ctx cosine':>12} {'K noise cosine':>14}")
    print("-" * 70)
    print(f"{'1. Raw proj (mega_proj)':<30} {cos_q_raw:>10.6f} {'N/A':>12} {'N/A':>14}")
    print(f"{'2. Proj+norm (qknorm)':<30} {cos_q_normed:>10.6f} {cos_k_normed_ctx:>12.6f} {cos_k_normed_noise:>14.6f}")
    print(f"{'3. Proj+norm+rope (mega_qkv)':<30} {cos_q_rope:>10.6f} {cos_k_rope_ctx:>12.6f} {cos_k_rope_noise:>14.6f}")
    print()
    print("If step 2 cosine is high and step 3 is low → RoPE is the problem")
    print("If step 2 cosine is already low → per-head norm is the problem")
    print("If step 1 cosine is low → projection is the problem")


if __name__ == "__main__":
    main()
