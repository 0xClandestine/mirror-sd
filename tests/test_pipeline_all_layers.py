"""Run full 5-layer pipeline and find where NaN/Inf first appears."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, TARGET_HIDDEN, N_DFLASH_LAYERS
from mirror_sd.loader import load_dflash_model

SEQ_Q = 16
CTX_LEN = 64

def check_tensor(name, tensor):
    data = tensor.read_f32()
    n_nan = sum(1 for x in data if x != x)
    n_inf = sum(1 for x in data if x == x and abs(x) == float('inf'))
    nonzero = [x for x in data if x != 0.0 and x == x and abs(x) != float('inf')]
    vmin = min(nonzero) if nonzero else 0
    vmax = max(nonzero) if nonzero else 0
    shape = tensor.shape
    status = "BAD" if n_nan > 0 or n_inf > 0 else "ok"
    print(f"  {name}: shape={shape} NaN={n_nan} Inf={n_inf} range=[{vmin:.2f},{vmax:.2f}] {status}")
    return n_nan > 0 or n_inf > 0

print("Loading draft model...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")

mx.random.seed(42)
noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
target_hidden = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16)

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
check_tensor("b_context", ane_model.b_context)

bad = False
for layer_idx in range(N_DFLASH_LAYERS):
    if bad:
        break
    print(f"\n--- Layer {layer_idx} ---")
    p = f"l{layer_idx}_"

    k['q_proj'].run_uncached(
        [ane_model.b_hidden, getattr(ane_model, f"w_{p}in_norm"), getattr(ane_model, f"w_{p}q_proj")],
        [ane_model.b_q_out],
    )
    bad = check_tensor("b_q_out", ane_model.b_q_out)

    ane_model._flat_to_4d_norm(ane_model.b_q_out, N_HEADS, ane_model.w_sq, ane_model.b_q_norm_4d)
    k['q_norm_4d'].run_uncached([ane_model.b_q_norm_4d, getattr(ane_model, f"w_{p}q_norm_4d")], [ane_model.b_q_norm_4d])
    bad = bad or check_tensor("b_q_norm_4d", ane_model.b_q_norm_4d)

    ane_model._4d_norm_to_4d_heads(ane_model.b_q_norm_4d, N_HEADS, ane_model.w_sq, ane_model.b_q_4d)

    k['k_proj_ctx'].run_uncached([ane_model.b_context, getattr(ane_model, f"w_{p}k_proj")], [ane_model.b_k_ctx])
    k['k_proj_noise'].run_uncached([ane_model.b_hidden, getattr(ane_model, f"w_{p}in_norm"), getattr(ane_model, f"w_{p}k_proj")], [ane_model.b_k_noise])
    k['k_concat'].run_uncached([ane_model.b_k_ctx, ane_model.b_k_noise], [ane_model.b_k_out])

    ane_model._flat_to_4d_norm(ane_model.b_k_out, N_KV_HEADS, ane_model.w_kv, ane_model.b_k_norm_4d)
    k['k_norm_4d'].run_uncached([ane_model.b_k_norm_4d, getattr(ane_model, f"w_{p}k_norm_4d")], [ane_model.b_k_norm_4d])
    ane_model._4d_norm_to_4d_heads(ane_model.b_k_norm_4d, N_KV_HEADS, ane_model.w_kv, ane_model.b_k_4d)

    k['rope_q'].run_uncached([ane_model.b_q_4d, ane_model.b_cos_q, ane_model.b_sin_q], [ane_model.b_q_rope_4d])
    bad = bad or check_tensor("b_q_rope_4d", ane_model.b_q_rope_4d)
    k['rope_k'].run_uncached([ane_model.b_k_4d, ane_model.b_cos_k, ane_model.b_sin_k], [ane_model.b_k_rope_4d])
    bad = bad or check_tensor("b_k_rope_4d", ane_model.b_k_rope_4d)

    k['v_proj_ctx'].run_uncached([ane_model.b_context, getattr(ane_model, f"w_{p}v_proj")], [ane_model.b_v_ctx])
    k['v_proj_noise'].run_uncached([ane_model.b_hidden, getattr(ane_model, f"w_{p}in_norm"), getattr(ane_model, f"w_{p}v_proj")], [ane_model.b_v_noise])
    k['v_concat'].run_uncached([ane_model.b_v_ctx, ane_model.b_v_noise], [ane_model.b_v_out])
    ane_model._flat_to_4d_heads(ane_model.b_v_out, N_KV_HEADS, ane_model.w_kv, ane_model.b_v_4d)

    k['gqa_tile'].run_uncached([ane_model.b_k_rope_4d, ane_model.b_v_4d], [ane_model.b_kv_tiled])
    k['attn_out'].run_uncached([ane_model.b_q_rope_4d, ane_model.b_kv_tiled], [ane_model.b_attn_out])
    bad = bad or check_tensor("b_attn_out", ane_model.b_attn_out)

    ane_model._flatten_attn_4d()
    k['o_proj_residual'].run_uncached(
        [ane_model.b_attn_flat, getattr(ane_model, f"w_{p}o_proj"), ane_model.b_hidden],
        [ane_model.b_attn_res],
    )
    bad = bad or check_tensor("b_attn_res", ane_model.b_attn_res)

    k['ffn_residual'].run_uncached(
        [ane_model.b_attn_res, getattr(ane_model, f"w_{p}post_norm"),
         getattr(ane_model, f"w_{p}gate"), getattr(ane_model, f"w_{p}up"), getattr(ane_model, f"w_{p}down")],
        [ane_model.b_hidden],
    )
    bad = bad or check_tensor("b_hidden (next)", ane_model.b_hidden)

if not bad:
    print("\n--- final_norm ---")
    k['final_norm'].run_uncached([ane_model.b_hidden, ane_model.w_final_norm], [ane_model.b_output])
    check_tensor("b_output", ane_model.b_output)
