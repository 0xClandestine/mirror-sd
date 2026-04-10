"""Diagnostic: step through ANE kernels one by one and check for NaN/Inf.

Runs each kernel individually and inspects the output to find where
the pipeline first produces invalid values.
"""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, TARGET_HIDDEN, INTERMEDIATE
from mirror_sd.loader import load_dflash_model

SEQ_Q = 16
CTX_LEN = 64
DRAFT_PATH = "z-lab/Qwen3-8B-DFlash-b16"


def check_tensor(name, tensor, expected_ch=None, expected_w=None):
    data = tensor.read_f32()
    n_nan = sum(1 for x in data if x != x)  # NaN check
    n_inf = sum(1 for x in data if x == x and abs(x) == float('inf'))
    n_zero = sum(1 for x in data if x == 0.0)
    shape = tensor.shape
    if n_nan > 0 or n_inf > 0:
        print(f"  {name}: shape={shape} NaN={n_nan} Inf={n_inf} — BAD")
    else:
        sample = data[:8]
        print(f"  {name}: shape={shape} NaN={n_nan} Inf={n_inf} zero={n_zero} sample={sample}")


def main():
    print("Loading draft model...")
    draft_model, config = load_dflash_model(DRAFT_PATH)

    print(f"Building ANE model...")
    ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
    print("Loading ANE weights...")
    ane_model.load_weights(draft_model)

    mx.random.seed(42)
    noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
    target_hidden = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16)

    # Write inputs
    ane_model._write_mlx_2d(ane_model.b_noise, noise_embedding)
    ane_model._write_mlx_2d(ane_model.b_target, target_hidden)

    # Copy noise to hidden
    noise_data = ane_model.b_noise.read_f32()
    ane_model.b_hidden.write_f32(noise_data)

    # Compute RoPE
    ane_model._compute_rope(0)

    check_tensor("b_noise", ane_model.b_noise)
    check_tensor("b_target", ane_model.b_target)
    check_tensor("b_hidden", ane_model.b_hidden)

    # Run fc_norm
    print("\n--- fc_norm ---")
    ane_model.kernels['fc_norm'].run(
        [ane_model.b_target, ane_model.w_fc, ane_model.w_hidden_norm],
        [ane_model.b_context],
    )
    check_tensor("b_context", ane_model.b_context)

    # Run layer 0
    for layer_idx in range(5):
        print(f"\n--- Layer {layer_idx} ---")
        p = f"l{layer_idx}_"
        k = ane_model.kernels

        print("  q_kernel...")
        k['q_kernel'].run(
            [ane_model.b_hidden, getattr(ane_model, f"w_{p}in_norm"),
             getattr(ane_model, f"w_{p}q_proj"), getattr(ane_model, f"w_{p}q_norm")],
            [ane_model.b_q_out],
        )
        check_tensor("b_q_out", ane_model.b_q_out)

        print("  k_proj_ctx...")
        k['k_proj_ctx'].run(
            [ane_model.b_context, getattr(ane_model, f"w_{p}k_proj")],
            [ane_model.b_k_ctx],
        )
        check_tensor("b_k_ctx", ane_model.b_k_ctx)

        print("  k_proj_noise...")
        k['k_proj_noise'].run(
            [ane_model.b_hidden, getattr(ane_model, f"w_{p}in_norm"), getattr(ane_model, f"w_{p}k_proj")],
            [ane_model.b_k_noise],
        )
        check_tensor("b_k_noise", ane_model.b_k_noise)

        print("  k_concat (K)...")
        k['k_concat'].run([ane_model.b_k_ctx, ane_model.b_k_noise], [ane_model.b_k_out])
        check_tensor("b_k_out", ane_model.b_k_out)

        print("  k_norm...")
        k['k_norm'].run(
            [ane_model.b_k_out, getattr(ane_model, f"w_{p}k_norm")],
            [ane_model.b_k_normed],
        )
        check_tensor("b_k_normed", ane_model.b_k_normed)

        print("  v_proj_ctx...")
        k['v_proj_ctx'].run(
            [ane_model.b_context, getattr(ane_model, f"w_{p}v_proj")],
            [ane_model.b_v_ctx],
        )
        check_tensor("b_v_ctx", ane_model.b_v_ctx)

        print("  v_proj_noise...")
        k['v_proj_noise'].run(
            [ane_model.b_hidden, getattr(ane_model, f"w_{p}in_norm"), getattr(ane_model, f"w_{p}v_proj")],
            [ane_model.b_v_noise],
        )
        check_tensor("b_v_noise", ane_model.b_v_noise)

        print("  v_concat (V)...")
        k['v_concat'].run([ane_model.b_v_ctx, ane_model.b_v_noise], [ane_model.b_v_out])
        check_tensor("b_v_out", ane_model.b_v_out)

        print("  rope_q...")
        k['rope_q'].run(
            [ane_model.b_q_out, ane_model.b_cos_q, ane_model.b_sin_q],
            [ane_model.b_q_rope],
        )
        check_tensor("b_q_rope", ane_model.b_q_rope)

        print("  rope_k...")
        k['rope_k'].run(
            [ane_model.b_k_normed, ane_model.b_cos_k, ane_model.b_sin_k],
            [ane_model.b_k_rope],
        )
        check_tensor("b_k_rope", ane_model.b_k_rope)

        print("  gqa_tile...")
        k['gqa_tile'].run(
            [ane_model.b_k_rope, ane_model.b_v_out],
            [ane_model.b_kv_tiled],
        )
        check_tensor("b_kv_tiled", ane_model.b_kv_tiled)

        print("  attn_residual...")
        k['attn_residual'].run(
            [ane_model.b_q_rope, ane_model.b_kv_tiled, getattr(ane_model, f"w_{p}o_proj"), ane_model.b_hidden],
            [ane_model.b_attn_res],
        )
        check_tensor("b_attn_res", ane_model.b_attn_res)

        print("  ffn_residual...")
        k['ffn_residual'].run(
            [ane_model.b_attn_res, getattr(ane_model, f"w_{p}post_norm"),
             getattr(ane_model, f"w_{p}gate"), getattr(ane_model, f"w_{p}up"), getattr(ane_model, f"w_{p}down")],
            [ane_model.b_hidden],
        )
        check_tensor("b_hidden (next)", ane_model.b_hidden)

    print("\n--- final_norm ---")
    ane_model.kernels['final_norm'].run(
        [ane_model.b_hidden, ane_model.w_final_norm],
        [ane_model.b_output],
    )
    check_tensor("b_output", ane_model.b_output)


if __name__ == "__main__":
    main()