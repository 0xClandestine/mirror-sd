"""Layer-by-layer ANE diagnostic test.

Isolates where quality degradation occurs between ANE and GPU paths.

Three measurements:
  1. Context: cosine(ane._compute_context(t), gpu.hidden_norm(gpu.fc(t)))
  2. Isolated per-layer: each layer fed GPU-computed input + GPU context,
     comparing GPU output vs ANE output (no error propagation).
  3. Chained: errors accumulate layer-to-layer (uses GPU context throughout
     to isolate layer error from context error).
"""

import argparse
import mlx.core as mx
import mlx.nn as nn

from mirror_sd.loader import load_dflash_model
from mirror_sd.ane_model import ANEDraftModel

SEQ_Q = 32
CTX_LEN = 64


def cosine_sim(a: mx.array, b: mx.array) -> float:
    a = a.flatten().astype(mx.float32)
    b = b.flatten().astype(mx.float32)
    return float((a * b).sum() / (mx.sqrt((a * a).sum()) * mx.sqrt((b * b).sum()) + 1e-12))


def write_hidden(ane: ANEDraftModel, h: mx.array) -> None:
    ane._write_padded(ane._padded_hidden, ane.b_hidden, h)


def write_context(ane: ANEDraftModel, ctx: mx.array) -> None:
    ane._write_padded(ane._padded_context, ane.b_context, ctx)


def read_hidden(ane: ANEDraftModel) -> mx.array:
    return ane._read_mlx_2d(ane.b_hidden, ane.seq_q, ane.hidden)


def run_diagnostics(draft_model, config, model_label: str) -> None:
    H = config.hidden_size

    print(f"\n{'='*64}")
    print(f"  {model_label}")
    print(f"  hidden={H}  intermediate={config.intermediate_size}")
    print(f"  n_heads={config.num_attention_heads}  n_kv_heads={config.num_key_value_heads}")
    print(f"  layers={config.num_hidden_layers}  SEQ_Q={SEQ_Q}  CTX_LEN={CTX_LEN}")
    print(f"{'='*64}\n")

    mx.random.seed(42)
    noise_emb = mx.random.normal((1, SEQ_Q, H), dtype=mx.float32) * 0.02
    target_hid = mx.random.normal((1, CTX_LEN, 5 * H), dtype=mx.float32) * 0.02
    mx.eval(noise_emb, target_hid)

    # GPU reference context (projected + normed, used as target_hidden in all layers)
    gpu_context = draft_model.hidden_norm(draft_model.fc(target_hid))
    mx.eval(gpu_context)

    # --- Build ANE model ---
    ane = ANEDraftModel(SEQ_Q, CTX_LEN, config=config)
    ane.load_weights(draft_model)
    print()

    # 1. Context comparison
    ane_context = ane._compute_context(target_hid)
    mx.eval(ane_context)
    ctx_cos = cosine_sim(ane_context, gpu_context)
    ctx_gpu_rms = float(mx.sqrt(mx.mean(gpu_context.astype(mx.float32) ** 2)))
    ctx_ane_rms = float(mx.sqrt(mx.mean(ane_context.astype(mx.float32) ** 2)))
    print(f"[1] Context (fc + norm): cosine={ctx_cos:.6f}  "
          f"gpu_rms={ctx_gpu_rms:.4f}  ane_rms={ctx_ane_rms:.4f}")
    print()

    # Set up RoPE + attn mask (constant for all layer tests)
    ane._compute_rope(0, CTX_LEN)
    ane._compute_attn_mask(CTX_LEN)
    k = ane.kernels

    # Write GPU context into b_context so layer tests use the same reference
    write_context(ane, gpu_context)

    # 2. Per-layer isolated test — each layer sees the same GPU-computed noise_emb
    print("[2] Per-layer isolated (GPU input → each layer independently):")
    for i in range(config.num_hidden_layers):
        # GPU: layer i with clean noise input
        gpu_out = draft_model.layers[i](
            hidden_states=noise_emb,
            target_hidden=gpu_context,
            mask=None,
            cache=None,
        )
        mx.eval(gpu_out)

        # ANE: layer i with same input
        write_hidden(ane, noise_emb)
        ane._run_layer(k, i)
        ane_out = read_hidden(ane)
        mx.eval(ane_out)

        cos = cosine_sim(ane_out, gpu_out)
        g_rms = float(mx.sqrt(mx.mean(gpu_out.astype(mx.float32) ** 2)))
        a_rms = float(mx.sqrt(mx.mean(ane_out.astype(mx.float32) ** 2)))
        print(f"  Layer {i}: cosine={cos:.6f}  gpu_rms={g_rms:.4f}  ane_rms={a_rms:.4f}")

    print()

    # 3. Chained test — ANE errors accumulate, GPU context used throughout
    print("[3] Chained (ANE error accumulates, GPU context):")
    gpu_h = noise_emb
    ane_h = noise_emb
    for i in range(config.num_hidden_layers):
        gpu_h = draft_model.layers[i](
            hidden_states=gpu_h,
            target_hidden=gpu_context,
            mask=None,
            cache=None,
        )
        mx.eval(gpu_h)

        write_hidden(ane, ane_h)
        ane._run_layer(k, i)
        ane_h = read_hidden(ane)
        mx.eval(ane_h)

        cos = cosine_sim(ane_h, gpu_h)
        print(f"  After layer {i}: cosine={cos:.6f}")

    # Final norm
    gpu_final = draft_model.norm(gpu_h)
    mx.eval(gpu_final)

    write_hidden(ane, ane_h)
    k['final_norm'].run_uncached(
        [ane.b_hidden, ane.w_final_norm],
        [ane.b_output],
    )
    ane_final = ane._read_mlx_2d(ane.b_output, ane.seq_q, ane.hidden)
    mx.eval(ane_final)

    final_cos = cosine_sim(ane_final, gpu_final)
    print(f"  After final_norm: cosine={final_cos:.6f}")
    print()

    # 4. Sub-layer diagnostic: attention path vs FFN path per layer
    print("[4] Sub-layer: attn_residual vs ffn_residual (GPU input to each):")
    print(f"  {'Layer':<8} {'post-attn cosine':<22} {'post-ffn cosine'}")
    for i in range(config.num_hidden_layers):
        # GPU attention path
        h_norm = draft_model.layers[i].input_layernorm(noise_emb)
        attn_out = draft_model.layers[i].self_attn(
            hidden_states=h_norm,
            target_hidden=gpu_context,
            mask=None,
            cache=None,
        )
        gpu_after_attn = noise_emb + attn_out
        mx.eval(gpu_after_attn)

        # GPU FFN path
        h2_norm = draft_model.layers[i].post_attention_layernorm(gpu_after_attn)
        ffn_out = draft_model.layers[i].mlp(h2_norm)
        gpu_after_ffn = gpu_after_attn + ffn_out
        mx.eval(gpu_after_ffn)

        # ANE: run individual kernels, read b_attn_res after o_proj_residual
        write_hidden(ane, noise_emb)
        p = f"l{i}_"
        k['mega_qkv'].run_uncached(
            [ane.b_hidden, getattr(ane, f"w_{p}in_norm"),
             ane.b_context,
             getattr(ane, f"w_{p}k_proj"),
             getattr(ane, f"w_{p}k_norm_4d"),
             ane.b_cos_k, ane.b_sin_k,
             getattr(ane, f"w_{p}v_proj"),
             getattr(ane, f"w_{p}q_proj"),
             getattr(ane, f"w_{p}q_norm_4d"),
             ane.b_cos_q, ane.b_sin_q],
            [ane.b_k_rope_4d, ane.b_v_4d_t, ane.b_q_rope_4d],
        )
        k['gqa_tile'].run_uncached([ane.b_k_rope_4d, ane.b_v_4d_t], [ane.b_kv_tiled])
        k['attn_out'].run_uncached([ane.b_q_rope_4d, ane.b_kv_tiled, ane.b_attn_mask], [ane.b_attn_flat])
        k['o_proj_residual'].run_uncached(
            [ane.b_attn_flat, getattr(ane, f"w_{p}o_proj"), ane.b_hidden],
            [ane.b_attn_res],
        )
        ane_after_attn = ane._read_mlx_2d(ane.b_attn_res, ane.seq_q, ane.hidden)
        mx.eval(ane_after_attn)
        attn_cos = cosine_sim(ane_after_attn, gpu_after_attn)

        # ANE: FFN on top of b_attn_res
        k['ffn_residual'].run_uncached(
            [ane.b_attn_res, getattr(ane, f"w_{p}post_norm"),
             getattr(ane, f"w_{p}gate"), getattr(ane, f"w_{p}up"), getattr(ane, f"w_{p}down")],
            [ane.b_hidden],
        )
        ane_after_ffn = read_hidden(ane)
        mx.eval(ane_after_ffn)
        ffn_cos = cosine_sim(ane_after_ffn, gpu_after_ffn)

        print(f"  Layer {i}:  {attn_cos:<22.6f} {ffn_cos:.6f}")

    print()

    # 5. Full forward pass sanity check
    gpu_full = draft_model(noise_emb, target_hid)
    mx.eval(gpu_full)

    ane._write_padded(ane._padded_hidden, ane.b_hidden, noise_emb)
    write_context(ane, ane_context)  # use ANE context for full pass
    ane.run_kernels()
    ane_full = ane.read_output()
    mx.eval(ane_full)

    full_cos = cosine_sim(ane_full, gpu_full)
    print(f"[5] Full forward pass (ANE context): cosine={full_cos:.6f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="z-lab/Qwen3-8B-DFlash-b16")
    args = parser.parse_args()

    print(f"Loading {args.model}...")
    draft_model, config = load_dflash_model(args.model)
    print("Loaded.")

    run_diagnostics(draft_model, config, args.model)


if __name__ == "__main__":
    main()
