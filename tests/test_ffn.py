"""Test ffn_residual and attn_residual kernels in isolation."""

import sys
sys.path.insert(0, "/Users/gg/Documents/GitHub/mirror-sd")

import mlx.core as mx
from mirror_sd.ane_model import ANEDraftModel, HIDDEN, N_HEADS, HEAD_DIM, N_KV_HEADS, INTERMEDIATE
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

def mlx_rmsnorm(x, weight, eps=1e-6):
    ms = x.mean(axis=-1, keepdims=True)
    diff = x - ms
    sq = diff * diff
    mean_sq = sq.mean(axis=-1, keepdims=True)
    inv_std = mx.rsqrt(mean_sq + eps)
    return x * inv_std * weight

print("Loading...")
draft_model, config = load_dflash_model("z-lab/Qwen3-8B-DFlash-b16")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)

layer = draft_model.layers[0]

# Create a known hidden state
mx.random.seed(42)
hidden = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)

# --- Test ffn_residual directly ---
# ffn_residual: rmsnorm(attn_res) → gate_proj, up_proj → swiglu → down_proj → residual add
post_norm_w = layer.post_attention_layernorm.weight.astype(mx.float32)
gate_w = layer.mlp.gate_proj.weight.astype(mx.float32)
up_w = layer.mlp.up_proj.weight.astype(mx.float32)
down_w = layer.mlp.down_proj.weight.astype(mx.float32)

hidden_f = hidden.astype(mx.float32)
normed = mlx_rmsnorm(hidden_f, post_norm_w)
gate_out = normed @ gate_w.T
up_out = normed @ up_w.T
silu = gate_out * (1.0 / (1.0 + mx.exp(-gate_out)))
gate = silu * up_out
down_out = gate @ down_w.T
ffn_expected = hidden_f + down_out

# ANE
ane_model._write_mlx_2d(ane_model.b_hidden, hidden)
ane_model.kernels['ffn_residual'].run(
    [ane_model.b_hidden, ane_model.w_l0_post_norm,
     ane_model.w_l0_gate, ane_model.w_l0_up, ane_model.w_l0_down],
    [ane_model.b_hidden],  # output overwrites input
)
ffn_ane = ane_model._read_mlx_2d(ane_model.b_hidden, SEQ_Q, HIDDEN)

cos = cosine_sim(ffn_expected, ffn_ane)
print(f"ffn_residual cosine: {cos:.6f}")
print(f"  Expected sample: {ffn_expected[0, 0, :4]}")
print(f"  ANE sample:      {ffn_ane[0, 0, :4]}")
