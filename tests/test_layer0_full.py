"""Compare full layer 0 output (after ffn_residual) against GPU."""

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
noise_embedding = mx.random.normal(shape=(1, SEQ_Q, HIDDEN), dtype=mx.bfloat16)
target_hidden = mx.random.normal(shape=(1, CTX_LEN, TARGET_HIDDEN), dtype=mx.bfloat16)

# GPU: full forward
gpu_out = draft_model(noise_embedding, target_hidden)  # [1, SEQ_Q, HIDDEN]

# ANE
print("Building ANE model...")
ane_model = ANEDraftModel(seq_q=SEQ_Q, ctx_len=CTX_LEN)
ane_model.load_weights(draft_model)
ane_out = ane_model.forward(noise_embedding, target_hidden, rope_offset=0)

print(f"GPU output range: [{float(gpu_out.min()):.2f}, {float(gpu_out.max()):.2f}]")
print(f"ANE output range: [{float(ane_out.min()):.2f}, {float(ane_out.max()):.2f}]")
cos = cosine_sim(ane_out, gpu_out)
print(f"Full output cosine: {cos:.6f}")

# Check intermediate: run GPU layer 0 step by step
layer = draft_model.layers[0]
hidden_f = noise_embedding.astype(mx.float32)

# The DFlashDraftModel.__call__ does:
# hidden_states = noise_embedding (no norm on noise)
# target_hidden = hidden_norm(fc(target_hidden))
# Then for each layer: residual = hidden_states; hidden = input_layernorm(hidden_states); hidden = self_attn(hidden, target_hidden); hidden = residual + hidden; ...
# So: residual = noise_embedding, hidden = input_layernorm(noise_embedding), then attn output = self_attn(normed, context)
# final = residual + o_proj(attn_output) → then ffn

# Compare o_proj + residual
# GPU: residual = noise_embedding
# GPU: attn_output = self_attn(input_layernorm(noise), context)
# The o_proj is inside self_attn

# ANE: b_attn_res = b_hidden + o_proj(attn_flat)
# where b_hidden is the raw noise_embedding (no norm)
# This should match: residual + o_proj(attn_output)

# Check if the residual in o_proj_residual kernel uses the correct buffer
# The o_proj_residual kernel takes [b_attn_flat, w_o_proj, b_hidden] → b_attn_res
# It computes: o_proj(attn_flat) + b_hidden
# b_hidden should be the RESIDUAL (input to the layer, before layernorm)

# In the current flow, b_hidden is set to noise_embedding at the start
# and then overwritten by ffn_residual at the end of each layer
# So for layer 0, b_hidden = noise_embedding (correct residual)
# For layer 1+, b_hidden = output of previous layer's ffn_residual (correct)

# The issue might be that ffn_residual also adds the residual (b_attn_res = o_proj + residual)
# and then ffn_residual adds ANOTHER residual: b_attn_res + ffn_output → b_hidden
# So the final hidden = noise_embedding + o_proj(attn) + ffn(attn_res_normed)

# Let me check what the GPU does
# Qwen3DFlashDecoderLayer.__call__:
#   residual = hidden_states
#   hidden_states = self.input_layernorm(hidden_states)
#   hidden_states = self.self_attn(hidden_states=hidden_states, target_hidden=target_hidden)
#   hidden_states = residual + hidden_states     ← this is o_proj + residual
#   residual = hidden_states
#   hidden_states = self.post_attention_layernorm(hidden_states)
#   hidden_states = self.mlp(hidden_states)
#   hidden_states = residual + hidden_states     ← this is ffn + residual
# So: final = noise_embedding + o_proj(attn) + mlp(post_norm(noise_embedding + o_proj(attn)))

# ANE o_proj_residual: o_proj(attn) + noise_embedding ✓
# ANE ffn_residual: ffn(post_norm(o_proj+noise)) + (o_proj+noise) ✓

# Let me compare the GPU layer 0 output directly
print("\nRunning GPU layer 0 step by step...")
from mirror_sd.dflash import Qwen3DFlashDecoderLayer
gpu_hidden = noise_embedding.astype(mx.float32)
gpu_context = draft_model.hidden_norm(draft_model.fc(target_hidden.astype(mx.float32)))
gpu_layer0_out = draft_model.layers[0](gpu_hidden, gpu_context)
print(f"GPU layer 0 output range: [{float(gpu_layer0_out.min()):.2f}, {float(gpu_layer0_out.max()):.2f}]")

# ANE layer 0 output (b_hidden after ffn_residual)
ane_hidden = ane_model._read_mlx_2d(ane_model.b_hidden, SEQ_Q, HIDDEN).astype(mx.float32)
cos_l0 = cosine_sim(ane_hidden, gpu_layer0_out)
print(f"ANE layer 0 output vs GPU: cos={cos_l0:.6f}")
