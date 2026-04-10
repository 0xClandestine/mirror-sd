"""DFlash draft model running on Apple Neural Engine.

Chains 11 ANE kernels (9 per decoder layer + fc_norm + final_norm)
with IOSurface buffers for intermediate results and real weight loading.

Data flow per layer:
  hidden ──┬──→ q_kernel ──→ rope_q ──→ (4D) ──┐
           │                                    ├──→ attn_residual ──→ attn_res ──→ ffn_residual ──→ hidden_next
  context ─┼──→ k_proj ──→ k_norm ──→ rope_k ──┐│   (residual input = hidden)
           │                    └──→ gqa_tile ──┘│
           └──→ v_proj ──────────────→ gqa_tile ─┘

Note: attn_residual writes to b_attn_res (NOT b_hidden). ffn_residual reads
b_attn_res and writes to b_hidden. This avoids read-write hazards on b_hidden.
"""

import math
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from .dflash import DFlashConfig


HIDDEN = 4096
HEAD_DIM = 128
N_HEADS = 32
N_KV_HEADS = 8
INTERMEDIATE = 12288
TARGET_HIDDEN = 5 * HIDDEN
MIN_SPATIAL_WIDTH = 64
N_DFLASH_LAYERS = 5


def align_width(w: int) -> int:
    aligned = ((w + MIN_SPATIAL_WIDTH - 1) // MIN_SPATIAL_WIDTH) * MIN_SPATIAL_WIDTH
    return max(aligned, MIN_SPATIAL_WIDTH)


class ANEDraftModel:
    def __init__(self, seq_q: int, ctx_len: int):
        import mirror_sd_ane as ane

        self.ane = ane
        self.seq_q = seq_q
        self.ctx_len = ctx_len
        self.w_sq = align_width(seq_q)
        self.w_ctx = align_width(ctx_len)
        self.w_kv = self.w_ctx + self.w_sq

        print(f"[ANE] Compiling 11 kernels (seq_q={seq_q}, ctx_len={ctx_len}, "
              f"w_sq={self.w_sq}, w_ctx={self.w_ctx}, w_kv={self.w_kv})...")
        self.kernels = {k.name: k for k in ane.compile_dflash_kernels(seq_q, ctx_len)}
        print(f"[ANE] All {len(self.kernels)} kernels compiled")

        self._alloc_buffers()
        self.weights_loaded = False

    def _alloc_buffers(self):
        ane = self.ane
        w_sq, w_ctx, w_kv = self.w_sq, self.w_ctx, self.w_kv

        self.b_noise = ane.ANETensor(1, HIDDEN, 1, w_sq)
        self.b_target = ane.ANETensor(1, TARGET_HIDDEN, 1, w_ctx)

        self.b_context = ane.ANETensor(1, HIDDEN, 1, w_ctx)
        self.b_hidden = ane.ANETensor(1, HIDDEN, 1, w_sq)

        self.b_q_out = ane.ANETensor(1, N_HEADS * HEAD_DIM, 1, w_sq)
        self.b_k_out = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_kv)
        self.b_k_normed = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_kv)
        self.b_v_out = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_kv)

        self.b_q_rope = ane.ANETensor(1, N_HEADS, w_sq, HEAD_DIM)
        self.b_k_rope = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_kv)

        self.b_kv_tiled = ane.ANETensor(1, 2 * N_HEADS, w_kv, HEAD_DIM)

        self.b_attn_res = ane.ANETensor(1, HIDDEN, 1, w_sq)

        self.b_output = ane.ANETensor(1, HIDDEN, 1, w_sq)

        self.b_cos_q = ane.ANETensor(1, 1, w_sq, HEAD_DIM)
        self.b_sin_q = ane.ANETensor(1, 1, w_sq, HEAD_DIM)
        self.b_cos_k = ane.ANETensor(1, 1, w_kv, HEAD_DIM)
        self.b_sin_k = ane.ANETensor(1, 1, w_kv, HEAD_DIM)

    def _make_weight(self, data_flat, oc, ic, height=1):
        return self.ane.ANETensor.from_f32(1, oc, height, ic, data_flat)

    def _make_norm_weight_expanded(self, weight_list, channels, width):
        w = align_width(width)
        data = []
        for c in range(channels):
            val = weight_list[c]
            data.extend([val] * w)
        return self.ane.ANETensor.from_f32(1, channels, 1, w, data)

    def _make_per_head_norm_weight(self, head_weight_list, n_heads, width):
        w = align_width(width)
        head_dim = len(head_weight_list)
        data = []
        for h in range(n_heads):
            for d in range(head_dim):
                val = head_weight_list[d]
                data.extend([val] * w)
        return self.ane.ANETensor.from_f32(1, n_heads * head_dim, 1, w, data)

    def load_weights(self, draft_model: nn.Module, target_model: nn.Module = None):
        self._load_fc_weights(draft_model)
        for i in range(N_DFLASH_LAYERS):
            self._load_layer_weights(draft_model, i)
        self._load_final_norm_weights(draft_model)
        self.weights_loaded = True
        print("[ANE] Weights loaded")

    def _mlx_to_f32_list(self, arr: mx.array) -> list:
        return arr.astype(mx.float32).flatten().tolist()

    def _load_fc_weights(self, model: nn.Module):
        fc_w = self._mlx_to_f32_list(model.fc.weight)
        self.w_fc = self._make_weight(fc_w, HIDDEN, TARGET_HIDDEN)
        hidden_norm_w = self._mlx_to_f32_list(model.hidden_norm.weight)
        self.w_hidden_norm = self._make_norm_weight_expanded(hidden_norm_w, HIDDEN, self.w_ctx)

    def _load_layer_weights(self, model: nn.Module, layer_idx: int):
        layer = model.layers[layer_idx]
        p = f"l{layer_idx}_"

        in_norm_w = self._mlx_to_f32_list(layer.input_layernorm.weight)
        setattr(self, f"w_{p}in_norm", self._make_norm_weight_expanded(in_norm_w, HIDDEN, self.w_sq))

        q_proj_w = self._mlx_to_f32_list(layer.self_attn.q_proj.weight)
        setattr(self, f"w_{p}q_proj", self._make_weight(q_proj_w, N_HEADS * HEAD_DIM, HIDDEN))

        q_norm_w = self._mlx_to_f32_list(layer.self_attn.q_norm.weight)
        setattr(self, f"w_{p}q_norm", self._make_per_head_norm_weight(q_norm_w, N_HEADS, self.w_sq))

        k_proj_w = self._mlx_to_f32_list(layer.self_attn.k_proj.weight)
        setattr(self, f"w_{p}k_proj", self._make_weight(k_proj_w, N_KV_HEADS * HEAD_DIM, HIDDEN))

        k_norm_w = self._mlx_to_f32_list(layer.self_attn.k_norm.weight)
        setattr(self, f"w_{p}k_norm", self._make_per_head_norm_weight(k_norm_w, N_KV_HEADS, self.w_kv))

        v_proj_w = self._mlx_to_f32_list(layer.self_attn.v_proj.weight)
        setattr(self, f"w_{p}v_proj", self._make_weight(v_proj_w, N_KV_HEADS * HEAD_DIM, HIDDEN))

        o_proj_w = self._mlx_to_f32_list(layer.self_attn.o_proj.weight)
        setattr(self, f"w_{p}o_proj", self._make_weight(o_proj_w, HIDDEN, N_HEADS * HEAD_DIM))

        post_norm_w = self._mlx_to_f32_list(layer.post_attention_layernorm.weight)
        setattr(self, f"w_{p}post_norm", self._make_norm_weight_expanded(post_norm_w, HIDDEN, self.w_sq))

        gate_w = self._mlx_to_f32_list(layer.mlp.gate_proj.weight)
        setattr(self, f"w_{p}gate", self._make_weight(gate_w, INTERMEDIATE, HIDDEN))

        up_w = self._mlx_to_f32_list(layer.mlp.up_proj.weight)
        setattr(self, f"w_{p}up", self._make_weight(up_w, INTERMEDIATE, HIDDEN))

        down_w = self._mlx_to_f32_list(layer.mlp.down_proj.weight)
        setattr(self, f"w_{p}down", self._make_weight(down_w, HIDDEN, INTERMEDIATE))

    def _load_final_norm_weights(self, model: nn.Module):
        norm_w = self._mlx_to_f32_list(model.norm.weight)
        self.w_final_norm = self._make_norm_weight_expanded(norm_w, HIDDEN, self.w_sq)

    def forward(self, noise_embedding_f32: list, target_hidden_f32: list, rope_offset: int = 0) -> list:
        k = self.kernels
        self.b_noise.write_f32(noise_embedding_f32)
        self.b_target.write_f32(target_hidden_f32)

        noise_data = self.b_noise.read_f32()
        self.b_hidden.write_f32(noise_data)

        self._compute_rope(rope_offset)

        k['fc_norm'].run(
            [self.b_target, self.w_fc, self.w_hidden_norm],
            [self.b_context],
        )

        for i in range(N_DFLASH_LAYERS):
            self._run_layer(k, i)

        k['final_norm'].run(
            [self.b_hidden, self.w_final_norm],
            [self.b_output],
        )

        return self.b_output.read_f32()

    def _run_layer(self, k, layer_idx: int):
        p = f"l{layer_idx}_"

        k['q_kernel'].run(
            [self.b_hidden, getattr(self, f"w_{p}in_norm"),
             getattr(self, f"w_{p}q_proj"), getattr(self, f"w_{p}q_norm")],
            [self.b_q_out],
        )

        k['k_proj'].run(
            [self.b_context, self.b_hidden, getattr(self, f"w_{p}in_norm"),
             getattr(self, f"w_{p}k_proj")],
            [self.b_k_out],
        )

        k['k_norm'].run(
            [self.b_k_out, getattr(self, f"w_{p}k_norm")],
            [self.b_k_normed],
        )

        k['v_proj'].run(
            [self.b_context, self.b_hidden, getattr(self, f"w_{p}in_norm"),
             getattr(self, f"w_{p}v_proj")],
            [self.b_v_out],
        )

        k['rope_q'].run(
            [self.b_q_out, self.b_cos_q, self.b_sin_q],
            [self.b_q_rope],
        )

        k['rope_k'].run(
            [self.b_k_normed, self.b_cos_k, self.b_sin_k],
            [self.b_k_rope],
        )

        k['gqa_tile'].run(
            [self.b_k_rope, self.b_v_out],
            [self.b_kv_tiled],
        )

        k['attn_residual'].run(
            [self.b_q_rope, self.b_kv_tiled, getattr(self, f"w_{p}o_proj"), self.b_hidden],
            [self.b_attn_res],
        )

        k['ffn_residual'].run(
            [self.b_attn_res, getattr(self, f"w_{p}post_norm"),
             getattr(self, f"w_{p}gate"), getattr(self, f"w_{p}up"), getattr(self, f"w_{p}down")],
            [self.b_hidden],
        )

    def _compute_rope(self, rope_offset: int):
        rope_theta = 1000000.0

        cos_q_data = []
        sin_q_data = []
        for pos in range(self.w_sq):
            angle_pos = rope_offset + self.ctx_len + pos
            for d in range(HEAD_DIM // 2):
                freq = 1.0 / (rope_theta ** (2.0 * d / HEAD_DIM))
                angle = angle_pos * freq
                cos_q_data.append(math.cos(angle))
                sin_q_data.append(math.sin(angle))
            for _ in range(HEAD_DIM // 2):
                cos_q_data.append(1.0)
                sin_q_data.append(0.0)

        self.b_cos_q.write_f32(cos_q_data)
        self.b_sin_q.write_f32(sin_q_data)

        cos_k_data = []
        sin_k_data = []
        for pos in range(self.w_kv):
            if pos < self.ctx_len:
                angle_pos = rope_offset + pos
            else:
                angle_pos = rope_offset + self.ctx_len + (pos - self.ctx_len)
            for d in range(HEAD_DIM // 2):
                freq = 1.0 / (rope_theta ** (2.0 * d / HEAD_DIM))
                angle = angle_pos * freq
                cos_k_data.append(math.cos(angle))
                sin_k_data.append(math.sin(angle))
            for _ in range(HEAD_DIM // 2):
                cos_k_data.append(1.0)
                sin_k_data.append(0.0)

        self.b_cos_k.write_f32(cos_k_data)
        self.b_sin_k.write_f32(sin_k_data)
