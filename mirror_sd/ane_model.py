"""DFlash draft model running on Apple Neural Engine.

Uses mega_qkv kernel (input-pack approach) to fuse Q/K/V projections +
per-head norms + RoPE into a single ANE dispatch per layer, eliminating
3 Python round-trips per layer.

Key innovation (ANE rule #22): Instead of separate k_proj_ctx + k_proj_noise
+ concat (which fails at runtime due to concat→reshape→transpose pattern),
we pack context + normed into one tensor [1, HIDDEN, 1, w_kv] and do ONE
conv1x1 for K (and V). This is mathematically equivalent and avoids the
problematic output concat pattern.

The attention computation (GQA tile + SDPA + o_proj) uses separate kernels
because the mega_qkv + GQA tile exceeds the ANE's per-dispatch operation limit.

Data flow per layer (7 kernels → 4 kernels + 1 Python round-trip):
  hidden ──┬──→ mega_qkv ──→ (k_rope_4d, v_4d_t, q_rope_4d) ──→ gqa_tile ──→ attn_out ──→ (Python 4D→flat) ──→ o_proj_residual ──→ ffn_residual
  context ─┘
"""

import time
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


def _interleave_head_dims(data_flat, oc, n_heads, head_dim):
    half = head_dim // 2
    w = mx.array(data_flat, dtype=mx.float32).reshape(oc, -1)
    w_4d = w.reshape(n_heads, head_dim, -1)
    w_first = w_4d[:, :half, :]
    w_second = w_4d[:, half:, :]
    stacked = mx.stack([w_first, w_second], axis=2)
    w_il = stacked.reshape(n_heads, head_dim, -1)
    return w_il.reshape(oc, -1).flatten().tolist()


def _interleave_head_dims_mx(w: mx.array, n_heads: int, head_dim: int) -> mx.array:
    half = head_dim // 2
    w_4d = w.reshape(n_heads, head_dim, -1)
    w_first = w_4d[:, :half, :]
    w_second = w_4d[:, half:, :]
    stacked = mx.stack([w_first, w_second], axis=2)
    w_il = stacked.reshape(n_heads, head_dim, -1)
    return w_il.reshape(-1)


class ANEDraftModel:
    def __init__(self, seq_q: int, ctx_len: int):
        import mirror_sd_ane as ane

        self.ane = ane
        self.seq_q = seq_q
        self.max_ctx_len = ctx_len
        self.w_sq = align_width(seq_q)
        self.w_ctx = align_width(ctx_len)
        self.w_kv = self.w_ctx + self.w_sq

        self.config = DFlashConfig.qwen3_8b()
        self.config.block_size = seq_q
        self.block_size = seq_q
        self.mask_token_id = self.config.mask_token_id

        print(f"[ANE] Compiling kernels (seq_q={seq_q}, ctx_len={ctx_len}, "
              f"w_sq={self.w_sq}, w_ctx={self.w_ctx}, w_kv={self.w_kv})...")
        self.kernels = {k.name: k for k in ane.compile_dflash_kernels(seq_q, ctx_len, 30000.0)}
        print(f"[ANE] All {len(self.kernels)} kernels compiled: {list(self.kernels.keys())}")

        self._alloc_buffers()
        self.weights_loaded = False

    def _alloc_buffers(self):
        ane = self.ane
        w_sq, w_ctx, w_kv = self.w_sq, self.w_ctx, self.w_kv

        self.b_noise = ane.ANETensor(1, HIDDEN, 1, w_sq)
        self.b_target = ane.ANETensor(1, TARGET_HIDDEN, 1, w_ctx)
        self.b_context = ane.ANETensor(1, HIDDEN, 1, w_ctx)
        self.b_hidden = ane.ANETensor(1, HIDDEN, 1, w_sq)

        # mega_qkv outputs
        self.b_k_rope_4d = ane.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)
        self.b_v_4d_t = ane.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)
        self.b_q_rope_4d = ane.ANETensor(1, N_HEADS, w_sq, HEAD_DIM)

        # Attention path intermediates
        self.b_kv_tiled = ane.ANETensor(1, 2 * N_HEADS, w_kv, HEAD_DIM)
        self.b_attn_out = ane.ANETensor(1, N_HEADS, w_sq, HEAD_DIM)
        self.b_attn_flat = ane.ANETensor(1, N_HEADS * HEAD_DIM, 1, w_sq)
        self.b_attn_res = ane.ANETensor(1, HIDDEN, 1, w_sq)

        self.b_output = ane.ANETensor(1, HIDDEN, 1, w_sq)

        self.b_cos_q = ane.ANETensor(1, 1, w_sq, HEAD_DIM)
        self.b_sin_q = ane.ANETensor(1, 1, w_sq, HEAD_DIM)
        self.b_cos_k = ane.ANETensor(1, 1, w_kv, HEAD_DIM)
        self.b_sin_k = ane.ANETensor(1, 1, w_kv, HEAD_DIM)

        self.b_attn_mask = ane.ANETensor(1, 1, w_sq, w_kv)

    def _make_weight(self, data_flat, oc, ic, height=1):
        w_oc = align_width(oc)
        data_arr = mx.array(data_flat, dtype=mx.float32).reshape(oc, ic)
        data_t = data_arr.T
        padded = mx.zeros((ic, w_oc), dtype=mx.float32)
        padded[:, :oc] = data_t
        return self.ane.ANETensor.from_buffer(1, ic, height, oc, memoryview(padded))

    def _make_norm_weight_expanded(self, weight_list, channels, width):
        w = align_width(width)
        arr = mx.array(weight_list, dtype=mx.float32).reshape(channels, 1)
        arr = mx.broadcast_to(arr, (channels, w))
        return self.ane.ANETensor.from_buffer(1, channels, 1, w, memoryview(arr))

    def _make_4d_norm_weight(self, head_weight_list, n_heads, width):
        w = align_width(width)
        head_dim = len(head_weight_list)
        half = head_dim // 2
        il = []
        for k in range(half):
            il.append(head_weight_list[k])
            il.append(head_weight_list[k + half])
        il_arr = mx.array(il, dtype=mx.float32).reshape(head_dim, 1)
        arr = mx.broadcast_to(il_arr, (head_dim, n_heads * w))
        return self.ane.ANETensor.from_buffer(1, head_dim, 1, n_heads * w, memoryview(arr))

    def load_weights(self, draft_model: nn.Module, target_model: nn.Module = None):
        self._load_fc_weights(draft_model)
        for i in range(N_DFLASH_LAYERS):
            t0 = time.time()
            self._load_layer_weights(draft_model, i)
            print(f"[ANE]   Layer {i} loaded ({time.time()-t0:.1f}s)")
        self._load_final_norm_weights(draft_model)
        self.weights_loaded = True
        print("[ANE] Weights loaded")

    def _mlx_to_f32_list(self, arr: mx.array) -> list:
        return arr.astype(mx.float32).flatten().tolist()

    def _mlx_to_buffer(self, arr: mx.array):
        return memoryview(arr.astype(mx.float32))

    def _make_weight_buf(self, w_flat, oc, ic, height=1):
        w_oc = align_width(oc)
        if isinstance(w_flat, mx.array):
            data_arr = w_flat.reshape(oc, ic)
        else:
            data_arr = mx.array(w_flat, dtype=mx.float32).reshape(oc, ic)
        data_t = data_arr.T
        padded = mx.zeros((ic, w_oc), dtype=mx.float32)
        padded[:, :oc] = data_t
        return self.ane.ANETensor.from_buffer(1, ic, height, oc, memoryview(padded))

    def _load_fc_weights(self, model: nn.Module):
        fc_buf = self._mlx_to_buffer(model.fc.weight)
        self.w_fc = self._make_weight_buf(model.fc.weight.astype(mx.float32), HIDDEN, TARGET_HIDDEN)
        hidden_norm_w = self._mlx_to_f32_list(model.hidden_norm.weight)
        self.w_hidden_norm = self._make_norm_weight_expanded(hidden_norm_w, HIDDEN, self.w_ctx)

    def _load_layer_weights(self, model: nn.Module, layer_idx: int):
        layer = model.layers[layer_idx]
        p = f"l{layer_idx}_"

        in_norm_w = self._mlx_to_f32_list(layer.input_layernorm.weight)
        setattr(self, f"w_{p}in_norm", self._make_norm_weight_expanded(in_norm_w, HIDDEN, self.w_sq))

        q_proj_w = layer.self_attn.q_proj.weight.astype(mx.float32)
        q_proj_w_il = _interleave_head_dims_mx(q_proj_w, N_HEADS, HEAD_DIM)
        setattr(self, f"w_{p}q_proj", self._make_weight_buf(q_proj_w_il, N_HEADS * HEAD_DIM, HIDDEN))

        q_norm_w = self._mlx_to_f32_list(layer.self_attn.q_norm.weight)
        setattr(self, f"w_{p}q_norm_4d", self._make_4d_norm_weight(q_norm_w, N_HEADS, self.w_sq))

        k_proj_w = layer.self_attn.k_proj.weight.astype(mx.float32)
        k_proj_w_il = _interleave_head_dims_mx(k_proj_w, N_KV_HEADS, HEAD_DIM)
        setattr(self, f"w_{p}k_proj", self._make_weight_buf(k_proj_w_il, N_KV_HEADS * HEAD_DIM, HIDDEN))

        k_norm_w = self._mlx_to_f32_list(layer.self_attn.k_norm.weight)
        setattr(self, f"w_{p}k_norm_4d", self._make_4d_norm_weight(k_norm_w, N_KV_HEADS, self.w_kv))

        v_proj_w = layer.self_attn.v_proj.weight.astype(mx.float32)
        setattr(self, f"w_{p}v_proj", self._make_weight_buf(v_proj_w, N_KV_HEADS * HEAD_DIM, HIDDEN))

        o_proj_w = layer.self_attn.o_proj.weight.astype(mx.float32)
        setattr(self, f"w_{p}o_proj", self._make_weight_buf(o_proj_w, HIDDEN, N_HEADS * HEAD_DIM))

        post_norm_w = self._mlx_to_f32_list(layer.post_attention_layernorm.weight)
        setattr(self, f"w_{p}post_norm", self._make_norm_weight_expanded(post_norm_w, HIDDEN, self.w_sq))

        gate_w = layer.mlp.gate_proj.weight.astype(mx.float32)
        setattr(self, f"w_{p}gate", self._make_weight_buf(gate_w, INTERMEDIATE, HIDDEN))

        up_w = layer.mlp.up_proj.weight.astype(mx.float32)
        setattr(self, f"w_{p}up", self._make_weight_buf(up_w, INTERMEDIATE, HIDDEN))

        down_w = layer.mlp.down_proj.weight.astype(mx.float32)
        setattr(self, f"w_{p}down", self._make_weight_buf(down_w, HIDDEN, INTERMEDIATE))

    def _load_final_norm_weights(self, model: nn.Module):
        norm_w = self._mlx_to_f32_list(model.norm.weight)
        self.w_final_norm = self._make_norm_weight_expanded(norm_w, HIDDEN, self.w_sq)
    def forward(self, noise_embedding: mx.array, target_hidden: mx.array,
                rope_offset: int = 0, ctx_len: int = None) -> mx.array:
        if ctx_len is None:
            ctx_len = target_hidden.shape[1]
        if ctx_len > self.max_ctx_len:
            raise ValueError(
                f"ctx_len={ctx_len} exceeds compiled max_ctx_len={self.max_ctx_len}. "
                f"Re-initialize ANEDraftModel with a larger ctx_len."
            )
        k = self.kernels
        self._write_mlx_2d(self.b_hidden, noise_embedding)
        self._write_mlx_2d(self.b_target, target_hidden)

        self._compute_rope(rope_offset, ctx_len)
        self._compute_attn_mask(ctx_len)

        k['fc_norm'].run_uncached(
            [self.b_target, self.w_fc, self.w_hidden_norm],
            [self.b_context],
        )

        for i in range(N_DFLASH_LAYERS):
            self._run_layer(k, i)

        k['final_norm'].run_uncached(
            [self.b_hidden, self.w_final_norm],
            [self.b_output],
        )

        return self._read_mlx_2d(self.b_output, self.seq_q, HIDDEN)

    def __call__(self, noise_embedding: mx.array, target_hidden: mx.array,
                 mask=None, cache=None, **kwargs) -> mx.array:
        rope_offset = 0
        if cache is not None and len(cache) > 0 and cache[0].offset > 0:
            rope_offset = cache[0].offset
        return self.forward(noise_embedding, target_hidden, rope_offset=rope_offset,
                           ctx_len=target_hidden.shape[1])

    def make_cache(self):
        from .dflash import DFlashKVCache
        return [DFlashKVCache() for _ in range(N_DFLASH_LAYERS)]

    def _write_mlx_2d(self, buf, arr: mx.array):
        seq_len = arr.shape[1]
        channels = arr.shape[2]
        w = buf.shape[3]
        f32 = arr.astype(mx.float32).transpose(0, 2, 1)
        padded = mx.zeros((1, channels, w), dtype=mx.float32)
        padded[:, :, :seq_len] = f32
        mx.eval(padded)
        buf.write_buffer(memoryview(padded))

    def _read_mlx_2d(self, buf, seq_len: int, channels: int) -> mx.array:
        w = buf.shape[3]
        data = buf.read_f32()
        arr = mx.array(data, dtype=mx.float32).reshape(1, channels, w)[:, :, :seq_len]
        return arr.transpose(0, 2, 1)

    def _clip_buffer(self, buf, max_val: float):
        data = buf.read_f32()
        arr = mx.array(data, dtype=mx.float32)
        arr = mx.clip(arr, -max_val, max_val)
        buf.write_buffer(memoryview(arr.flatten().astype(mx.float32)))

    def _softcap_buffer(self, buf, cap: float):
        data = buf.read_f32()
        arr = mx.array(data, dtype=mx.float32)
        arr = cap * mx.tanh(arr / cap)
        buf.write_buffer(memoryview(arr.flatten().astype(mx.float32)))

    def _run_layer(self, k, layer_idx: int):
        p = f"l{layer_idx}_"

        # --- mega_qkv: in_norm + Q/K/V projections + per-head norms + RoPE ---
        # Input-pack approach: concat(context, normed) → one conv1x1 for K/V
        # 3 outputs: k_rope_4d, v_4d_t, q_rope_4d
        k['mega_qkv'].run_uncached(
            [self.b_hidden, getattr(self, f"w_{p}in_norm"),
             self.b_context,
             getattr(self, f"w_{p}k_proj"),
             getattr(self, f"w_{p}k_norm_4d"),
             self.b_cos_k, self.b_sin_k,
             getattr(self, f"w_{p}v_proj"),
             getattr(self, f"w_{p}q_proj"),
             getattr(self, f"w_{p}q_norm_4d"),
             self.b_cos_q, self.b_sin_q],
            [self.b_k_rope_4d, self.b_v_4d_t, self.b_q_rope_4d],
        )

        # --- GQA tile + attention + o_proj ---
        k['gqa_tile'].run_uncached(
            [self.b_k_rope_4d, self.b_v_4d_t],
            [self.b_kv_tiled],
        )

        k['attn_out'].run_uncached(
            [self.b_q_rope_4d, self.b_kv_tiled, self.b_attn_mask],
            [self.b_attn_flat],
        )

        k['o_proj_residual'].run_uncached(
            [self.b_attn_flat, getattr(self, f"w_{p}o_proj"), self.b_hidden],
            [self.b_attn_res],
        )

        k['ffn_residual'].run_uncached(
            [self.b_attn_res, getattr(self, f"w_{p}post_norm"),
             getattr(self, f"w_{p}gate"), getattr(self, f"w_{p}up"), getattr(self, f"w_{p}down")],
            [self.b_hidden],
        )

    def _compute_rope(self, rope_offset: int, ctx_len: int):
        rope_theta = 1000000.0
        half = HEAD_DIM // 2

        q_positions = mx.array([rope_offset + ctx_len + p for p in range(self.w_sq)], dtype=mx.float32)
        q_freqs = mx.array([1.0 / (rope_theta ** (2.0 * d / HEAD_DIM)) for d in range(half)], dtype=mx.float32)
        q_angles = q_positions[:, None] * q_freqs[None, :]
        q_cos = mx.cos(q_angles)
        q_sin = mx.sin(q_angles)
        q_cos = mx.repeat(q_cos, 2, axis=1)
        q_sin = mx.repeat(q_sin, 2, axis=1)
        self.b_cos_q.write_buffer(memoryview(q_cos.flatten().astype(mx.float32)))
        self.b_sin_q.write_buffer(memoryview(q_sin.flatten().astype(mx.float32)))

        k_ctx_pos = mx.array([rope_offset + p for p in range(ctx_len)], dtype=mx.float32)
        k_noise_pos = mx.array([rope_offset + ctx_len + p for p in range(self.w_sq)], dtype=mx.float32)
        k_positions = mx.concatenate([k_ctx_pos, k_noise_pos])
        k_angles = k_positions[:, None] * q_freqs[None, :]
        k_cos = mx.cos(k_angles)
        k_sin = mx.sin(k_angles)
        k_cos = mx.repeat(k_cos, 2, axis=1)
        k_sin = mx.repeat(k_sin, 2, axis=1)
        self.b_cos_k.write_buffer(memoryview(k_cos.flatten().astype(mx.float32)))
        self.b_sin_k.write_buffer(memoryview(k_sin.flatten().astype(mx.float32)))

    def _compute_attn_mask(self, ctx_len: int):
        mask = mx.full((1, 1, self.w_sq, self.w_kv), -1e4, dtype=mx.float32)
        mask[:, :, :, :ctx_len] = 0.0
        mask[:, :, :, self.w_ctx:self.w_ctx + self.seq_q] = 0.0
        self.b_attn_mask.write_buffer(memoryview(mask.flatten().astype(mx.float32)))

    def _zero_pad_context(self, ctx_len: int):
        w = self.b_context.shape[3]
        data = self.b_context.read_f32()
        arr = mx.array(data, dtype=mx.float32).reshape(1, HIDDEN, 1, w)
        arr[:, :, :, ctx_len:] = 0.0
        self.b_context.write_buffer(memoryview(arr.flatten().astype(mx.float32)))
