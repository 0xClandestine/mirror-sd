"""DFlash draft model running on Apple Neural Engine.

Chains 17 ANE kernels per layer with IOSurface buffers for intermediate
results and real weight loading.

K/V projections are split into ctx/noise sub-kernels (ANE can't handle
concat+conv1x1 in one dispatch). Python concatenates the outputs.

K and V concat use separate kernel instances because run_cached requires
the same IOSurface objects on every call.

RoPE uses interleaved rotation (ANE pairs consecutive dims 2k,2k+1).
To match Qwen3's half-rotation (pairs d, d+64), we interleave the
q_proj/k_proj output dimensions: [d0..d63,d64..d127] -> [d0,d64,d1,d65,...]
This makes the interleaved ANE RoPE equivalent to the standard half-rotation.
Q and K are both interleaved, so Q@K^T is invariant; V stays in original
format, so attn_out is also original format and o_proj needs no changes.

CRITICAL: q_norm and k_norm must be per-head rmsnorm (reduce over HEAD_DIM=128
per head), NOT global rmsnorm (reduce over all N_HEADS*HEAD_DIM channels).
ANE reduce_mean only works on channel axis, so we reshape to [1, HEAD_DIM, 1, N_HEADS*w_sq]
where each spatial position is one (head, seq_pos) pair, then rmsnorm reduces
over channels (HEAD_DIM values) per position.

Data flow per layer:
  hidden ──┬──→ q_proj ──→ (Python flat→4d_norm) ──→ q_norm_4d ──→ (Python 4d_norm→4d) ──→ rope_q ──┐
           │                                                                                ├──→ attn_out ──→ (Python 4D→flat) ──→ o_proj_residual ──→ attn_res ──→ ffn_residual ──→ hidden_next
  context ─┼──→ k_proj_ctx ──┐                                                               │
           │                 ├── k_concat → (Python flat→4d_norm) → k_norm_4d → (Python 4d_norm→4d) → rope_k ─┤
           └──→ k_proj_noise ┘                                                               ├──→ gqa_tile ──┐
                 v_proj_ctx ──┐                                                                              │
                 v_proj_noise ┘── v_concat ──→ (Python flat→4d) ───────────────────────────────────────────┘
"""

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
    """Reorder OC rows of [OC, IC] weight so that within each head,
    dimensions are interleaved: [d0..d63, d64..d127] -> [d0,d64,d1,d65,...,d63,d127].
    This makes the ANE interleaved RoPE equivalent to the standard half-rotation."""
    half = head_dim // 2
    w = mx.array(data_flat, dtype=mx.float32).reshape(oc, -1)
    w_4d = w.reshape(n_heads, head_dim, -1)  # [NH, HD, IC]
    w_first = w_4d[:, :half, :]   # [NH, half, IC]
    w_second = w_4d[:, half:, :]  # [NH, half, IC]
    # Interleave: stack along dim 1 → [NH, half, 2, IC] → reshape [NH, HD, IC]
    stacked = mx.stack([w_first, w_second], axis=2)  # [NH, half, 2, IC]
    w_il = stacked.reshape(n_heads, head_dim, -1)  # [NH, HD, IC]
    return w_il.reshape(oc, -1).flatten().tolist()


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

        print(f"[ANE] Compiling 17 kernels (seq_q={seq_q}, ctx_len={ctx_len}, "
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
        self.b_q_norm_4d = ane.ANETensor(1, HEAD_DIM, 1, N_HEADS * w_sq)

        self.b_q_4d = ane.ANETensor(1, N_HEADS, w_sq, HEAD_DIM)
        self.b_q_rope_4d = ane.ANETensor(1, N_HEADS, w_sq, HEAD_DIM)
        self.b_k_4d = ane.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)
        self.b_k_rope_4d = ane.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)
        self.b_v_4d = ane.ANETensor(1, N_KV_HEADS, w_kv, HEAD_DIM)

        self.b_k_ctx = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_ctx)
        self.b_k_noise = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_sq)
        self.b_k_out = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_kv)
        self.b_k_norm_4d = ane.ANETensor(1, HEAD_DIM, 1, N_KV_HEADS * w_kv)

        self.b_v_ctx = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_ctx)
        self.b_v_noise = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_sq)
        self.b_v_out = ane.ANETensor(1, N_KV_HEADS * HEAD_DIM, 1, w_kv)

        self.b_kv_tiled = ane.ANETensor(1, 2 * N_HEADS, w_kv, HEAD_DIM)

        self.b_attn_out = ane.ANETensor(1, N_HEADS, w_sq, HEAD_DIM)
        self.b_attn_flat = ane.ANETensor(1, N_HEADS * HEAD_DIM, 1, w_sq)
        self.b_attn_res = ane.ANETensor(1, HIDDEN, 1, w_sq)
        self.b_output = ane.ANETensor(1, HIDDEN, 1, w_sq)

        self.b_cos_q = ane.ANETensor(1, 1, w_sq, HEAD_DIM)
        self.b_sin_q = ane.ANETensor(1, 1, w_sq, HEAD_DIM)
        self.b_cos_k = ane.ANETensor(1, 1, w_kv, HEAD_DIM)
        self.b_sin_k = ane.ANETensor(1, 1, w_kv, HEAD_DIM)

    def _make_weight(self, data_flat, oc, ic, height=1):
        # ANE concat-slice pattern requires weight shape [1, IC, 1, OC] with transposed data.
        # MLX weight is row-major [OC, IC]; we store as W.T (column-major) so element
        # [0, ic, 0, oc] = W[oc][ic] at offset ic*OC + oc.
        w_oc = align_width(oc)
        data_arr = mx.array(data_flat, dtype=mx.float32).reshape(oc, ic)
        data_t = data_arr.T  # [IC, OC]
        padded = mx.zeros((ic, w_oc), dtype=mx.float32)
        padded[:, :oc] = data_t
        return self.ane.ANETensor.from_f32(1, ic, height, oc, padded.flatten().tolist())

    def _make_norm_weight_expanded(self, weight_list, channels, width):
        w = align_width(width)
        arr = mx.array(weight_list, dtype=mx.float32).reshape(channels, 1)
        arr = mx.broadcast_to(arr, (channels, w))
        return self.ane.ANETensor.from_f32(1, channels, 1, w, arr.flatten().tolist())

    def _make_per_head_norm_weight(self, head_weight_list, n_heads, width, interleave=False):
        w = align_width(width)
        head_dim = len(head_weight_list)
        half = head_dim // 2
        if interleave:
            il = []
            for k in range(half):
                il.append(head_weight_list[k])
                il.append(head_weight_list[k + half])
            per_head = mx.array(il, dtype=mx.float32).reshape(1, head_dim)
        else:
            per_head = mx.array(head_weight_list, dtype=mx.float32).reshape(1, head_dim)
        all_heads = mx.repeat(per_head, n_heads, axis=0)  # [n_heads*head_dim]
        all_heads = all_heads.reshape(n_heads * head_dim, 1)
        arr = mx.broadcast_to(all_heads, (n_heads * head_dim, w))
        return self.ane.ANETensor.from_f32(1, n_heads * head_dim, 1, w, arr.flatten().tolist())

    def _make_4d_norm_weight(self, head_weight_list, n_heads, width):
        """Build weight for q_norm_4d/k_norm_4d kernels.

        Weight shape: [1, HEAD_DIM, 1, N_HEADS * w_sq] with interleaved
        norm weight values repeated for each (head, position) spatial location.
        IOSurface stores data channels-first: iterate (d, h, pos) not (h, pos, d).
        """
        w = align_width(width)
        head_dim = len(head_weight_list)
        half = head_dim // 2
        il = []
        for k in range(half):
            il.append(head_weight_list[k])
            il.append(head_weight_list[k + half])
        il_arr = mx.array(il, dtype=mx.float32).reshape(head_dim, 1)
        # Broadcast: [head_dim, 1] -> [head_dim, n_heads * w]
        arr = mx.broadcast_to(il_arr, (head_dim, n_heads * w))
        return self.ane.ANETensor.from_f32(1, head_dim, 1, n_heads * w, arr.flatten().tolist())

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
        q_proj_w_il = _interleave_head_dims(q_proj_w, N_HEADS * HEAD_DIM, N_HEADS, HEAD_DIM)
        setattr(self, f"w_{p}q_proj", self._make_weight(q_proj_w_il, N_HEADS * HEAD_DIM, HIDDEN))

        q_norm_w = self._mlx_to_f32_list(layer.self_attn.q_norm.weight)
        setattr(self, f"w_{p}q_norm_4d", self._make_4d_norm_weight(q_norm_w, N_HEADS, self.w_sq))

        k_proj_w = self._mlx_to_f32_list(layer.self_attn.k_proj.weight)
        k_proj_w_il = _interleave_head_dims(k_proj_w, N_KV_HEADS * HEAD_DIM, N_KV_HEADS, HEAD_DIM)
        setattr(self, f"w_{p}k_proj", self._make_weight(k_proj_w_il, N_KV_HEADS * HEAD_DIM, HIDDEN))

        k_norm_w = self._mlx_to_f32_list(layer.self_attn.k_norm.weight)
        setattr(self, f"w_{p}k_norm_4d", self._make_4d_norm_weight(k_norm_w, N_KV_HEADS, self.w_kv))

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
        self._write_mlx_2d(self.b_noise, noise_embedding)
        self._write_mlx_2d(self.b_target, target_hidden)

        noise_data = self.b_noise.read_f32()
        self.b_hidden.write_f32(noise_data)

        self._compute_rope(rope_offset, ctx_len)

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
        """Drop-in replacement for DFlashDraftModel.__call__.

        Ignores mask and cache (ANE runs full non-causal attention each step).
        rope_offset is derived from cache offset if provided.
        ctx_len is derived from target_hidden.shape[1].
        """
        rope_offset = 0
        if cache is not None and len(cache) > 0 and cache[0].offset > 0:
            rope_offset = cache[0].offset
        return self.forward(noise_embedding, target_hidden, rope_offset=rope_offset,
                           ctx_len=target_hidden.shape[1])

    def make_cache(self):
        """Return list of DFlashKVCache instances (unused by ANE but needed by spec_generate)."""
        from .dflash import DFlashKVCache
        return [DFlashKVCache() for _ in range(N_DFLASH_LAYERS)]

    def _write_mlx_2d(self, buf, arr: mx.array):
        seq_len = arr.shape[1]
        channels = arr.shape[2]
        w = buf.shape[3]
        f32 = arr.astype(mx.float32).transpose(0, 2, 1)  # [1, channels, seq]
        padded = mx.zeros((1, channels, w), dtype=mx.float32)
        padded[:, :, :seq_len] = f32
        buf.write_f32(padded.flatten().tolist())

    def _read_mlx_2d(self, buf, seq_len: int, channels: int) -> mx.array:
        w = buf.shape[3]
        data = buf.read_f32()
        arr = mx.array(data, dtype=mx.float32).reshape(1, channels, w)[:, :, :seq_len]
        return arr.transpose(0, 2, 1)  # [1, seq, channels]

    def _flatten_attn_4d(self):
        w_sq = self.w_sq
        data = self.b_attn_out.read_f32()
        attn_4d = mx.array(data, dtype=mx.float32).reshape(1, N_HEADS, w_sq, HEAD_DIM)
        flat = attn_4d.transpose(0, 2, 1, 3).reshape(1, w_sq, N_HEADS * HEAD_DIM)
        flat_t = flat.transpose(0, 2, 1)
        padded = mx.zeros((1, N_HEADS * HEAD_DIM, w_sq), dtype=mx.float32)
        padded[:, :, :self.seq_q] = flat_t[:, :, :self.seq_q]
        self.b_attn_flat.write_f32(padded.flatten().tolist())

    def _flat_to_4d_heads(self, flat_buf, n_heads, seq_w, out_4d_buf):
        data = flat_buf.read_f32()
        channels = n_heads * HEAD_DIM
        arr = mx.array(data, dtype=mx.float32).reshape(1, channels, 1, seq_w)
        arr2 = arr.reshape(1, n_heads, HEAD_DIM, seq_w)
        arr3 = arr2.transpose(0, 1, 3, 2)
        out_4d_buf.write_f32(arr3.flatten().tolist())

    def _flat_to_4d_norm(self, flat_buf, n_heads, seq_w, out_norm_buf):
        """Rearrange flat [1, N_HEADS*HEAD_DIM, 1, w] to norm format [1, HEAD_DIM, 1, N_HEADS*w].

        For per-head rmsnorm: HEAD_DIM must be the channel axis.
        Rearrange: [NH, HD, w] → [HD, NH, w] → flatten to [HD, NH*w].
        Data is interleaved (from q_proj/k_proj), so channel ordering within
        each head's HEAD_DIM already matches the interleaved norm weight.
        """
        data = flat_buf.read_f32()
        channels = n_heads * HEAD_DIM
        w = flat_buf.shape[3]
        arr = mx.array(data, dtype=mx.float32).reshape(1, n_heads, HEAD_DIM, w)
        arr_t = arr.transpose(0, 2, 1, 3)  # [1, HEAD_DIM, N_HEADS, w]
        arr_flat = arr_t.reshape(1, HEAD_DIM, 1, n_heads * w)
        out_norm_buf.write_f32(arr_flat.flatten().tolist())

    def _4d_norm_to_4d_heads(self, norm_buf, n_heads, seq_w, out_4d_buf):
        """Convert norm format [1, HEAD_DIM, 1, N_HEADS*w] back to 4D [1, N_HEADS, w, HEAD_DIM].

        Inverse of _flat_to_4d_norm: [HD, NH*w] → [HD, NH, w] → [NH, w, HD].
        """
        data = norm_buf.read_f32()
        w = seq_w
        arr = mx.array(data, dtype=mx.float32).reshape(1, HEAD_DIM, n_heads, w)
        arr_t = arr.transpose(0, 2, 3, 1)  # [1, N_HEADS, w, HEAD_DIM]
        out_4d_buf.write_f32(arr_t.flatten().tolist())

    def _4d_to_flat_heads(self, buf_4d, n_heads, seq_w, out_flat_buf):
        data = buf_4d.read_f32()
        arr = mx.array(data, dtype=mx.float32).reshape(1, n_heads, seq_w, HEAD_DIM)
        arr2 = arr.transpose(0, 1, 3, 2)
        arr3 = arr2.reshape(1, n_heads * HEAD_DIM, 1, seq_w)
        out_flat_buf.write_f32(arr3.flatten().tolist())

    def _run_layer(self, k, layer_idx: int):
        p = f"l{layer_idx}_"

        # Q: in_norm + q_proj (no q_norm — that's separate now)
        k['q_proj'].run_uncached(
            [self.b_hidden, getattr(self, f"w_{p}in_norm"),
             getattr(self, f"w_{p}q_proj")],
            [self.b_q_out],
        )

        k['k_proj_ctx'].run_uncached(
            [self.b_context, getattr(self, f"w_{p}k_proj")],
            [self.b_k_ctx],
        )

        k['k_proj_noise'].run_uncached(
            [self.b_hidden, getattr(self, f"w_{p}in_norm"), getattr(self, f"w_{p}k_proj")],
            [self.b_k_noise],
        )

        k['k_concat'].run_uncached([self.b_k_ctx, self.b_k_noise], [self.b_k_out])

        k['v_proj_ctx'].run_uncached(
            [self.b_context, getattr(self, f"w_{p}v_proj")],
            [self.b_v_ctx],
        )

        k['v_proj_noise'].run_uncached(
            [self.b_hidden, getattr(self, f"w_{p}in_norm"), getattr(self, f"w_{p}v_proj")],
            [self.b_v_noise],
        )

        k['v_concat'].run_uncached([self.b_v_ctx, self.b_v_noise], [self.b_v_out])

        # Q: flat → norm_4d format → per-head rmsnorm → 4D heads → rope_q
        self._flat_to_4d_norm(self.b_q_out, N_HEADS, self.w_sq, self.b_q_norm_4d)
        k['q_norm_4d'].run_uncached(
            [self.b_q_norm_4d, getattr(self, f"w_{p}q_norm_4d")],
            [self.b_q_norm_4d],
        )
        self._4d_norm_to_4d_heads(self.b_q_norm_4d, N_HEADS, self.w_sq, self.b_q_4d)
        k['rope_q'].run_uncached(
            [self.b_q_4d, self.b_cos_q, self.b_sin_q],
            [self.b_q_rope_4d],
        )

        # K: flat → norm_4d format → per-head rmsnorm → 4D heads → rope_k
        self._flat_to_4d_norm(self.b_k_out, N_KV_HEADS, self.w_kv, self.b_k_norm_4d)
        k['k_norm_4d'].run_uncached(
            [self.b_k_norm_4d, getattr(self, f"w_{p}k_norm_4d")],
            [self.b_k_norm_4d],
        )
        self._4d_norm_to_4d_heads(self.b_k_norm_4d, N_KV_HEADS, self.w_kv, self.b_k_4d)
        k['rope_k'].run_uncached(
            [self.b_k_4d, self.b_cos_k, self.b_sin_k],
            [self.b_k_rope_4d],
        )

        # V: flat → 4D (no norm for V)
        self._flat_to_4d_heads(self.b_v_out, N_KV_HEADS, self.w_kv, self.b_v_4d)

        # GQA tile + attention
        k['gqa_tile'].run_uncached(
            [self.b_k_rope_4d, self.b_v_4d],
            [self.b_kv_tiled],
        )

        k['attn_out'].run_uncached(
            [self.b_q_rope_4d, self.b_kv_tiled],
            [self.b_attn_out],
        )

        self._flatten_attn_4d()

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
        self.b_cos_q.write_f32(q_cos.flatten().tolist())
        self.b_sin_q.write_f32(q_sin.flatten().tolist())

        k_ctx_pos = mx.array([rope_offset + p for p in range(ctx_len)], dtype=mx.float32)
        k_noise_pos = mx.array([rope_offset + ctx_len + p for p in range(self.w_sq)], dtype=mx.float32)
        k_positions = mx.concatenate([k_ctx_pos, k_noise_pos])
        k_angles = k_positions[:, None] * q_freqs[None, :]
        k_cos = mx.cos(k_angles)
        k_sin = mx.sin(k_angles)
        k_cos = mx.repeat(k_cos, 2, axis=1)
        k_sin = mx.repeat(k_sin, 2, axis=1)
        self.b_cos_k.write_f32(k_cos.flatten().tolist())
        self.b_sin_k.write_f32(k_sin.flatten().tolist())
