"""DFlash draft model running on Apple Neural Engine.

Uses mega_qkv kernel (input-pack approach) to fuse Q/K/V projections +
per-head norms + RoPE into a single ANE dispatch per layer.

Data flow:
  target_hidden → fc_norm → context (on ANE, not GPU)
  hidden ──┬──→ mega_qkv ──→ (k_rope_4d, v_4d_t, q_rope_4d) ──→ gqa_tile ──→ attn_out ──→ o_proj_residual ──→ ffn_residual
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


def _interleave_head_dims_mx(w: mx.array, n_heads: int, head_dim: int) -> mx.array:
    half = head_dim // 2
    w_4d = w.reshape(n_heads, head_dim, -1)
    w_first = w_4d[:, :half, :]
    w_second = w_4d[:, half:, :]
    stacked = mx.stack([w_first, w_second], axis=2)
    w_il = stacked.reshape(n_heads, head_dim, -1)
    return w_il.reshape(-1)


class ANEDraftModel:
    def __init__(self, seq_q: int, ctx_len: int, config=None, vocab_size: Optional[int] = None):
        import mirror_sd_ane as ane

        if config is not None:
            self.hidden = config.hidden_size
            self.head_dim = config.head_dim
            self.n_heads = config.num_attention_heads
            self.n_kv_heads = config.num_key_value_heads
            self.intermediate = config.intermediate_size
            self.target_hidden = 5 * config.hidden_size
            self.n_layers = config.num_hidden_layers
            self.config = config
        else:
            self.hidden = HIDDEN
            self.head_dim = HEAD_DIM
            self.n_heads = N_HEADS
            self.n_kv_heads = N_KV_HEADS
            self.intermediate = INTERMEDIATE
            self.target_hidden = TARGET_HIDDEN
            self.n_layers = N_DFLASH_LAYERS
            self.config = DFlashConfig.qwen3_8b()

        self.ane = ane
        self.seq_q = seq_q
        self.max_ctx_len = ctx_len
        self.w_sq = align_width(seq_q)
        self.w_ctx = align_width(ctx_len)
        self.w_kv = self.w_ctx + self.w_sq

        self.config.block_size = seq_q
        self.block_size = seq_q
        self.mask_token_id = self.config.mask_token_id

        is_27b = self.hidden == 5120
        compile_fn = ane.compile_dflash_kernels_27b if is_27b else ane.compile_dflash_kernels
        self.softcap = float(getattr(self.config, 'attn_logit_softcapping', 0) or 0)
        self.is_27b = is_27b

        # Set True after load_weights_q8() — switches run_kernels to use per-layer q8 kernels.
        self.use_q8 = False
        self.layer_kernels = None

        print(f"[ANE] Compiling kernels (seq_q={seq_q}, ctx_len={ctx_len}, "
              f"w_sq={self.w_sq}, w_ctx={self.w_ctx}, w_kv={self.w_kv}, "
              f"model={'27B' if is_27b else '8B'}, softcap={self.softcap})...")
        self.kernels = {k.name: k for k in compile_fn(seq_q, ctx_len, self.softcap)}
        print(f"[ANE] All {len(self.kernels)} kernels compiled: {list(self.kernels.keys())}")

        self.vocab_size = vocab_size
        if vocab_size is not None:
            print(f"[ANE] Compiling lm_head kernel (vocab={vocab_size})...")
            t0 = time.time()
            lm_head_k = ane.compile_lm_head_kernel(seq_q, vocab_size, is_27b)
            self.kernels['final_norm_lm_head'] = lm_head_k
            print(f"[ANE] lm_head kernel compiled ({time.time()-t0:.1f}s)")

        self._alloc_buffers()
        self._fc_weight = None
        self._hidden_norm_weight = None
        self.weights_loaded = False

        self._rope_freqs = None
        self._rope_cache_key = None
        self._attn_mask_ctx_len = None

    def _alloc_buffers(self):
        ane = self.ane
        w_sq, w_ctx, w_kv = self.w_sq, self.w_ctx, self.w_kv
        H = self.hidden
        HD = self.head_dim
        NH = self.n_heads
        NKV = self.n_kv_heads
        TH = self.target_hidden

        self.b_noise = ane.ANETensor(1, H, 1, w_sq)
        self.b_target = ane.ANETensor(1, TH, 1, w_ctx)
        self.b_context = ane.ANETensor(1, H, 1, w_ctx)
        self.b_hidden = ane.ANETensor(1, H, 1, w_sq)

        self.b_k_rope_4d = ane.ANETensor(1, NKV, w_kv, HD)
        self.b_v_4d_t = ane.ANETensor(1, NKV, w_kv, HD)
        self.b_q_rope_4d = ane.ANETensor(1, NH, w_sq, HD)

        self.b_kv_tiled = ane.ANETensor(1, 2 * NH, w_kv, HD)
        self.b_attn_flat = ane.ANETensor(1, NH * HD, 1, w_sq)
        self.b_attn_res = ane.ANETensor(1, H, 1, w_sq)

        self.b_output = ane.ANETensor(1, H, 1, w_sq)

        self.b_cos_q = ane.ANETensor(1, 1, w_sq, HD)
        self.b_sin_q = ane.ANETensor(1, 1, w_sq, HD)
        self.b_cos_k = ane.ANETensor(1, 1, w_kv, HD)
        self.b_sin_k = ane.ANETensor(1, 1, w_kv, HD)

        self.b_attn_mask = ane.ANETensor(1, 1, w_sq, w_kv)

        # Allocated only when vocab_size is set
        self.b_logits = (
            ane.ANETensor(1, self.vocab_size, 1, w_sq) if self.vocab_size is not None else None
        )

        self._padded_hidden = mx.zeros((1, H, w_sq), dtype=mx.float32)
        self._padded_target = mx.zeros((1, TH, w_ctx), dtype=mx.float32)
        self._padded_context = mx.zeros((1, H, w_ctx), dtype=mx.float32)

    def _make_norm_weight_expanded(self, weight: mx.array, width: int):
        w = align_width(width)
        channels = weight.shape[0]
        arr = weight.astype(mx.float32).reshape(channels, 1)
        arr = mx.broadcast_to(arr, (channels, w))
        return self.ane.ANETensor.from_buffer(1, channels, 1, w, memoryview(arr))

    def _make_per_head_norm_weight(self, head_weight: mx.array, n_heads: int, width: int):
        w = align_width(width)
        head_dim = head_weight.shape[0]
        arr = mx.broadcast_to(head_weight.astype(mx.float32).reshape(1, 1, head_dim, 1),
                              (1, n_heads, head_dim, w))
        return self.ane.ANETensor.from_buffer(1, n_heads, head_dim, w, memoryview(arr))

    def load_weights(self, draft_model: nn.Module, target_model: nn.Module = None):
        self._load_fc_weights(draft_model)
        self._fc_weight = draft_model.fc.weight.astype(mx.float32)
        self._hidden_norm_weight = draft_model.hidden_norm.weight.astype(mx.float32)
        for i in range(self.n_layers):
            t0 = time.time()
            self._load_layer_weights(draft_model, i)
            print(f"[ANE]   Layer {i} loaded ({time.time()-t0:.1f}s)")
        self._load_final_norm_weights(draft_model)
        if self.vocab_size is not None and target_model is not None:
            self._load_lm_head_weights(target_model)
        self.weights_loaded = True
        print("[ANE] Weights loaded")

    def _make_weight_buf(self, w_flat, oc, ic, height=1):
        w_oc = align_width(oc)
        if isinstance(w_flat, mx.array):
            data_arr = w_flat.reshape(oc, ic)
        else:
            data_arr = mx.array(w_flat, dtype=mx.float16).reshape(oc, ic)
        data_t = data_arr.T.astype(mx.float16)
        padded = mx.zeros((ic, w_oc), dtype=mx.float16)
        padded[:, :oc] = data_t
        mx.eval(padded)
        t = self.ane.ANETensor(1, ic, height, oc)
        t.write_buffer_f16(memoryview(padded))
        return t

    def _load_fc_weights(self, model: nn.Module):
        self.w_fc = self._make_weight_buf(model.fc.weight, self.hidden, self.target_hidden)
        self.w_hidden_norm = self._make_norm_weight_expanded(
            model.hidden_norm.weight, self.w_ctx)

    def _load_layer_weights(self, model: nn.Module, layer_idx: int):
        layer = model.layers[layer_idx]
        p = f"l{layer_idx}_"
        sq_w = layer.self_attn
        H = self.hidden
        NH = self.n_heads
        NKV = self.n_kv_heads
        HD = self.head_dim

        setattr(self, f"w_{p}in_norm",
                self._make_norm_weight_expanded(layer.input_layernorm.weight, self.w_sq))

        setattr(self, f"w_{p}q_proj",
                self._make_weight_buf(sq_w.q_proj.weight.astype(mx.float16), NH * HD, H))

        setattr(self, f"w_{p}q_norm_4d",
                self._make_per_head_norm_weight(sq_w.q_norm.weight, NH, self.w_sq))

        setattr(self, f"w_{p}k_proj",
                self._make_weight_buf(sq_w.k_proj.weight.astype(mx.float16), NKV * HD, H))

        setattr(self, f"w_{p}k_norm_4d",
                self._make_per_head_norm_weight(sq_w.k_norm.weight, NKV, self.w_kv))

        setattr(self, f"w_{p}v_proj",
                self._make_weight_buf(sq_w.v_proj.weight, NKV * HD, H))

        setattr(self, f"w_{p}o_proj",
                self._make_weight_buf(sq_w.o_proj.weight, H, NH * HD))

        setattr(self, f"w_{p}post_norm",
                self._make_norm_weight_expanded(layer.post_attention_layernorm.weight, self.w_sq))

        setattr(self, f"w_{p}gate",
                self._make_weight_buf(layer.mlp.gate_proj.weight, self.intermediate, H))
        setattr(self, f"w_{p}up",
                self._make_weight_buf(layer.mlp.up_proj.weight, self.intermediate, H))
        setattr(self, f"w_{p}down",
                self._make_weight_buf(layer.mlp.down_proj.weight, H, self.intermediate))

    def _load_final_norm_weights(self, model: nn.Module):
        self.w_final_norm = self._make_norm_weight_expanded(model.norm.weight, self.w_sq)

    def _load_lm_head_weights(self, target_model: nn.Module):
        # lm_head weight: [vocab_size, hidden] — may be tied to embed_tokens
        inner = target_model
        for attr in ('language_model', 'model'):
            if hasattr(inner, attr):
                inner = getattr(inner, attr)
                break
        lm_head_w = getattr(inner, 'lm_head', None) or getattr(target_model, 'lm_head', None)
        if lm_head_w is None:
            raise AttributeError(
                "Cannot find lm_head on target_model. "
                "Pass target_model with a .lm_head attribute."
            )
        w = lm_head_w.weight if hasattr(lm_head_w, 'weight') else lm_head_w
        print(f"[ANE] Loading lm_head weight {w.shape}...")
        t0 = time.time()
        self.w_lm_head = self._make_weight_buf(w.astype(mx.float16), self.vocab_size, self.hidden)
        print(f"[ANE] lm_head weight loaded ({time.time()-t0:.1f}s)")

    # ------------------------------------------------------------------ #
    # W8A16 / Int8 quantized path                                        #
    # ------------------------------------------------------------------ #

    def _quantize_weight_q8(self, w: mx.array):
        """Quantize [oc, ic] weight to (int8_bytes, fp16_scale_bytes, oc, ic).

        Symmetric per-output-channel int8: scale = max(|w|, axis=1) / 127.
        Weights must already be in [oc, ic] row-major order.
        """
        import numpy as np
        w_np = np.array(w.astype(mx.float32), dtype=np.float32)  # [oc, ic]
        oc, ic = w_np.shape
        max_vals = np.abs(w_np).max(axis=1)  # [oc]
        max_vals = np.where(max_vals == 0, 1.0, max_vals)
        scales = max_vals / 127.0  # [oc]
        w_int8 = np.clip(np.round(w_np / scales[:, None]), -127, 127).astype(np.int8)
        scales_f16 = scales.astype(np.float16)
        return w_int8.tobytes(), scales_f16.tobytes(), oc, ic

    def _load_layer_norm_weights(self, model: nn.Module, layer_idx: int):
        """Load per-head norm + layer norm weights only (no projection weights)."""
        layer = model.layers[layer_idx]
        p = f"l{layer_idx}_"
        setattr(self, f"w_{p}in_norm",
                self._make_norm_weight_expanded(layer.input_layernorm.weight, self.w_sq))
        setattr(self, f"w_{p}q_norm_4d",
                self._make_per_head_norm_weight(layer.self_attn.q_norm.weight, self.n_heads, self.w_sq))
        setattr(self, f"w_{p}k_norm_4d",
                self._make_per_head_norm_weight(layer.self_attn.k_norm.weight, self.n_kv_heads, self.w_kv))
        setattr(self, f"w_{p}post_norm",
                self._make_norm_weight_expanded(layer.post_attention_layernorm.weight, self.w_sq))

    def _load_layer_weights_q8(self, model: nn.Module, layer_idx: int):
        """Quantize one layer's projection weights → Q8LayerWeights pyobject."""
        layer = model.layers[layer_idx]
        sq_w = layer.self_attn
        H, NH, NKV, HD = self.hidden, self.n_heads, self.n_kv_heads, self.head_dim
        INT = self.intermediate

        wq = self._quantize_weight_q8(sq_w.q_proj.weight.astype(mx.float32).reshape(NH * HD, H))
        wk = self._quantize_weight_q8(sq_w.k_proj.weight.astype(mx.float32).reshape(NKV * HD, H))

        wv = self._quantize_weight_q8(sq_w.v_proj.weight.astype(mx.float32).reshape(NKV * HD, H))
        wo = self._quantize_weight_q8(sq_w.o_proj.weight.astype(mx.float32).reshape(H, NH * HD))
        w_gate = self._quantize_weight_q8(layer.mlp.gate_proj.weight.astype(mx.float32).reshape(INT, H))
        w_up = self._quantize_weight_q8(layer.mlp.up_proj.weight.astype(mx.float32).reshape(INT, H))
        w_down = self._quantize_weight_q8(layer.mlp.down_proj.weight.astype(mx.float32).reshape(H, INT))

        return self.ane.Q8LayerWeights(wq, wk, wv, wo, w_gate, w_up, w_down)

    def load_weights_q8(self, draft_model: nn.Module, target_model: nn.Module = None):
        """Load weights as W8A16 int8 and compile per-layer quantized kernels.

        This replaces load_weights() for the quantized inference path.
        After this call, run_kernels() uses per-layer kernels with baked int8 weights
        (no fp16 weight IOSurfaces), reducing ANE memory bandwidth by ~2×.
        """
        # Non-projection weights (fc, norms) — unchanged from fp16 path
        self._load_fc_weights(draft_model)
        self._fc_weight = draft_model.fc.weight.astype(mx.float32)
        self._hidden_norm_weight = draft_model.hidden_norm.weight.astype(mx.float32)

        print(f"[ANE] Loading layer norm weights ({self.n_layers} layers)...")
        for i in range(self.n_layers):
            self._load_layer_norm_weights(draft_model, i)

        print(f"[ANE] Quantizing + compiling {self.n_layers} q8 layer kernels...")
        t0 = time.time()
        layer_weights = [self._load_layer_weights_q8(draft_model, i) for i in range(self.n_layers)]
        layer_kernel_lists = self.ane.compile_dflash_kernels_q8(
            self.seq_q, self.max_ctx_len, self.softcap, layer_weights, self.is_27b,
        )
        self.layer_kernels = [{k.name: k for k in lklist} for lklist in layer_kernel_lists]
        print(f"[ANE] q8 kernels compiled ({time.time() - t0:.1f}s, "
              f"{len(self.layer_kernels)} layers × {len(layer_kernel_lists[0])} kernels)")

        self._load_final_norm_weights(draft_model)
        if self.vocab_size is not None and target_model is not None:
            self._load_lm_head_weights(target_model)

        self.use_q8 = True
        self.weights_loaded = True
        print("[ANE] Weights loaded (q8)")

    def _run_layer_q8(self, layer_idx: int):
        """Run one transformer layer using baked int8 projection weights."""
        lk = self.layer_kernels[layer_idx]
        p = f"l{layer_idx}_"

        # mega_qkv_q8: projection weights baked in — only norm/rope inputs
        lk['mega_qkv_q8'].run_uncached(
            [self.b_hidden, getattr(self, f"w_{p}in_norm"),
             self.b_context,
             getattr(self, f"w_{p}k_norm_4d"),
             self.b_cos_k, self.b_sin_k,
             getattr(self, f"w_{p}q_norm_4d"),
             self.b_cos_q, self.b_sin_q],
            [self.b_k_rope_4d, self.b_v_4d_t, self.b_q_rope_4d],
        )

        lk['gqa_tile'].run_uncached(
            [self.b_k_rope_4d, self.b_v_4d_t],
            [self.b_kv_tiled],
        )

        lk['attn_out'].run_uncached(
            [self.b_q_rope_4d, self.b_kv_tiled, self.b_attn_mask],
            [self.b_attn_flat],
        )

        # o_proj_residual_q8: wo baked in — only attn_flat + residual input
        lk['o_proj_residual_q8'].run_uncached(
            [self.b_attn_flat, self.b_hidden],
            [self.b_attn_res],
        )

        # ffn_residual_q8: gate/up/down baked in — only h1 + post_norm input
        lk['ffn_residual_q8'].run_uncached(
            [self.b_attn_res, getattr(self, f"w_{p}post_norm")],
            [self.b_hidden],
        )

    def forward(self, noise_embedding: mx.array, target_hidden: mx.array,
                rope_offset: int = 0, ctx_len: int = None,
                precomputed_context: mx.array = None) -> mx.array:
        if ctx_len is None:
            ctx_len = target_hidden.shape[1]
        if ctx_len > self.max_ctx_len:
            raise ValueError(
                f"ctx_len={ctx_len} exceeds compiled max_ctx_len={self.max_ctx_len}. "
                f"Re-initialize ANEDraftModel with a larger ctx_len."
            )
        k = self.kernels

        self._write_padded(self._padded_hidden, self.b_hidden, noise_embedding)

        context = precomputed_context if precomputed_context is not None else self._compute_context(target_hidden)
        self._write_padded(self._padded_context, self.b_context, context)

        self._compute_rope(rope_offset, ctx_len)
        self._compute_attn_mask(ctx_len)

        if self.use_q8:
            for i in range(self.n_layers):
                self._run_layer_q8(i)
        else:
            for i in range(self.n_layers):
                self._run_layer(k, i)

        k['final_norm'].run_uncached(
            [self.b_hidden, self.w_final_norm],
            [self.b_output],
        )

        return self._read_mlx_2d(self.b_output, self.seq_q, self.hidden)

    def __call__(self, noise_embedding: mx.array, target_hidden: mx.array,
                 mask=None, cache=None, precomputed_context: mx.array = None,
                 **kwargs) -> mx.array:
        rope_offset = 0
        if cache is not None and len(cache) > 0 and cache[0].offset > 0:
            rope_offset = cache[0].offset
        return self.forward(noise_embedding, target_hidden, rope_offset=rope_offset,
                           ctx_len=target_hidden.shape[1],
                           precomputed_context=precomputed_context)

    def make_cache(self):
        from .dflash import DFlashKVCache
        return [DFlashKVCache() for _ in range(self.n_layers)]

    def _write_padded(self, padded: mx.array, buf, arr: mx.array):
        seq_len = arr.shape[1]
        channels = arr.shape[2]
        f32 = arr.astype(mx.float32).transpose(0, 2, 1)
        padded[:, :, :] = 0.0
        padded[:, :, :seq_len] = f32
        mx.eval(padded)
        buf.write_buffer(memoryview(padded))

    def _read_mlx_2d(self, buf, seq_len: int, channels: int) -> mx.array:
        w = buf.shape[3]
        data = buf.read_f32()
        arr = mx.array(data, dtype=mx.float32).reshape(1, channels, w)[:, :, :seq_len]
        return arr.transpose(0, 2, 1)

    def _run_layer(self, k, layer_idx: int):
        p = f"l{layer_idx}_"

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
        cache_key = (rope_offset, ctx_len)
        if self._rope_cache_key == cache_key:
            return
        self._rope_cache_key = cache_key

        rope_theta = float(self.config.rope_theta)
        half = self.head_dim // 2

        if self._rope_freqs is None:
            self._rope_freqs = mx.array(
                [1.0 / (rope_theta ** (2.0 * d / HEAD_DIM)) for d in range(half)],
                dtype=mx.float32)

        freqs = self._rope_freqs

        # Neox (split-half) RoPE: cos/sin are tiled [f0..f_{hd/2-1}, f0..f_{hd/2-1}]
        # matching mx.fast.rope(traditional=False) used by the GPU draft model.
        def _tile(angles):
            c = mx.concatenate([mx.cos(angles), mx.cos(angles)], axis=1)
            s = mx.concatenate([mx.sin(angles), mx.sin(angles)], axis=1)
            return c, s

        q_positions = mx.array([rope_offset + ctx_len + p for p in range(self.w_sq)], dtype=mx.float32)
        q_cos, q_sin = _tile(q_positions[:, None] * freqs[None, :])
        self.b_cos_q.write_buffer(memoryview(q_cos.flatten().astype(mx.float32)))
        self.b_sin_q.write_buffer(memoryview(q_sin.flatten().astype(mx.float32)))

        k_ctx_pos = mx.array([rope_offset + p for p in range(ctx_len)], dtype=mx.float32)
        k_noise_pos = mx.array([rope_offset + ctx_len + p for p in range(self.w_sq)], dtype=mx.float32)

        k_ctx_cos, k_ctx_sin = _tile(k_ctx_pos[:, None] * freqs[None, :])
        k_noise_cos, k_noise_sin = _tile(k_noise_pos[:, None] * freqs[None, :])

        k_cos_full = mx.zeros((1, 1, self.w_kv, HEAD_DIM), dtype=mx.float32)
        k_sin_full = mx.zeros((1, 1, self.w_kv, HEAD_DIM), dtype=mx.float32)
        k_cos_full[:, :, :ctx_len, :] = k_ctx_cos[None, None]
        k_cos_full[:, :, self.w_ctx:, :] = k_noise_cos[None, None]
        k_sin_full[:, :, :ctx_len, :] = k_ctx_sin[None, None]
        k_sin_full[:, :, self.w_ctx:, :] = k_noise_sin[None, None]
        mx.eval(k_cos_full, k_sin_full)

        self.b_cos_k.write_buffer(memoryview(k_cos_full.flatten().astype(mx.float32)))
        self.b_sin_k.write_buffer(memoryview(k_sin_full.flatten().astype(mx.float32)))

    def _compute_attn_mask(self, ctx_len: int):
        if self._attn_mask_ctx_len == ctx_len:
            return
        self._attn_mask_ctx_len = ctx_len
        mask = mx.full((1, 1, self.w_sq, self.w_kv), -1e4, dtype=mx.float32)
        mask[:, :, :, :ctx_len] = 0.0
        mask[:, :, :, self.w_ctx:self.w_ctx + self.seq_q] = 0.0
        self.b_attn_mask.write_buffer(memoryview(mask.flatten().astype(mx.float32)))

    def prepare_forward(
        self,
        noise_embedding: mx.array,
        precomputed_context: mx.array,
        cache,
        target_hidden: mx.array,
    ) -> None:
        """Write all ANE input buffers on the calling (main) thread.

        Runs all mx.eval() / Metal operations here so that run_prepared()
        can be called from a background thread with zero Metal work —
        enabling true ANE||GPU concurrency without racing on Metal state.
        """
        rope_offset = 0
        if cache is not None and len(cache) > 0 and cache[0].offset > 0:
            rope_offset = cache[0].offset
        ctx_len = target_hidden.shape[1]

        self._write_padded(self._padded_hidden, self.b_hidden, noise_embedding)

        context = precomputed_context if precomputed_context is not None else self._compute_context(target_hidden)
        self._write_padded(self._padded_context, self.b_context, context)

        self._compute_rope(rope_offset, ctx_len)
        self._compute_attn_mask(ctx_len)

    def run_kernels(self) -> None:
        """Execute ANE kernels on already-prepared buffers. No mx ops.

        Pure ANE (IOSurface + CoreML) — safe to call from a background thread
        while the main thread runs GPU verify. Writes result to self.b_output
        (hidden states) and optionally self.b_logits (lm_head logits).
        Call read_output() / read_draft_tokens() on the main thread after join.
        """
        k = self.kernels
        if self.use_q8:
            for i in range(self.n_layers):
                self._run_layer_q8(i)
        else:
            for i in range(self.n_layers):
                self._run_layer(k, i)
        if self.b_logits is not None:
            # Fused final-norm + lm_head → logits [1, vocab_size, 1, w_sq]
            k['final_norm_lm_head'].run_uncached(
                [self.b_hidden, self.w_final_norm, self.w_lm_head],
                [self.b_logits],
            )
        else:
            k['final_norm'].run_uncached(
                [self.b_hidden, self.w_final_norm],
                [self.b_output],
            )

    def read_output(self) -> mx.array:
        """Read b_output into an mx.array. Must be called on the main thread."""
        return self._read_mlx_2d(self.b_output, self.seq_q, self.hidden)

    def read_draft_tokens(self) -> mx.array:
        """Argmax over vocab from b_logits → token ids [1, seq_q].

        Only valid when vocab_size was set at construction. Must be called on
        the main thread after run_kernels() completes.
        """
        if self.b_logits is None:
            raise RuntimeError("vocab_size not set — lm_head kernel not compiled")
        ids = self.b_logits.read_argmax(self.seq_q, self.vocab_size)
        return mx.array(ids, dtype=mx.int32).reshape(1, self.seq_q)

    def run_prepared(self) -> mx.array:
        """Convenience: run_kernels() + read_output() for single-threaded callers."""
        self.run_kernels()
        return self.read_output()

    def set_block_size(self, new_seq_q: int) -> None:
        """Change the active block size without recompiling kernels.

        Valid for any new_seq_q in [1, w_sq] (w_sq = align_width(original seq_q)).
        All seq_q values that map to the same w_sq share identical compiled kernels
        and buffer layouts, so this is safe to call at any time between forward passes.
        The attn-mask and RoPE caches are invalidated so they are recomputed on
        the next forward call with the correct dimensions.
        """
        if new_seq_q < 1 or new_seq_q > self.w_sq:
            raise ValueError(
                f"new_seq_q={new_seq_q} must be in [1, {self.w_sq}] "
                f"(kernel compiled for w_sq={self.w_sq})"
            )
        self.seq_q = new_seq_q
        self.block_size = new_seq_q
        self.config.block_size = new_seq_q
        # Invalidate caches so _compute_rope / _compute_attn_mask rebuild them.
        self._rope_cache_key = None
        self._attn_mask_ctx_len = None

    def _compute_context(self, target_hidden: mx.array) -> mx.array:
        fc_out = target_hidden @ self._fc_weight.T
        rms = mx.sqrt(mx.mean(fc_out.astype(mx.float32) ** 2, axis=-1, keepdims=True) + 1e-6)
        context = (fc_out / rms) * self._hidden_norm_weight
        mx.eval(context)
        return context.astype(mx.float32)
