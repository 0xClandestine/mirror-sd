# Optimized Metal kernels for speculative decoding.
# Based on DFlash (arXiv:2602.06036) — MIT License
# Adapted from dflash-mlx/dflash_mlx/kernels.py

from __future__ import annotations

from typing import Optional

import mlx.core as mx


# ── GatedDeltaNet SSM state replay ─────────────────────────────────────────────

def _make_gated_delta_state_kernel():
    if not mx.metal.is_available():
        return None

    source = r"""
        auto n = thread_position_in_grid.z;
        auto b_idx = n / Hv;
        auto hv_idx = n % Hv;
        auto dv_idx = thread_position_in_grid.y;
        constexpr int n_per_t = Dk / 32;

        auto k_ = k + (b_idx * T * Hv + hv_idx) * Dk;
        auto v_ = v + (b_idx * T * Hv + hv_idx) * Dv;
        auto g_ = g + b_idx * T * Hv;
        auto beta_ = beta + b_idx * T * Hv;

        auto i_state = state_in + (n * Dv + dv_idx) * Dk;
        auto o_state = state_out + (n * Dv + dv_idx) * Dk;

        float state[n_per_t];
        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
          state[i] = static_cast<float>(i_state[s_idx]);
        }

        for (int t = 0; t < T; ++t) {
          float kv_mem = 0.0f;
          auto g_t = g_[hv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
            state[i] = state[i] * g_t;
            kv_mem += state[i] * static_cast<float>(k_[s_idx]);
          }
          kv_mem = simd_sum(kv_mem);

          auto delta = (static_cast<float>(v_[dv_idx]) - kv_mem) * beta_[hv_idx];
          for (int i = 0; i < n_per_t; ++i) {
            auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
            state[i] = state[i] + static_cast<float>(k_[s_idx]) * delta;
          }

          k_ += Hv * Dk;
          v_ += Hv * Dv;
          g_ += Hv;
          beta_ += Hv;
        }

        for (int i = 0; i < n_per_t; ++i) {
          auto s_idx = n_per_t * thread_position_in_threadgroup.x + i;
          o_state[s_idx] = static_cast<InT>(state[i]);
        }
    """
    return mx.fast.metal_kernel(
        name="gated_delta_state_update",
        input_names=["k", "v", "g", "beta", "state_in", "T"],
        output_names=["state_out"],
        source=source,
    )


_gda_state_kernel = _make_gated_delta_state_kernel()


def _advance_gated_delta_states_python(initial_state, keys, values, g, beta):
    state = initial_state.astype(mx.float32)
    keys_f = keys.astype(mx.float32)
    values_f = values.astype(mx.float32)
    g_f = g.astype(mx.float32)
    beta_f = beta.astype(mx.float32)
    for t in range(keys.shape[1]):
        state = state * g_f[:, t, :, None, None]
        kv_mem = mx.sum(state * keys_f[:, t, :, None, :], axis=-1)
        delta = (values_f[:, t] - kv_mem) * beta_f[:, t, :, None]
        state = state + delta[..., None] * keys_f[:, t, :, None, :]
    return state.astype(initial_state.dtype)


def advance_gated_delta_states_metal(
    initial_states: mx.array,
    keys: mx.array,
    values: mx.array,
    g: mx.array,
    beta: mx.array,
) -> mx.array:
    """Advance GatedDeltaNet SSM state using a single Metal kernel dispatch.

    Replaces the Python per-token loop in rollback with one GPU dispatch.
    Falls back to the Python loop if Metal is unavailable or Dk % 32 != 0.

    Args:
        initial_states: [B, Hv, Dv, Dk]
        keys:           [B, T, Hv, Dk] (already GQA-repeated)
        values:         [B, T, Hv, Dv]
        g:              [B, T, Hv]
        beta:           [B, T, Hv]

    Returns:
        Updated SSM state [B, Hv, Dv, Dk]
    """
    if (
        _gda_state_kernel is not None
        and mx.default_device() == mx.gpu
        and keys.shape[-1] % 32 == 0
    ):
        batch_size, _, _, head_dim = keys.shape
        num_v_heads = values.shape[2]
        value_dim = values.shape[-1]
        output = _gda_state_kernel(
            inputs=[keys, values, g, beta, initial_states, keys.shape[1]],
            template=[
                ("InT", initial_states.dtype),
                ("Dk", head_dim),
                ("Dv", value_dim),
                ("Hv", num_v_heads),
            ],
            grid=(32, value_dim, batch_size * num_v_heads),
            threadgroup=(32, 4, 1),
            output_shapes=[initial_states.shape],
            output_dtypes=[initial_states.dtype],
        )
        return output[0] if isinstance(output, (list, tuple)) else output

    return _advance_gated_delta_states_python(initial_states, keys, values, g, beta)


# ── 2-pass SDPA for q_len=16 verify ────────────────────────────────────────────


def _compute_sdpa_2pass_blocks(gqa_factor: int, n_kv: int, device_arch: Optional[str] = None) -> int:
    arch = device_arch or str(mx.device_info().get("architecture", ""))
    devc = arch[-1] if arch else ""
    n_simds = int(gqa_factor)
    N = int(n_kv)

    if devc == "d":
        blocks = 128
        if n_simds <= 2 and N > 8192:
            blocks = 256
        elif n_simds >= 6:
            if 16384 <= N < 65536:
                blocks = 512
            elif N >= 65536:
                blocks = 1024
    elif devc == "s":
        blocks = 64
        if N > 1024 and n_simds > 4:
            if N <= 8192:
                blocks = 128
            elif N <= 32768:
                blocks = 256
            elif N <= 65536:
                blocks = 512
            else:
                blocks = 1024
    else:
        blocks = 64 if n_simds >= 4 else 32

    return int(blocks)


def _make_batched_sdpa_2pass_partials_kernel(*, has_mask: bool = False):
    if not mx.metal.is_available():
        return None

    mask_setup = ""
    mask_use_key = ""
    mask_score = ""
    mask_advance = ""
    inputs = [
        "queries",
        "keys",
        "values",
        "gqa_factor",
        "N",
        "k_head_stride",
        "k_seq_stride",
        "v_head_stride",
        "v_seq_stride",
        "scale",
        "blocks",
    ]
    if has_mask:
        inputs.append("mask")
        mask_setup = """
        auto mask_ = mask + (((b_idx * Hq + q_head_idx) * M_FIXED + q_seq_idx) * N + block_idx);
        """
        mask_use_key = """
            auto mask_value = static_cast<float>(mask_[0]);
            use_key = use_key && (mask_value >= Limits<InT>::finite_min);
        """
        mask_score = """
            score += static_cast<float>(mask_[0]);
        """
        mask_advance = """
            mask_ += blocks;
        """

    source = f"""
        constexpr int BD = 32;
        constexpr int qk_per_thread = D / BD;
        constexpr int v_per_thread = V / BD;

        auto q_head_idx = threadgroup_position_in_grid.x;
        auto b_idx = threadgroup_position_in_grid.y;
        auto block_idx = threadgroup_position_in_grid.z;
        auto q_seq_idx = thread_position_in_threadgroup.z;
        auto simd_lid = thread_index_in_simdgroup;

        auto Hq = threadgroups_per_grid.x;
        auto hk_idx = q_head_idx / gqa_factor;
        auto q_batch_head_idx = b_idx * Hq + q_head_idx;
        auto o_offset = q_batch_head_idx * M_FIXED + q_seq_idx;

        auto q_ = queries + (o_offset * D) + simd_lid * qk_per_thread;
        auto k_ = keys + ((b_idx * Hk + hk_idx) * k_head_stride) + block_idx * k_seq_stride + simd_lid * qk_per_thread;
        auto v_ = values + ((b_idx * Hk + hk_idx) * v_head_stride) + block_idx * v_seq_stride + simd_lid * v_per_thread;

        partials += (o_offset * blocks + block_idx) * V + simd_lid * v_per_thread;
        sums += o_offset * blocks + block_idx;
        maxs += o_offset * blocks + block_idx;
        {mask_setup}

        thread float q[qk_per_thread];
        thread float o[v_per_thread];
        threadgroup InT tg_k[BD * qk_per_thread];
        threadgroup InT tg_v[BD * v_per_thread];

        for (int i = 0; i < qk_per_thread; ++i) {{
            q[i] = static_cast<float>(scale) * static_cast<float>(q_[i]);
        }}
        for (int i = 0; i < v_per_thread; ++i) {{
            o[i] = 0.0f;
        }}

        float max_score = Limits<float>::finite_min;
        float sum_exp_score = 0.0f;

        for (int n = block_idx; n < N; n += blocks) {{
            if (q_seq_idx == 0) {{
                for (int i = 0; i < qk_per_thread; ++i) {{
                    tg_k[simd_lid * qk_per_thread + i] = k_[i];
                }}
                for (int i = 0; i < v_per_thread; ++i) {{
                    tg_v[simd_lid * v_per_thread + i] = v_[i];
                }}
            }}
            threadgroup_barrier(mem_flags::mem_threadgroup);

            bool use_key = (n <= (N - M_FIXED + q_seq_idx));
            {mask_use_key}

            if (use_key) {{
                float score = 0.0f;
                for (int i = 0; i < qk_per_thread; ++i) {{
                    score += q[i] * static_cast<float>(tg_k[simd_lid * qk_per_thread + i]);
                }}
                score = simd_sum(score);
                {mask_score}

                float new_max = metal::max(max_score, score);
                float factor = fast::exp(max_score - new_max);
                float exp_score = fast::exp(score - new_max);

                max_score = new_max;
                sum_exp_score = sum_exp_score * factor + exp_score;
                for (int i = 0; i < v_per_thread; ++i) {{
                    o[i] = o[i] * factor + exp_score * static_cast<float>(tg_v[simd_lid * v_per_thread + i]);
                }}
            }}

            threadgroup_barrier(mem_flags::mem_threadgroup);
            k_ += blocks * int(k_seq_stride);
            v_ += blocks * int(v_seq_stride);
            {mask_advance}
        }}

        if (simd_lid == 0) {{
            sums[0] = sum_exp_score;
            maxs[0] = max_score;
        }}
        for (int i = 0; i < v_per_thread; ++i) {{
            partials[i] = static_cast<InT>(o[i]);
        }}
    """

    return mx.fast.metal_kernel(
        name=f"batched_sdpa_2pass_partials{'_mask' if has_mask else ''}",
        input_names=inputs,
        output_names=["partials", "sums", "maxs"],
        source=source,
    )


def _make_batched_sdpa_2pass_reduce_kernel():
    if not mx.metal.is_available():
        return None

    source = """
        constexpr int BN = 32;
        constexpr int BD = 32;
        constexpr int elem_per_thread = V / BD;

        auto head_idx = threadgroup_position_in_grid.x;
        auto q_seq_idx = threadgroup_position_in_grid.y;
        auto simd_gid = simdgroup_index_in_threadgroup;
        auto simd_lid = thread_index_in_simdgroup;

        auto q_offset = head_idx * M_FIXED + q_seq_idx;
        partials += (q_offset * blocks + simd_gid) * V + simd_lid * elem_per_thread;
        sums += q_offset * blocks;
        maxs += q_offset * blocks;
        out += q_offset * V + simd_gid * elem_per_thread;

        thread float o[elem_per_thread];
        threadgroup float outputs[BN * BD];

        for (int i = 0; i < elem_per_thread; ++i) {
            o[i] = 0.0f;
        }

        float sum_exp_score = 0.0f;
        float max_score = Limits<float>::finite_min;

        for (int b = 0; b < blocks / BN; ++b) {
            max_score = metal::max(max_score, maxs[simd_lid + BN * b]);
        }
        max_score = simd_max(max_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_lid + BN * b] - max_score);
            sum_exp_score += factor * sums[simd_lid + BN * b];
        }
        sum_exp_score = simd_sum(sum_exp_score);

        for (int b = 0; b < blocks / BN; ++b) {
            float factor = fast::exp(maxs[simd_gid] - max_score);
            for (int i = 0; i < elem_per_thread; ++i) {
                o[i] += factor * static_cast<float>(partials[i]);
            }
            maxs += BN;
            partials += BN * V;
        }

        for (int i = 0; i < elem_per_thread; ++i) {
            outputs[simd_lid * BD + simd_gid] = o[i];
            threadgroup_barrier(mem_flags::mem_threadgroup);
            o[i] = simd_sum(outputs[simd_gid * BD + simd_lid]);
            o[i] = sum_exp_score == 0.0f ? o[i] : (o[i] / sum_exp_score);
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (simd_lid == 0) {
            for (int i = 0; i < elem_per_thread; ++i) {
                out[i] = static_cast<InT>(o[i]);
            }
        }
    """

    return mx.fast.metal_kernel(
        name="batched_sdpa_2pass_reduce",
        input_names=["partials", "sums", "maxs", "blocks"],
        output_names=["out"],
        source=source,
    )


_batched_sdpa_2pass_partials_kernel = _make_batched_sdpa_2pass_partials_kernel(has_mask=False)
_batched_sdpa_2pass_partials_kernel_masked = _make_batched_sdpa_2pass_partials_kernel(has_mask=True)
_batched_sdpa_2pass_reduce_kernel = _make_batched_sdpa_2pass_reduce_kernel()


def batched_sdpa_2pass_exact(
    queries: mx.array,
    keys: mx.array,
    values: mx.array,
    scale: float,
    mask: Optional[mx.array] = None,
) -> Optional[mx.array]:
    """2-pass SDPA kernel optimized for exactly q_len=16 (DFlash verify step).

    Returns None if requirements are not met (non-16 block size, unsupported
    dtype/head_dim, CPU device), in which case caller should fall back to
    mx.fast.scaled_dot_product_attention.

    queries: [B, Hq, 16, D]
    keys:    [B, Hk, n_kv, D]
    values:  [B, Hk, n_kv, D]

    Causal masking is applied implicitly — query i only attends to keys 0..i
    within the block plus all prefill tokens. No explicit mask needed for the
    standard speculative verify use case.
    """
    if not mx.metal.is_available():
        return None

    if queries.ndim != 4 or keys.ndim != 4 or values.ndim != 4:
        return None

    bsz, hq, q_len, d = queries.shape
    _, hk, n_kv, _ = keys.shape
    vdim = values.shape[-1]
    input_type = queries.dtype

    if q_len != 16:
        return None
    if input_type not in (mx.bfloat16, mx.float16):
        return None
    if d not in (128, 256) or vdim not in (128, 256) or d != vdim:
        return None
    if hk <= 0 or hq % hk != 0:
        return None

    queries = mx.contiguous(queries)
    keys = mx.contiguous(keys)
    values = mx.contiguous(values)

    gqa_factor = hq // hk
    blocks = _compute_sdpa_2pass_blocks(gqa_factor, n_kv)
    if blocks <= 0 or blocks % 32 != 0:
        return None

    k_head_stride = keys.shape[2] * keys.shape[3]
    k_seq_stride = keys.shape[3]
    v_head_stride = values.shape[2] * values.shape[3]
    v_seq_stride = values.shape[3]

    kernel = _batched_sdpa_2pass_partials_kernel
    inputs = [
        queries,
        keys,
        values,
        gqa_factor,
        n_kv,
        k_head_stride,
        k_seq_stride,
        v_head_stride,
        v_seq_stride,
        float(scale),
        blocks,
    ]

    if mask is not None:
        input_min = mx.finfo(input_type).min
        if mask.dtype == mx.bool_:
            mask_tensor = mx.where(
                mask,
                mx.zeros(mask.shape, dtype=input_type),
                mx.full(mask.shape, input_min, dtype=input_type),
            )
        else:
            mask_tensor = mask.astype(input_type) if mask.dtype != input_type else mask
        mask_tensor = mx.broadcast_to(mask_tensor, (bsz, hq, q_len, n_kv))
        mask_tensor = mx.contiguous(mask_tensor)
        kernel = _batched_sdpa_2pass_partials_kernel_masked
        inputs.append(mask_tensor)

    if kernel is None or _batched_sdpa_2pass_reduce_kernel is None:
        return None

    partial_shape = (bsz * hq, q_len, blocks, vdim)
    stats_shape = (bsz * hq, q_len, blocks)
    partials, sums, maxs = kernel(
        inputs=inputs,
        template=[
            ("InT", input_type),
            ("D", d),
            ("V", vdim),
            ("Hk", hk),
            ("M_FIXED", q_len),
        ],
        grid=(hq * 32, bsz, blocks * q_len),
        threadgroup=(32, 1, q_len),
        output_shapes=[partial_shape, stats_shape, stats_shape],
        output_dtypes=[input_type, mx.float32, mx.float32],
    )

    (out,) = _batched_sdpa_2pass_reduce_kernel(
        inputs=[partials, sums, maxs, blocks],
        template=[
            ("InT", input_type),
            ("V", vdim),
            ("M_FIXED", q_len),
        ],
        grid=((bsz * hq) * 1024, q_len, 1),
        threadgroup=(1024, 1, 1),
        output_shapes=[queries.shape],
        output_dtypes=[input_type],
    )
    return out
