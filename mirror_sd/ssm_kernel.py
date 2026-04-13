"""Custom Metal kernel for GatedDeltaNet SSM state replay.

Ported from dflash-mlx (bstnxbt/dflash-mlx). Replaces the Python
per-token loop in _advance_gated_delta_states with a single Metal
dispatch that runs the entire recurrence on GPU.

The kernel processes all (batch, head, value_dim) combinations in parallel
via a 3D grid (32, Dv, B*Hv). Each thread processes Dk/32 elements of the
state vector, using simd_sum for the dot-product reduction.
"""

import mlx.core as mx


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


_KERNEL = None


def _get_kernel():
    global _KERNEL
    if _KERNEL is None:
        _KERNEL = _make_gated_delta_state_kernel()
    return _KERNEL


def advance_gated_delta_states_metal(
    initial_states: mx.array,
    keys: mx.array,
    values: mx.array,
    g: mx.array,
    beta: mx.array,
) -> mx.array:
    """Advance GatedDeltaNet SSM state using custom Metal kernel.

    Replaces the Python per-token loop with a single GPU dispatch.
    Falls back to the Python loop if Metal is unavailable.

    Args:
        initial_states: [B, Hv, Dv, Dk] initial SSM state
        keys: [B, T, Hv, Dk] keys (already repeated for GQA)
        values: [B, T, Hv, Dv] values
        g: [B, T, Hv] gate values
        beta: [B, T, Hv] beta values

    Returns:
        Updated SSM state [B, Hv, Dv, Dk]
    """
    kernel = _get_kernel()
    if (
        kernel is not None
        and mx.default_device() == mx.gpu
        and keys.shape[-1] % 32 == 0
    ):
        batch_size, _, _, head_dim = keys.shape
        num_v_heads = values.shape[2]
        value_dim = values.shape[-1]

        output = kernel(
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
        if isinstance(output, (list, tuple)):
            return output[0]
        return output

    return _advance_gated_delta_states_python(initial_states, keys, values, g, beta)


def _advance_gated_delta_states_python(initial_state, keys, values, g, beta):
    """Python fallback: advance SSM state token-by-token."""
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