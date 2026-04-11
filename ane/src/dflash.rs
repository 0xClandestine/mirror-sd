use ane::{Graph, Shape, Tensor, MIN_SPATIAL_WIDTH};

pub const HIDDEN: usize = 4096;
pub const HEAD_DIM: usize = 128;
pub const N_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 8;
pub const INTERMEDIATE: usize = 12288;
pub const N_LAYERS: usize = 5;
pub const N_TARGET_FEATURES: usize = 5;
pub const TARGET_HIDDEN: usize = N_TARGET_FEATURES * HIDDEN;
const GQA_RATIO: usize = N_HEADS / N_KV_HEADS;

pub fn align_width(w: usize) -> usize {
    let aligned = ((w + MIN_SPATIAL_WIDTH - 1) / MIN_SPATIAL_WIDTH) * MIN_SPATIAL_WIDTH;
    aligned.max(MIN_SPATIAL_WIDTH)
}

pub fn rmsnorm(g: &mut Graph, x: Tensor, weight: Tensor) -> Tensor {
    let inv_s = g.constant_with_scalar(
        1.0 / 128.0,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let x_scaled = g.multiplication(x, inv_s);
    let ms = g.reduce_mean(x_scaled, 1);
    let diff = g.subtraction(x_scaled, ms);
    let sq = g.multiplication(diff, diff);
    let mean_sq = g.reduce_mean(sq, 1);
    let eps_t = g.constant_with_scalar(
        1e-6,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let mean_sq_eps = g.addition(mean_sq, eps_t);
    let neg_half = g.constant_with_scalar(
        -0.5,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let inv_std = g.power(mean_sq_eps, neg_half);
    let normed = g.multiplication(x_scaled, inv_std);
    g.multiplication(normed, weight)
}

fn tile_kv_heads(
    g: &mut Graph,
    kv: Tensor,
    kv_heads: usize,
    gqa_ratio: usize,
    seq: usize,
    hd: usize,
) -> Tensor {
    if gqa_ratio == 1 {
        return kv;
    }
    let mut tiled = Vec::with_capacity(kv_heads * gqa_ratio);
    for kv_head in 0..kv_heads {
        let head = g.slice(kv, [0, kv_head, 0, 0], [1, 1, seq, hd]);
        for _ in 0..gqa_ratio {
            tiled.push(head);
        }
    }
    g.concat(&tiled, 1)
}

fn apply_rope(
    g: &mut Graph,
    x: Tensor,
    cos: Tensor,
    sin: Tensor,
    n_heads: usize,
    seq: usize,
    hd: usize,
) -> Tensor {
    let pairs = seq * hd / 2;
    let xp = g.reshape(
        x,
        Shape {
            batch: 1,
            channels: n_heads,
            height: pairs,
            width: 2,
        },
    );
    let x_e = g.slice(xp, [0, 0, 0, 0], [1, n_heads, pairs, 1]);
    let x_o = g.slice(xp, [0, 0, 0, 1], [1, n_heads, pairs, 1]);
    let neg1 = g.constant_with_scalar(
        -1.0,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let neg_xo = g.multiplication(x_o, neg1);
    let rotated = g.concat(&[neg_xo, x_e], 3);
    let x_rot = g.reshape(
        rotated,
        Shape {
            batch: 1,
            channels: n_heads,
            height: seq,
            width: hd,
        },
    );
    let xc = g.multiplication(x, cos);
    let xs = g.multiplication(x_rot, sin);
    g.addition(xc, xs)
}

fn conv1x1_proj(
    g: &mut Graph,
    input: Tensor,
    weight: Tensor,
    oc: usize,
    ic: usize,
    seq: usize,
) -> Tensor {
    let packed = g.concat(&[input, weight], 3);
    let a = g.slice(packed, [0, 0, 0, 0], [1, ic, 1, seq]);
    let w = g.slice(packed, [0, 0, 0, seq], [1, ic, 1, oc]);
    let wt = g.transpose(w, [0, 3, 2, 1]);
    let w_conv = g.reshape(
        wt,
        Shape {
            batch: oc,
            channels: ic,
            height: 1,
            width: 1,
        },
    );
    g.convolution_2d_1x1_dynamic(a, w_conv)
}

// ANE rule: norm weight spatial width MUST match input spatial width
// (ANE does not broadcast mismatched spatial dims)

/// Fused attention layer: QKV projections + per-head norm + RoPE + GQA tile + SDPA (with mask)
/// + flatten + o_proj + residual.
///
/// This mega-kernel eliminates ALL Python round-trips per layer for the attention path.
/// The FFN is kept separate (it's already fast as a single kernel).
///
/// Inputs:
///   hidden [1, HIDDEN, 1, w_sq]         - noise hidden state
///   context [1, HIDDEN, 1, w_ctx]        - context hidden state (after fc_norm)
///   in_norm_w [1, HIDDEN, 1, w_sq]       - input layernorm weight
///   wq [1, HIDDEN, 1, N_HEADS*HEAD_DIM]  - Q projection weight (interleaved)
///   wk [1, HIDDEN, 1, N_KV_HEADS*HEAD_DIM] - K projection weight (interleaved)
///   wv [1, HIDDEN, 1, N_KV_HEADS*HEAD_DIM] - V projection weight
///   q_norm_w [1, HEAD_DIM, 1, N_HEADS*w_sq] - per-head Q norm weight (interleaved)
///   k_norm_w [1, HEAD_DIM, 1, N_KV_HEADS*w_kv] - per-head K norm weight (interleaved)
///   cos_q [1, 1, w_sq, HEAD_DIM]         - RoPE cos for Q
///   sin_q [1, 1, w_sq, HEAD_DIM]         - RoPE sin for Q
///   cos_k [1, 1, w_kv, HEAD_DIM]         - RoPE cos for K
///   sin_k [1, 1, w_kv, HEAD_DIM]         - RoPE sin for K
///   attn_mask [1, 1, w_sq, w_kv]         - attention mask (0=valid, -1e4=masked)
///   wo [1, N_HEADS*HEAD_DIM, 1, HIDDEN]  - output projection weight
///
/// Output:
///   attn_res [1, HIDDEN, 1, w_sq]        - attention output + residual
pub fn build_fused_attn_layer_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    // --- Inputs ---
    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });

    // --- RMSNorm(hidden) ---
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    // --- Q projection ---
    let wq = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_HEADS * HEAD_DIM,
    });
    let q_out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);

    // --- Q: reshape + transpose for per-head norm ---
    let q_4d = g.reshape(
        q_out,
        Shape {
            batch: 1,
            channels: N_HEADS,
            height: HEAD_DIM,
            width: w_sq,
        },
    );
    let q_t = g.transpose(q_4d, [0, 2, 1, 3]); // [1, HEAD_DIM, N_HEADS, w_sq]
    let q_for_norm = g.reshape(
        q_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_HEADS * w_sq,
        },
    );

    // --- Per-head Q norm ---
    let q_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_HEADS * w_sq,
    });
    let q_normed = rmsnorm(&mut g, q_for_norm, q_norm_w);

    // --- Q: transpose back for RoPE ---
    let q_norm_4d = g.reshape(
        q_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_HEADS,
            width: w_sq,
        },
    );
    let q_norm_t = g.transpose(q_norm_4d, [0, 2, 3, 1]); // [1, N_HEADS, w_sq, HEAD_DIM]

    // --- RoPE on Q ---
    let cos_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let sin_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let q_rope = apply_rope(&mut g, q_norm_t, cos_q, sin_q, N_HEADS, w_sq, HEAD_DIM);

    // --- K ctx projection ---
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_ctx = conv1x1_proj(&mut g, context, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_ctx);

    // --- K noise projection ---
    let k_noise = conv1x1_proj(&mut g, normed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_sq);

    // --- K concat ---
    let k_concat = g.concat(&[k_ctx, k_noise], 3);

    // --- K: reshape + transpose for per-head norm ---
    let k_4d = g.reshape(
        k_concat,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]); // [1, HEAD_DIM, N_KV_HEADS, w_kv]
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );

    // --- Per-head K norm ---
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);

    // --- K: transpose back for RoPE ---
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]); // [1, N_KV_HEADS, w_kv, HEAD_DIM]

    // --- RoPE on K ---
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    // --- V projections ---
    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let v_ctx = conv1x1_proj(&mut g, context, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_ctx);
    let v_noise = conv1x1_proj(&mut g, normed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_sq);
    let v_concat = g.concat(&[v_ctx, v_noise], 3);

    // --- V: reshape + transpose to 4D ---
    let v_4d = g.reshape(
        v_concat,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let v_4d_t = g.transpose(v_4d, [0, 1, 3, 2]); // [1, N_KV_HEADS, w_kv, HEAD_DIM]

    // --- GQA tile ---
    let k_tiled = tile_kv_heads(&mut g, k_rope, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);
    let v_tiled = tile_kv_heads(&mut g, v_4d_t, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    // --- SDPA ---
    let scores = g.matrix_multiplication(q_rope, k_tiled, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let scores_scaled = g.multiplication(scores, scale);

    let attn_mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: w_kv,
    });
    let masked_scores = g.addition(scores_scaled, attn_mask);
    let attn_probs = g.soft_max(masked_scores, 3);
    let attn_out_4d = g.matrix_multiplication(attn_probs, v_tiled, false, false);

    // --- Flatten attn output ---
    let attn_t = g.transpose(attn_out_4d, [0, 1, 3, 2]); // [1, N_HEADS, HEAD_DIM, w_sq]
    let attn_flat = g.reshape(
        attn_t,
        Shape {
            batch: 1,
            channels: N_HEADS * HEAD_DIM,
            height: 1,
            width: w_sq,
        },
    );

    // --- o_proj + residual ---
    let wo = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS * HEAD_DIM,
        height: 1,
        width: HIDDEN,
    });
    let o_proj = conv1x1_proj(&mut g, attn_flat, wo, HIDDEN, N_HEADS * HEAD_DIM, w_sq);

    let _out = g.addition(hidden, o_proj);
    g
}

/// Fused K path: k_proj_ctx + k_proj_noise + k_concat + reshape + transpose + per-head k_norm
/// + transpose back + rope_k.
/// Input: context [1, HIDDEN, 1, w_ctx], hidden [1, HIDDEN, 1, w_sq], in_norm_w, k_proj_w,
///         k_norm_w, cos_k, sin_k
/// Output: K after rope [1, N_KV_HEADS, w_kv, HEAD_DIM]
pub fn build_fused_k_path_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_ctx = conv1x1_proj(&mut g, context, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_ctx);

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);
    let k_noise = conv1x1_proj(&mut g, normed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_sq);

    // Concat ctx + noise
    let k_concat = g.concat(&[k_ctx, k_noise], 3);

    // Reshape+transpose for per-head k_norm
    let k_4d = g.reshape(
        k_concat,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]); // [1, HEAD_DIM, N_KV_HEADS, w_kv]
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );

    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);

    // Transpose back: [1, HEAD_DIM, 1, N_KV_HEADS*w_kv] → [1, N_KV_HEADS, w_kv, HEAD_DIM]
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]); // [1, N_KV_HEADS, w_kv, HEAD_DIM]

    // RoPE
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let _out = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);
    g
}

/// Fused V path: v_proj_ctx + v_proj_noise + v_concat + reshape + transpose.
/// Input: context [1, HIDDEN, 1, w_ctx], hidden [1, HIDDEN, 1, w_sq], in_norm_w, v_proj_w
/// Output: V in 4D [1, N_KV_HEADS, w_kv, HEAD_DIM]
pub fn build_fused_v_path_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let v_ctx = conv1x1_proj(&mut g, context, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_ctx);

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);
    let v_noise = conv1x1_proj(&mut g, normed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_sq);

    // Concat ctx + noise
    let v_concat = g.concat(&[v_ctx, v_noise], 3);

    // Reshape+transpose: [1, N_KV_HEADS*HEAD_DIM, 1, w_kv] → [1, N_KV_HEADS, w_kv, HEAD_DIM]
    let v_4d = g.reshape(
        v_concat,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let _out = g.transpose(v_4d, [0, 1, 3, 2]); // [1, N_KV_HEADS, w_kv, HEAD_DIM]
    g
}

/// Fused attention + o_proj + residual:
/// gqa_tile + attn_out (with mask) + flatten + o_proj + residual add.
/// Input: q_rope_4d [1, N_HEADS, w_sq, HEAD_DIM], k_rope_4d [1, N_KV_HEADS, w_kv, HEAD_DIM],
///         v_4d [1, N_KV_HEADS, w_kv, HEAD_DIM], attn_mask [1, 1, w_sq, w_kv],
///         o_proj_w, hidden (residual)
/// Output: attn_res [1, HIDDEN, 1, w_sq]
pub fn build_fused_attn_out_kernel(w_sq: usize, w_kv: usize) -> Graph {
    let mut g = Graph::new();

    // GQA tile
    let k4 = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let k_t = tile_kv_heads(&mut g, k4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    let v4 = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let v_t = tile_kv_heads(&mut g, v4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    // Attention
    let q = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_sq,
        width: HEAD_DIM,
    });

    let scores = g.matrix_multiplication(q, k_t, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let scores_scaled = g.multiplication(scores, scale);

    let attn_mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: w_kv,
    });
    let masked_scores = g.addition(scores_scaled, attn_mask);
    let attn_probs = g.soft_max(masked_scores, 3);
    let attn_out_4d = g.matrix_multiplication(attn_probs, v_t, false, false);
    // attn_out_4d is [1, N_HEADS, w_sq, HEAD_DIM]

    // Flatten: [1, N_HEADS, w_sq, HEAD_DIM] → [1, N_HEADS*HEAD_DIM, 1, w_sq]
    // transpose [0, 1, 3, 2] → [1, N_HEADS, HEAD_DIM, w_sq] → reshape
    let attn_t = g.transpose(attn_out_4d, [0, 1, 3, 2]); // [1, N_HEADS, HEAD_DIM, w_sq]
    let attn_flat = g.reshape(
        attn_t,
        Shape {
            batch: 1,
            channels: N_HEADS * HEAD_DIM,
            height: 1,
            width: w_sq,
        },
    );

    // o_proj + residual
    let wo = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS * HEAD_DIM,
        height: 1,
        width: HIDDEN,
    });
    let o_proj = conv1x1_proj(&mut g, attn_flat, wo, HIDDEN, N_HEADS * HEAD_DIM, w_sq);

    let h_res = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let _out = g.addition(h_res, o_proj);
    g
}

/// Full attention layer mega-kernel using INPUT-PACK approach for K and V.
///
/// Key innovation: instead of separate k_proj_ctx + k_proj_noise + concat (which fails
/// at runtime due to concat→reshape→transpose pattern), we pack context+normed into one
/// tensor and do a single conv1x1 for K (and V). Mathematically equivalent.
///
/// This eliminates ALL Python round-trips per layer for the attention path.
///
/// Inputs:
///   hidden [1, HIDDEN, 1, w_sq]           - noise hidden state
///   context [1, HIDDEN, 1, w_ctx]          - context hidden state (after fc_norm)
///   in_norm_w [1, HIDDEN, 1, w_sq]         - input layernorm weight
///   wq [1, HIDDEN, 1, N_HEADS*HEAD_DIM]    - Q projection weight (interleaved)
///   wk [1, HIDDEN, 1, N_KV_HEADS*HEAD_DIM] - K projection weight (interleaved)
///   wv [1, HIDDEN, 1, N_KV_HEADS*HEAD_DIM] - V projection weight
///   q_norm_w [1, HEAD_DIM, 1, N_HEADS*w_sq] - per-head Q norm weight (interleaved)
///   k_norm_w [1, HEAD_DIM, 1, N_KV_HEADS*w_kv] - per-head K norm weight (interleaved)
///   cos_q [1, 1, w_sq, HEAD_DIM]           - RoPE cos for Q
///   sin_q [1, 1, w_sq, HEAD_DIM]           - RoPE sin for Q
///   cos_k [1, 1, w_kv, HEAD_DIM]           - RoPE cos for K
///   sin_k [1, 1, w_kv, HEAD_DIM]           - RoPE sin for K
///   attn_mask [1, 1, w_sq, w_kv]           - attention mask (0=valid, -1e4=masked)
///   wo [1, N_HEADS*HEAD_DIM, 1, HIDDEN]    - output projection weight
///
/// Output:
///   attn_res [1, HIDDEN, 1, w_sq]          - attention output + residual
pub fn build_mega_attn_layer_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    // --- Inputs ---
    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });

    // --- RMSNorm(hidden) ---
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    // --- Pack context + normed for K/V projections ---
    let packed = g.concat(&[context, normed], 3); // [1, HIDDEN, 1, w_kv]

    // --- Q projection (on normed only) ---
    let wq = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_HEADS * HEAD_DIM,
    });
    let q_out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);

    // --- Q: reshape + transpose for per-head norm ---
    let q_4d = g.reshape(
        q_out,
        Shape {
            batch: 1,
            channels: N_HEADS,
            height: HEAD_DIM,
            width: w_sq,
        },
    );
    let q_t = g.transpose(q_4d, [0, 2, 1, 3]); // [1, HEAD_DIM, N_HEADS, w_sq]
    let q_for_norm = g.reshape(
        q_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_HEADS * w_sq,
        },
    );

    // --- Per-head Q norm ---
    let q_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_HEADS * w_sq,
    });
    let q_normed = rmsnorm(&mut g, q_for_norm, q_norm_w);

    // --- Q: transpose back for RoPE ---
    let q_norm_4d = g.reshape(
        q_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_HEADS,
            width: w_sq,
        },
    );
    let q_norm_t = g.transpose(q_norm_4d, [0, 2, 3, 1]); // [1, N_HEADS, w_sq, HEAD_DIM]

    // --- RoPE on Q ---
    let cos_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let sin_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let q_rope = apply_rope(&mut g, q_norm_t, cos_q, sin_q, N_HEADS, w_sq, HEAD_DIM);

    // --- K: input-pack projection ---
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);

    // --- K: reshape + transpose for per-head norm ---
    let k_4d = g.reshape(
        k_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]); // [1, HEAD_DIM, N_KV_HEADS, w_kv]
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );

    // --- Per-head K norm ---
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);

    // --- K: transpose back for RoPE ---
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]); // [1, N_KV_HEADS, w_kv, HEAD_DIM]

    // --- RoPE on K ---
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    // --- V: input-pack projection ---
    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let v_all = conv1x1_proj(&mut g, packed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);

    // --- V: reshape + transpose to 4D ---
    let v_4d = g.reshape(
        v_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let v_4d_t = g.transpose(v_4d, [0, 1, 3, 2]); // [1, N_KV_HEADS, w_kv, HEAD_DIM]

    // --- GQA tile ---
    let k_tiled = tile_kv_heads(&mut g, k_rope, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);
    let v_tiled = tile_kv_heads(&mut g, v_4d_t, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    // --- SDPA ---
    let scores = g.matrix_multiplication(q_rope, k_tiled, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let scores_scaled = g.multiplication(scores, scale);

    let attn_mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: w_kv,
    });
    let masked_scores = g.addition(scores_scaled, attn_mask);
    let attn_probs = g.soft_max(masked_scores, 3);
    let attn_out_4d = g.matrix_multiplication(attn_probs, v_tiled, false, false);

    // --- Flatten attn output ---
    let attn_t = g.transpose(attn_out_4d, [0, 1, 3, 2]); // [1, N_HEADS, HEAD_DIM, w_sq]
    let attn_flat = g.reshape(
        attn_t,
        Shape {
            batch: 1,
            channels: N_HEADS * HEAD_DIM,
            height: 1,
            width: w_sq,
        },
    );

    // --- o_proj + residual ---
    let wo = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS * HEAD_DIM,
        height: 1,
        width: HIDDEN,
    });
    let o_proj = conv1x1_proj(&mut g, attn_flat, wo, HIDDEN, N_HEADS * HEAD_DIM, w_sq);

    let _out = g.addition(hidden, o_proj);
    g
}

/// Input-pack K projection: concat(context, normed) before conv1x1.
/// Avoids the output concat that causes "Program Inference error" at runtime.
/// Tests whether packing inputs into [1, HIDDEN, 1, w_kv] and doing ONE
/// conv1x1 for K works at runtime (mathematically equivalent to separate
/// k_proj_ctx + k_proj_noise + concat).
pub fn build_input_pack_k_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let packed = g.concat(&[context, normed], 3);

    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let _out = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    g
}

/// Input-pack K + reshape + per-head k_norm + reshape back + rope_k.
/// The full K path using input packing instead of output concat.
pub fn build_input_pack_k_full_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let packed = g.concat(&[context, normed], 3);

    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);

    // Reshape + transpose for per-head k_norm
    let k_4d = g.reshape(
        k_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]); // [1, HEAD_DIM, N_KV_HEADS, w_kv]
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );

    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);

    // Transpose back: [1, HEAD_DIM, 1, N_KV_HEADS*w_kv] → [1, N_KV_HEADS, w_kv, HEAD_DIM]
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]); // [1, N_KV_HEADS, w_kv, HEAD_DIM]

    // RoPE
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let _out = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);
    g
}

/// Incremental test: K path (input-pack) + V projection only (no reshape).
/// Tests whether adding another conv1x1 on the same packed input causes issues.
pub fn build_k_plus_v_proj_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let packed = g.concat(&[context, normed], 3);

    // K path (full)
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);

    let k_4d = g.reshape(
        k_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]);
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );

    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);

    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]);

    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let _k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    // V projection (just the conv1x1, no reshape)
    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let _v_out = conv1x1_proj(&mut g, packed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);

    g
}

/// Test: K path (input-pack) + Q proj only (no norm, no rope).
/// Tests whether the fan-out of `normed` (used by both concat and Q conv) causes issues.
pub fn build_k_plus_q_proj_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let packed = g.concat(&[context, normed], 3);

    // K path (full)
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let k_4d = g.reshape(
        k_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]);
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]);
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let _k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    // Q projection only (no norm, no rope) - just to test fan-out
    let wq = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_HEADS * HEAD_DIM,
    });
    let _q_out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);

    g
}

/// Test: K path + Q proj with DOUBLE rmsnorm (two separate norm calls for the
/// fan-out test). If fan-out is the issue, this should work.
pub fn build_k_plus_q_doublenorm_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed1 = rmsnorm(&mut g, hidden, in_norm_w);
    let normed2 = rmsnorm(&mut g, hidden, in_norm_w); // duplicate norm

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let packed = g.concat(&[context, normed1], 3);

    // K path (full)
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let k_4d = g.reshape(
        k_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]);
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]);
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let _k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    // Q projection using normed2 (separate rmsnorm output)
    let wq = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_HEADS * HEAD_DIM,
    });
    let _q_out = conv1x1_proj(&mut g, normed2, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);

    g
}

/// K full + Q proj + V proj + V reshape+transpose + Q norm + Q rope.
/// Incrementally building up from k_plus_q_proj to find the breaking point.
pub fn build_kqv_plus_vnorm_qnorm_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let packed = g.concat(&[context, normed], 3);

    // K path (full)
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let k_4d = g.reshape(
        k_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]);
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]);
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let _k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    // V path (full)
    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let v_all = conv1x1_proj(&mut g, packed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let v_4d = g.reshape(
        v_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let _v_4d_t = g.transpose(v_4d, [0, 1, 3, 2]);

    // Q path (full)
    let wq = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_HEADS * HEAD_DIM,
    });
    let q_out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);
    let q_4d = g.reshape(
        q_out,
        Shape {
            batch: 1,
            channels: N_HEADS,
            height: HEAD_DIM,
            width: w_sq,
        },
    );
    let q_t = g.transpose(q_4d, [0, 2, 1, 3]);
    let q_for_norm = g.reshape(
        q_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_HEADS * w_sq,
        },
    );
    let q_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_HEADS * w_sq,
    });
    let q_normed = rmsnorm(&mut g, q_for_norm, q_norm_w);
    let q_norm_4d = g.reshape(
        q_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_HEADS,
            width: w_sq,
        },
    );
    let q_norm_t = g.transpose(q_norm_4d, [0, 2, 3, 1]);
    let cos_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let sin_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let _q_rope = apply_rope(&mut g, q_norm_t, cos_q, sin_q, N_HEADS, w_sq, HEAD_DIM);

    g
}

/// K+V+Q full paths + GQA tile + SDPA (no o_proj/residual).
/// Tests whether the attention computation causes issues.
pub fn build_kqv_plus_attn_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let packed = g.concat(&[context, normed], 3);

    // Q path
    let wq = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_HEADS * HEAD_DIM,
    });
    let q_out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);
    let q_4d = g.reshape(
        q_out,
        Shape {
            batch: 1,
            channels: N_HEADS,
            height: HEAD_DIM,
            width: w_sq,
        },
    );
    let q_t = g.transpose(q_4d, [0, 2, 1, 3]);
    let q_for_norm = g.reshape(
        q_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_HEADS * w_sq,
        },
    );
    let q_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_HEADS * w_sq,
    });
    let q_normed = rmsnorm(&mut g, q_for_norm, q_norm_w);
    let q_norm_4d = g.reshape(
        q_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_HEADS,
            width: w_sq,
        },
    );
    let q_norm_t = g.transpose(q_norm_4d, [0, 2, 3, 1]);
    let cos_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let sin_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let q_rope = apply_rope(&mut g, q_norm_t, cos_q, sin_q, N_HEADS, w_sq, HEAD_DIM);

    // K path
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let k_4d = g.reshape(
        k_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]);
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]);
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    // V path
    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let v_all = conv1x1_proj(&mut g, packed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let v_4d = g.reshape(
        v_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let v_4d_t = g.transpose(v_4d, [0, 1, 3, 2]);

    // GQA tile
    let k_tiled = tile_kv_heads(&mut g, k_rope, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);
    let v_tiled = tile_kv_heads(&mut g, v_4d_t, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    // SDPA
    let scores = g.matrix_multiplication(q_rope, k_tiled, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let scores_scaled = g.multiplication(scores, scale);
    let attn_mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: w_kv,
    });
    let masked_scores = g.addition(scores_scaled, attn_mask);
    let attn_probs = g.soft_max(masked_scores, 3);
    let _out = g.matrix_multiplication(attn_probs, v_tiled, false, false);

    g
}

/// QKV full paths + GQA tile only (no SDPA).
pub fn build_kqv_plus_gqa_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let packed = g.concat(&[context, normed], 3);

    let wq = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_HEADS * HEAD_DIM,
    });
    let q_out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);
    let q_4d = g.reshape(
        q_out,
        Shape {
            batch: 1,
            channels: N_HEADS,
            height: HEAD_DIM,
            width: w_sq,
        },
    );
    let q_t = g.transpose(q_4d, [0, 2, 1, 3]);
    let q_for_norm = g.reshape(
        q_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_HEADS * w_sq,
        },
    );
    let q_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_HEADS * w_sq,
    });
    let q_normed = rmsnorm(&mut g, q_for_norm, q_norm_w);
    let q_norm_4d = g.reshape(
        q_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_HEADS,
            width: w_sq,
        },
    );
    let q_norm_t = g.transpose(q_norm_4d, [0, 2, 3, 1]);
    let cos_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let sin_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let _q_rope = apply_rope(&mut g, q_norm_t, cos_q, sin_q, N_HEADS, w_sq, HEAD_DIM);

    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let k_4d = g.reshape(
        k_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]);
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]);
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let v_all = conv1x1_proj(&mut g, packed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let v_4d = g.reshape(
        v_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let v_4d_t = g.transpose(v_4d, [0, 1, 3, 2]);

    // Just GQA tile, no SDPA
    let _k_tiled = tile_kv_heads(&mut g, k_rope, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);
    let _v_tiled = tile_kv_heads(&mut g, v_4d_t, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    g
}

/// QKV full paths + just the scores computation (Q@K^T * scale)
/// Tests whether the first matmul in SDPA causes issues.
pub fn build_kqv_plus_scores_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let packed = g.concat(&[context, normed], 3);

    let wq = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_HEADS * HEAD_DIM,
    });
    let q_out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);
    let q_4d = g.reshape(
        q_out,
        Shape {
            batch: 1,
            channels: N_HEADS,
            height: HEAD_DIM,
            width: w_sq,
        },
    );
    let q_t = g.transpose(q_4d, [0, 2, 1, 3]);
    let q_for_norm = g.reshape(
        q_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_HEADS * w_sq,
        },
    );
    let q_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_HEADS * w_sq,
    });
    let q_normed = rmsnorm(&mut g, q_for_norm, q_norm_w);
    let q_norm_4d = g.reshape(
        q_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_HEADS,
            width: w_sq,
        },
    );
    let q_norm_t = g.transpose(q_norm_4d, [0, 2, 3, 1]);
    let cos_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let sin_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let q_rope = apply_rope(&mut g, q_norm_t, cos_q, sin_q, N_HEADS, w_sq, HEAD_DIM);

    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let k_4d = g.reshape(
        k_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]);
    let k_for_norm = g.reshape(
        k_t,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: 1,
            width: N_KV_HEADS * w_kv,
        },
    );
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);
    let k_norm_4d = g.reshape(
        k_normed,
        Shape {
            batch: 1,
            channels: HEAD_DIM,
            height: N_KV_HEADS,
            width: w_kv,
        },
    );
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]);
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let v_all = conv1x1_proj(&mut g, packed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let v_4d = g.reshape(
        v_all,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: HEAD_DIM,
            width: w_kv,
        },
    );
    let v_4d_t = g.transpose(v_4d, [0, 1, 3, 2]);

    let k_tiled = tile_kv_heads(&mut g, k_rope, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);
    let v_tiled = tile_kv_heads(&mut g, v_4d_t, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    // Just Q@K^T * scale, no softmax/matmul with V
    let scores = g.matrix_multiplication(q_rope, k_tiled, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let _out = g.multiplication(scores, scale);

    g
}

/// GQA tile + score matmul only (Q@K^T * scale).
/// Takes 4D Q, K as inputs. Tests whether GQA tile + first matmul works together.
pub fn build_gqa_plus_scores_kernel(w_sq: usize, w_kv: usize) -> Graph {
    let mut g = Graph::new();

    let k4 = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let k_t = tile_kv_heads(&mut g, k4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    let q = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_sq,
        width: HEAD_DIM,
    });

    let scores = g.matrix_multiplication(q, k_t, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let _out = g.multiplication(scores, scale);
    g
}

/// GQA tile only (no matmul). Takes 4D K, V as inputs.
pub fn build_gqa_tile_only_kernel(w_kv: usize) -> Graph {
    let mut g = Graph::new();

    let k4 = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let _k_t = tile_kv_heads(&mut g, k4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    let v4 = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let _v_t = tile_kv_heads(&mut g, v4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    g
}

/// SDPA without GQA tile: takes pre-tiled Q, K, V as inputs.
/// Q [1, N_HEADS, w_sq, HD], K [1, N_HEADS, w_kv, HD], V [1, N_HEADS, w_kv, HD]
pub fn build_sdpa_no_gqa_kernel(w_sq: usize, w_kv: usize) -> Graph {
    let mut g = Graph::new();

    let q = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_sq,
        width: HEAD_DIM,
    });
    let k = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let v = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });

    let scores = g.matrix_multiplication(q, k, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let scores_scaled = g.multiplication(scores, scale);
    let attn_mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: w_kv,
    });
    let masked_scores = g.addition(scores_scaled, attn_mask);
    let attn_probs = g.soft_max(masked_scores, 3);
    let _out = g.matrix_multiplication(attn_probs, v, false, false);
    g
}

/// SDPA + flatten + o_proj + residual (no GQA tile).
/// Takes pre-tiled Q, K, V + mask + o_proj_w + hidden.
pub fn build_sdpa_o_proj_kernel(w_sq: usize, w_kv: usize) -> Graph {
    let mut g = Graph::new();

    let q = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_sq,
        width: HEAD_DIM,
    });
    let k = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let v = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });

    let scores = g.matrix_multiplication(q, k, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let scores_scaled = g.multiplication(scores, scale);
    let attn_mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: w_kv,
    });
    let masked_scores = g.addition(scores_scaled, attn_mask);
    let attn_probs = g.soft_max(masked_scores, 3);
    let attn_out_4d = g.matrix_multiplication(attn_probs, v, false, false);

    let attn_t = g.transpose(attn_out_4d, [0, 1, 3, 2]);
    let attn_flat = g.reshape(
        attn_t,
        Shape {
            batch: 1,
            channels: N_HEADS * HEAD_DIM,
            height: 1,
            width: w_sq,
        },
    );

    let wo = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS * HEAD_DIM,
        height: 1,
        width: HIDDEN,
    });
    let o_proj = conv1x1_proj(&mut g, attn_flat, wo, HIDDEN, N_HEADS * HEAD_DIM, w_sq);

    let h_res = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let _out = g.addition(h_res, o_proj);
    g
}

pub fn build_fc_norm_kernel(w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let target_hid = g.placeholder(Shape {
        batch: 1,
        channels: TARGET_HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let fc_w = g.placeholder(Shape {
        batch: 1,
        channels: TARGET_HIDDEN,
        height: 1,
        width: HIDDEN,
    });
    let norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let fc_out = conv1x1_proj(&mut g, target_hid, fc_w, HIDDEN, TARGET_HIDDEN, w_ctx);
    let _out = rmsnorm(&mut g, fc_out, norm_w);
    g
}

pub fn build_q_proj_kernel(w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);
    let wq = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_HEADS * HEAD_DIM,
    });
    let _out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);
    g
}

/// Per-head rmsnorm for Q: input is [1, HEAD_DIM, 1, N_HEADS * w_sq].
/// Each spatial position is one (head, seq_pos) pair; rmsnorm reduces over
/// channels (HEAD_DIM) per position, giving the same result as per-head RMSNorm(head_dim).
pub fn build_q_norm_4d_kernel(w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let q_in = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_HEADS * w_sq,
    });
    let q_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_HEADS * w_sq,
    });
    let _out = rmsnorm(&mut g, q_in, q_norm_w);
    g
}

pub fn build_kv_concat_kernel(w_ctx: usize, w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let ctx = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS * HEAD_DIM,
        height: 1,
        width: w_ctx,
    });
    let noise = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS * HEAD_DIM,
        height: 1,
        width: w_sq,
    });
    let _out = g.concat(&[ctx, noise], 3);
    g
}

pub fn build_k_proj_ctx_kernel(w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let target_hid = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let _out = conv1x1_proj(&mut g, target_hid, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_ctx);
    g
}

pub fn build_k_proj_noise_kernel(w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let _out = conv1x1_proj(&mut g, normed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_sq);
    g
}

pub fn build_v_proj_ctx_kernel(w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let target_hid = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let _out = conv1x1_proj(&mut g, target_hid, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_ctx);
    g
}

pub fn build_v_proj_noise_kernel(w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let hidden = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let in_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);
    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let _out = conv1x1_proj(&mut g, normed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_sq);
    g
}

/// Per-head rmsnorm for K: input is [1, HEAD_DIM, 1, N_KV_HEADS * w_kv].
/// Same approach as q_norm_4d — each spatial position is one (kv_head, seq_pos) pair.
pub fn build_k_norm_4d_kernel(w_kv: usize) -> Graph {
    let mut g = Graph::new();
    let k_in = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HEAD_DIM,
        height: 1,
        width: N_KV_HEADS * w_kv,
    });
    let _out = rmsnorm(&mut g, k_in, k_norm_w);
    g
}

/// RoPE on Q: takes 4D input [1, N_HEADS, w_sq, HEAD_DIM], applies interleaved RoPE.
/// The flat→4D conversion is done in Python to avoid IOSurface reshape issues.
pub fn build_rope_q_kernel(w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let q4 = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_sq,
        width: HEAD_DIM,
    });
    let cos_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let sin_q = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: HEAD_DIM,
    });
    let _out = apply_rope(&mut g, q4, cos_q, sin_q, N_HEADS, w_sq, HEAD_DIM);
    g
}

/// RoPE on K: takes 4D input [1, N_KV_HEADS, w_kv, HEAD_DIM], applies interleaved RoPE.
/// Python does flat→4D conversion before calling, and 4D→flat after.
/// Caller pre-fills cos/sin with correct position IDs for ctx and noise segments.
pub fn build_rope_k_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let k4 = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let cos_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let sin_k = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_kv,
        width: HEAD_DIM,
    });
    let _out = apply_rope(&mut g, k4, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);
    g
}

/// GQA tile K and V: takes 4D K [1, N_KV_HEADS, w_kv, HEAD_DIM] and 4D V [1, N_KV_HEADS, w_kv, HEAD_DIM],
/// tiles KV heads, outputs concat [1, 2*N_HEADS, w_kv, HEAD_DIM].
pub fn build_gqa_tile_kernel(w_kv: usize) -> Graph {
    let mut g = Graph::new();

    let k4 = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let k_t = tile_kv_heads(&mut g, k4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    let v4 = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let v_t = tile_kv_heads(&mut g, v4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    let _out = g.concat(&[k_t, v_t], 1);
    g
}

/// SDPA with attention mask, logit softcapping, and flatten output:
/// Q, K_tiled, V_tiled, mask → flat attention output [1, NH*HD, 1, w_sq]
/// mask is [1, 1, w_sq, w_kv] with 0 for valid positions and -1e4 for masked (padded) positions.
/// Softcapping: cap * tanh(scores / cap) bounds attention scores to [-cap, +cap].
/// Output is flat [1, N_HEADS*HEAD_DIM, 1, w_sq] ready for o_proj_residual (no Python round-trip).
pub fn build_attn_out_kernel(w_sq: usize, w_kv: usize, softcap: f32) -> Graph {
    let mut g = Graph::new();

    let q = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS,
        height: w_sq,
        width: HEAD_DIM,
    });

    let kv = g.placeholder(Shape {
        batch: 1,
        channels: 2 * N_HEADS,
        height: w_kv,
        width: HEAD_DIM,
    });
    let k = g.slice(kv, [0, 0, 0, 0], [1, N_HEADS, w_kv, HEAD_DIM]);
    let v = g.slice(kv, [0, N_HEADS, 0, 0], [1, N_HEADS, w_kv, HEAD_DIM]);

    let scores = g.matrix_multiplication(q, k, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let scores_scaled = g.multiplication(scores, scale);

    let attn_mask = g.placeholder(Shape {
        batch: 1,
        channels: 1,
        height: w_sq,
        width: w_kv,
    });
    let masked_scores = g.addition(scores_scaled, attn_mask);

    let cap_val = g.constant_with_scalar(
        softcap,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let inv_cap = g.constant_with_scalar(
        1.0 / softcap,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let scores_div_cap = g.multiplication(masked_scores, inv_cap);
    let tanh_out = g.tanh(scores_div_cap);
    let softcapped_scores = g.multiplication(cap_val, tanh_out);

    let attn_probs = g.soft_max(softcapped_scores, 3);
    let attn_out_4d = g.matrix_multiplication(attn_probs, v, false, false);

    let attn_t = g.transpose(attn_out_4d, [0, 1, 3, 2]);
    let _out = g.reshape(
        attn_t,
        Shape {
            batch: 1,
            channels: N_HEADS * HEAD_DIM,
            height: 1,
            width: w_sq,
        },
    );
    g
}

/// o_proj + residual + softcapping: takes flat attn output [1, NH*HD, 1, w_sq], applies o_proj, adds residual,
/// then applies cap * tanh(output / cap) to bound the residual stream and prevent fp16 overflow.
pub fn build_o_proj_residual_kernel(w_sq: usize, softcap: f32) -> Graph {
    let mut g = Graph::new();
    let attn_flat = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS * HEAD_DIM,
        height: 1,
        width: w_sq,
    });
    let wo = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS * HEAD_DIM,
        height: 1,
        width: HIDDEN,
    });
    let o_proj = conv1x1_proj(&mut g, attn_flat, wo, HIDDEN, N_HEADS * HEAD_DIM, w_sq);

    let h_res = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let residual = g.addition(h_res, o_proj);

    let inv_cap = g.constant_with_scalar(
        1.0 / softcap,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let cap_val = g.constant_with_scalar(
        softcap,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let divided = g.multiplication(residual, inv_cap);
    let tanh_out = g.tanh(divided);
    let _out = g.multiplication(cap_val, tanh_out);
    g
}

pub fn build_ffn_residual_kernel(w_sq: usize, softcap: f32) -> Graph {
    let mut g = Graph::new();
    let h1 = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let post_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let normed = rmsnorm(&mut g, h1, post_norm_w);

    let w_gate = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: INTERMEDIATE,
    });
    let gate_out = conv1x1_proj(&mut g, normed, w_gate, INTERMEDIATE, HIDDEN, w_sq);
    let w_up = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: INTERMEDIATE,
    });
    let up_out = conv1x1_proj(&mut g, normed, w_up, INTERMEDIATE, HIDDEN, w_sq);

    let sig = g.sigmoid(gate_out);
    let silu = g.multiplication(gate_out, sig);
    let gate = g.multiplication(silu, up_out);

    let w_down = g.placeholder(Shape {
        batch: 1,
        channels: INTERMEDIATE,
        height: 1,
        width: HIDDEN,
    });
    let ffn_out = conv1x1_proj(&mut g, gate, w_down, HIDDEN, INTERMEDIATE, w_sq);

    let residual = g.addition(h1, ffn_out);

    let inv_cap = g.constant_with_scalar(
        1.0 / softcap,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let cap_val = g.constant_with_scalar(
        softcap,
        Shape {
            batch: 1,
            channels: 1,
            height: 1,
            width: 1,
        },
    );
    let divided = g.multiplication(residual, inv_cap);
    let tanh_out = g.tanh(divided);
    let _out = g.multiplication(cap_val, tanh_out);
    g
}

pub fn build_final_norm_kernel(w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let h = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let norm_w = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_sq,
    });
    let _out = rmsnorm(&mut g, h, norm_w);
    g
}
