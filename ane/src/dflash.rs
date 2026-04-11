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
    let ms = g.reduce_mean(x, 1);
    let diff = g.subtraction(x, ms);
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
    let normed = g.multiplication(x, inv_std);
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

/// SDPA only: Q, K, V → attention output (4D [1, NH, w_sq, HD])
pub fn build_attn_out_kernel(w_sq: usize, w_kv: usize) -> Graph {
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
    let attn_probs = g.soft_max(scores_scaled, 3);
    let _out = g.matrix_multiplication(attn_probs, v, false, false);
    g
}

/// o_proj + residual: takes flat attn output [1, NH*HD, 1, w_sq], applies o_proj, adds residual
pub fn build_o_proj_residual_kernel(w_sq: usize) -> Graph {
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
    let _out = g.addition(h_res, o_proj);
    g
}

pub fn build_ffn_residual_kernel(w_sq: usize) -> Graph {
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

    let _out = g.addition(h1, ffn_out);
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
