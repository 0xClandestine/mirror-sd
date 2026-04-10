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

fn conv1x1_proj(g: &mut Graph, input: Tensor, weight: Tensor, oc: usize, ic: usize) -> Tensor {
    let wt = g.transpose(weight, [0, 3, 2, 1]);
    let w_conv = g.reshape(
        wt,
        Shape {
            batch: oc,
            channels: ic,
            height: 1,
            width: 1,
        },
    );
    g.convolution_2d_1x1_dynamic(input, w_conv)
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
    let fc_out = conv1x1_proj(&mut g, target_hid, fc_w, HIDDEN, TARGET_HIDDEN);
    let _out = rmsnorm(&mut g, fc_out, norm_w);
    g
}

pub fn build_q_kernel(w_sq: usize) -> Graph {
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
    let q_out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN);
    let q_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS * HEAD_DIM,
        height: 1,
        width: w_sq,
    });
    let _out = rmsnorm(&mut g, q_out, q_norm_w);
    g
}

pub fn build_k_proj_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let target_hid = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
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
        channels: N_KV_HEADS * HEAD_DIM,
        height: 1,
        width: HIDDEN,
    });
    let wk_t = g.transpose(wk, [0, 3, 2, 1]);
    let wk_conv = g.reshape(
        wk_t,
        Shape {
            batch: N_KV_HEADS * HEAD_DIM,
            channels: HIDDEN,
            height: 1,
            width: 1,
        },
    );
    let k_ctx = g.convolution_2d_1x1_dynamic(target_hid, wk_conv);
    let k_noise = g.convolution_2d_1x1_dynamic(normed, wk_conv);
    let _out = g.concat(&[k_ctx, k_noise], 3);
    g
}

pub fn build_k_norm_kernel(w_kv: usize) -> Graph {
    let mut g = Graph::new();
    let k_out = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS * HEAD_DIM,
        height: 1,
        width: w_kv,
    });
    let k_norm_w = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS * HEAD_DIM,
        height: 1,
        width: w_kv,
    });
    let _out = rmsnorm(&mut g, k_out, k_norm_w);
    g
}

pub fn build_v_proj_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let target_hid = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: w_ctx,
    });
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
    let wv_t = g.transpose(wv, [0, 3, 2, 1]);
    let wv_conv = g.reshape(
        wv_t,
        Shape {
            batch: N_KV_HEADS * HEAD_DIM,
            channels: HIDDEN,
            height: 1,
            width: 1,
        },
    );
    let v_ctx = g.convolution_2d_1x1_dynamic(target_hid, wv_conv);
    let v_noise = g.convolution_2d_1x1_dynamic(normed, wv_conv);
    let _out = g.concat(&[v_ctx, v_noise], 3);
    g
}

pub fn build_rope_q_kernel(w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let q_in = g.placeholder(Shape {
        batch: 1,
        channels: N_HEADS * HEAD_DIM,
        height: 1,
        width: w_sq,
    });
    let q4 = g.reshape(
        q_in,
        Shape {
            batch: 1,
            channels: N_HEADS,
            height: w_sq,
            width: HEAD_DIM,
        },
    );
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

/// RoPE on K: reshape, single apply_rope with precomputed cos/sin for full sequence.
/// Caller pre-fills cos/sin with correct position IDs for ctx and noise segments.
pub fn build_rope_k_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let k_in = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS * HEAD_DIM,
        height: 1,
        width: w_kv,
    });
    let k4 = g.reshape(
        k_in,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: w_kv,
            width: HEAD_DIM,
        },
    );
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
    let k_rope = apply_rope(&mut g, k4, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);
    let _out = g.reshape(
        k_rope,
        Shape {
            batch: 1,
            channels: N_KV_HEADS * HEAD_DIM,
            height: 1,
            width: w_kv,
        },
    );
    g
}

/// GQA tile K and V: reshape to [N_KV_HEADS, w_kv, HEAD_DIM], tile, concat
pub fn build_gqa_tile_kernel(w_kv: usize) -> Graph {
    let mut g = Graph::new();

    let k_in = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS * HEAD_DIM,
        height: 1,
        width: w_kv,
    });
    let k4 = g.reshape(
        k_in,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: w_kv,
            width: HEAD_DIM,
        },
    );
    let k_t = tile_kv_heads(&mut g, k4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    let v_in = g.placeholder(Shape {
        batch: 1,
        channels: N_KV_HEADS * HEAD_DIM,
        height: 1,
        width: w_kv,
    });
    let v4 = g.reshape(
        v_in,
        Shape {
            batch: 1,
            channels: N_KV_HEADS,
            height: w_kv,
            width: HEAD_DIM,
        },
    );
    let v_t = tile_kv_heads(&mut g, v4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    let _out = g.concat(&[k_t, v_t], 1);
    g
}

pub fn build_attn_residual_kernel(w_sq: usize, w_kv: usize) -> Graph {
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
    let attn_out = g.matrix_multiplication(attn_probs, v, false, false);

    let attn_flat = g.reshape(
        attn_out,
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
    let o_proj = conv1x1_proj(&mut g, attn_flat, wo, HIDDEN, N_HEADS * HEAD_DIM);

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
    let gate_out = conv1x1_proj(&mut g, normed, w_gate, INTERMEDIATE, HIDDEN);
    let w_up = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: INTERMEDIATE,
    });
    let up_out = conv1x1_proj(&mut g, normed, w_up, INTERMEDIATE, HIDDEN);

    let sig = g.sigmoid(gate_out);
    let silu = g.multiplication(gate_out, sig);
    let gate = g.multiplication(silu, up_out);

    let w_down = g.placeholder(Shape {
        batch: 1,
        channels: INTERMEDIATE,
        height: 1,
        width: HIDDEN,
    });
    let ffn_out = conv1x1_proj(&mut g, gate, w_down, HIDDEN, INTERMEDIATE);

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
