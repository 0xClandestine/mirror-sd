use ane::{Graph, Shape, Tensor, MIN_SPATIAL_WIDTH};

pub struct DFlashDims {
    pub hidden: usize,
    pub head_dim: usize,
    pub n_heads: usize,
    pub n_kv_heads: usize,
    pub intermediate: usize,
    pub target_hidden: usize,
    pub gqa_ratio: usize,
}

pub const DIMS_8B: DFlashDims = DFlashDims {
    hidden: 4096,
    head_dim: 128,
    n_heads: 32,
    n_kv_heads: 8,
    intermediate: 12288,
    target_hidden: 5 * 4096,
    gqa_ratio: 4,
};

pub const DIMS_27B: DFlashDims = DFlashDims {
    hidden: 5120,
    head_dim: 128,
    n_heads: 32,
    n_kv_heads: 8,
    intermediate: 17408,
    target_hidden: 5 * 5120,
    gqa_ratio: 4,
};

pub fn align_width(w: usize) -> usize {
    let aligned = ((w + MIN_SPATIAL_WIDTH - 1) / MIN_SPATIAL_WIDTH) * MIN_SPATIAL_WIDTH;
    aligned.max(MIN_SPATIAL_WIDTH)
}

pub fn rmsnorm(g: &mut Graph, x: Tensor, weight: Tensor) -> Tensor {
    rmsnorm_with_eps(g, x, weight, 1e-6)
}

pub fn rmsnorm_with_eps(g: &mut Graph, x: Tensor, weight: Tensor, eps: f32) -> Tensor {
    let inv_s =
        g.constant_with_scalar(1.0 / 128.0, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let x_scaled = g.multiplication(x, inv_s);
    let sq = g.multiplication(x_scaled, x_scaled);
    let mean_sq = g.reduce_mean(sq, 1);
    let eps_t = g.constant_with_scalar(eps, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let mean_sq_eps = g.addition(mean_sq, eps_t);
    let neg_half =
        g.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let inv_std = g.power(mean_sq_eps, neg_half);
    let normed = g.multiplication(x_scaled, inv_std);
    g.multiplication(normed, weight)
}

fn rmsnorm_per_head(g: &mut Graph, x: Tensor, weight: Tensor, eps: f32) -> Tensor {
    let inv_s =
        g.constant_with_scalar(1.0 / 128.0, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let x_scaled = g.multiplication(x, inv_s);
    let sq = g.multiplication(x_scaled, x_scaled);
    let mean_sq = g.reduce_mean(sq, 2);
    let eps_t = g.constant_with_scalar(eps, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let mean_sq_eps = g.addition(mean_sq, eps_t);
    let neg_half =
        g.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });
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
    let xp = g.reshape(x, Shape { batch: 1, channels: n_heads, height: pairs, width: 2 });
    let x_e = g.slice(xp, [0, 0, 0, 0], [1, n_heads, pairs, 1]);
    let x_o = g.slice(xp, [0, 0, 0, 1], [1, n_heads, pairs, 1]);
    let neg1 = g.constant_with_scalar(-1.0, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let neg_xo = g.multiplication(x_o, neg1);
    let rotated = g.concat(&[neg_xo, x_e], 3);
    let x_rot = g.reshape(rotated, Shape { batch: 1, channels: n_heads, height: seq, width: hd });
    let xc = g.multiplication(x, cos);
    let xs = g.multiplication(x_rot, sin);
    g.addition(xc, xs)
}

fn conv1x1_proj(g: &mut Graph, input: Tensor, weight: Tensor, oc: usize, ic: usize) -> Tensor {
    let wt = g.transpose(weight, [0, 3, 2, 1]);
    let w_conv = g.reshape(wt, Shape { batch: oc, channels: ic, height: 1, width: 1 });
    g.convolution_2d_1x1_dynamic(input, w_conv)
}

pub fn build_fc_norm_kernel(d: &DFlashDims, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let target_hid =
        g.placeholder(Shape { batch: 1, channels: d.target_hidden, height: 1, width: w_ctx });
    let fc_w =
        g.placeholder(Shape { batch: 1, channels: d.target_hidden, height: 1, width: d.hidden });
    let norm_w = g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: w_ctx });
    let fc_out = conv1x1_proj(&mut g, target_hid, fc_w, d.hidden, d.target_hidden);
    let _out = rmsnorm_with_eps(&mut g, fc_out, norm_w, 1e-6);
    g
}

pub fn build_mega_qkv_kernel(d: &DFlashDims, w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: w_sq });
    let in_norm_w = g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: w_sq });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: w_ctx });
    let packed = g.concat(&[context, normed], 3);

    let wk = g.placeholder(Shape {
        batch: 1,
        channels: d.hidden,
        height: 1,
        width: d.n_kv_heads * d.head_dim,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, d.n_kv_heads * d.head_dim, d.hidden);
    let k_4d = g.reshape(
        k_all,
        Shape { batch: 1, channels: d.n_kv_heads, height: d.head_dim, width: w_kv },
    );
    let k_norm_w =
        g.placeholder(Shape { batch: 1, channels: d.n_kv_heads, height: d.head_dim, width: w_kv });
    let k_normed = rmsnorm_per_head(&mut g, k_4d, k_norm_w, 1e-6);
    let k_norm_t = g.transpose(k_normed, [0, 1, 3, 2]);
    let cos_k = g.placeholder(Shape { batch: 1, channels: 1, height: w_kv, width: d.head_dim });
    let sin_k = g.placeholder(Shape { batch: 1, channels: 1, height: w_kv, width: d.head_dim });
    let _k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, d.n_kv_heads, w_kv, d.head_dim);

    let wv = g.placeholder(Shape {
        batch: 1,
        channels: d.hidden,
        height: 1,
        width: d.n_kv_heads * d.head_dim,
    });
    let v_all = conv1x1_proj(&mut g, packed, wv, d.n_kv_heads * d.head_dim, d.hidden);
    let v_4d = g.reshape(
        v_all,
        Shape { batch: 1, channels: d.n_kv_heads, height: d.head_dim, width: w_kv },
    );
    let _v_4d_t = g.transpose(v_4d, [0, 1, 3, 2]);

    let wq = g.placeholder(Shape {
        batch: 1,
        channels: d.hidden,
        height: 1,
        width: d.n_heads * d.head_dim,
    });
    let q_out = conv1x1_proj(&mut g, normed, wq, d.n_heads * d.head_dim, d.hidden);
    let q_4d =
        g.reshape(q_out, Shape { batch: 1, channels: d.n_heads, height: d.head_dim, width: w_sq });
    let q_norm_w =
        g.placeholder(Shape { batch: 1, channels: d.n_heads, height: d.head_dim, width: w_sq });
    let q_normed = rmsnorm_per_head(&mut g, q_4d, q_norm_w, 1e-6);
    let q_norm_t = g.transpose(q_normed, [0, 1, 3, 2]);
    let cos_q = g.placeholder(Shape { batch: 1, channels: 1, height: w_sq, width: d.head_dim });
    let sin_q = g.placeholder(Shape { batch: 1, channels: 1, height: w_sq, width: d.head_dim });
    let _q_rope = apply_rope(&mut g, q_norm_t, cos_q, sin_q, d.n_heads, w_sq, d.head_dim);

    g
}

pub fn build_gqa_tile_kernel(d: &DFlashDims, w_kv: usize) -> Graph {
    let mut g = Graph::new();
    let k4 =
        g.placeholder(Shape { batch: 1, channels: d.n_kv_heads, height: w_kv, width: d.head_dim });
    let k_t = tile_kv_heads(&mut g, k4, d.n_kv_heads, d.gqa_ratio, w_kv, d.head_dim);
    let v4 =
        g.placeholder(Shape { batch: 1, channels: d.n_kv_heads, height: w_kv, width: d.head_dim });
    let v_t = tile_kv_heads(&mut g, v4, d.n_kv_heads, d.gqa_ratio, w_kv, d.head_dim);
    let _out = g.concat(&[k_t, v_t], 1);
    g
}

pub fn build_attn_out_kernel(d: &DFlashDims, w_sq: usize, w_kv: usize, softcap: f32) -> Graph {
    let mut g = Graph::new();
    let q = g.placeholder(Shape { batch: 1, channels: d.n_heads, height: w_sq, width: d.head_dim });
    let kv =
        g.placeholder(Shape { batch: 1, channels: 2 * d.n_heads, height: w_kv, width: d.head_dim });
    let k = g.slice(kv, [0, 0, 0, 0], [1, d.n_heads, w_kv, d.head_dim]);
    let v = g.slice(kv, [0, d.n_heads, 0, 0], [1, d.n_heads, w_kv, d.head_dim]);

    let scores = g.matrix_multiplication(q, k, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (d.head_dim as f32).sqrt(),
        Shape { batch: 1, channels: 1, height: 1, width: 1 },
    );
    let scores_scaled = g.multiplication(scores, scale);

    let attn_mask = g.placeholder(Shape { batch: 1, channels: 1, height: w_sq, width: w_kv });
    let masked_scores = g.addition(scores_scaled, attn_mask);

    let softcapped_scores = if softcap > 0.0 {
        let cap_val =
            g.constant_with_scalar(softcap, Shape { batch: 1, channels: 1, height: 1, width: 1 });
        let inv_cap = g.constant_with_scalar(
            1.0 / softcap,
            Shape { batch: 1, channels: 1, height: 1, width: 1 },
        );
        let scores_div_cap = g.multiplication(masked_scores, inv_cap);
        let tanh_out = g.tanh(scores_div_cap);
        g.multiplication(cap_val, tanh_out)
    } else {
        masked_scores
    };

    let attn_probs = g.soft_max(softcapped_scores, 3);
    let attn_out_4d = g.matrix_multiplication(attn_probs, v, false, false);
    let attn_t = g.transpose(attn_out_4d, [0, 1, 3, 2]);
    let _out = g.reshape(
        attn_t,
        Shape { batch: 1, channels: d.n_heads * d.head_dim, height: 1, width: w_sq },
    );
    g
}

pub fn build_o_proj_residual_kernel(d: &DFlashDims, w_sq: usize, softcap: f32) -> Graph {
    let mut g = Graph::new();
    let attn_flat =
        g.placeholder(Shape { batch: 1, channels: d.n_heads * d.head_dim, height: 1, width: w_sq });
    let wo = g.placeholder(Shape {
        batch: 1,
        channels: d.n_heads * d.head_dim,
        height: 1,
        width: d.hidden,
    });
    let o_proj = conv1x1_proj(&mut g, attn_flat, wo, d.hidden, d.n_heads * d.head_dim);
    let h_in = g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: w_sq });
    let residual = g.addition(h_in, o_proj);
    if softcap > 0.0 {
        let inv_cap = g.constant_with_scalar(
            1.0 / softcap,
            Shape { batch: 1, channels: 1, height: 1, width: 1 },
        );
        let cap_val =
            g.constant_with_scalar(softcap, Shape { batch: 1, channels: 1, height: 1, width: 1 });
        let divided = g.multiplication(residual, inv_cap);
        let tanh_out = g.tanh(divided);
        let _out = g.multiplication(cap_val, tanh_out);
    }
    g
}

pub fn build_ffn_residual_kernel(d: &DFlashDims, w_sq: usize, softcap: f32) -> Graph {
    let mut g = Graph::new();
    let h1 = g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: w_sq });
    let post_norm_w = g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: w_sq });
    let normed = rmsnorm(&mut g, h1, post_norm_w);

    let w_gate =
        g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: d.intermediate });
    let gate_out = conv1x1_proj(&mut g, normed, w_gate, d.intermediate, d.hidden);
    let w_up =
        g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: d.intermediate });
    let up_out = conv1x1_proj(&mut g, normed, w_up, d.intermediate, d.hidden);
    let sig = g.sigmoid(gate_out);
    let silu = g.multiplication(gate_out, sig);
    let gate = g.multiplication(silu, up_out);
    let w_down =
        g.placeholder(Shape { batch: 1, channels: d.intermediate, height: 1, width: d.hidden });
    let ffn_out = conv1x1_proj(&mut g, gate, w_down, d.hidden, d.intermediate);

    let residual = g.addition(h1, ffn_out);
    if softcap > 0.0 {
        let inv_cap = g.constant_with_scalar(
            1.0 / softcap,
            Shape { batch: 1, channels: 1, height: 1, width: 1 },
        );
        let cap_val =
            g.constant_with_scalar(softcap, Shape { batch: 1, channels: 1, height: 1, width: 1 });
        let divided = g.multiplication(residual, inv_cap);
        let tanh_out = g.tanh(divided);
        let _out = g.multiplication(cap_val, tanh_out);
    }
    g
}


pub fn build_final_norm_kernel(d: &DFlashDims, w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let h = g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: w_sq });
    let norm_w = g.placeholder(Shape { batch: 1, channels: d.hidden, height: 1, width: w_sq });
    let _out = rmsnorm(&mut g, h, norm_w);
    g
}
