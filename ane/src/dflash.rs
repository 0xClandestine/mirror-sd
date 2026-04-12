use ane::{Graph, MIN_SPATIAL_WIDTH, Shape, Tensor};

pub const HIDDEN: usize = 4096;
pub const HEAD_DIM: usize = 128;
pub const N_HEADS: usize = 32;
pub const N_KV_HEADS: usize = 8;
pub const INTERMEDIATE: usize = 12288;
pub const N_TARGET_FEATURES: usize = 5;
pub const TARGET_HIDDEN: usize = N_TARGET_FEATURES * HIDDEN;
const GQA_RATIO: usize = N_HEADS / N_KV_HEADS;

pub fn align_width(w: usize) -> usize {
    let aligned = ((w + MIN_SPATIAL_WIDTH - 1) / MIN_SPATIAL_WIDTH) * MIN_SPATIAL_WIDTH;
    aligned.max(MIN_SPATIAL_WIDTH)
}

pub fn rmsnorm(g: &mut Graph, x: Tensor, weight: Tensor) -> Tensor {
    let inv_s =
        g.constant_with_scalar(1.0 / 128.0, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let x_scaled = g.multiplication(x, inv_s);
    let ms = g.reduce_mean(x_scaled, 1);
    let diff = g.subtraction(x_scaled, ms);
    let sq = g.multiplication(diff, diff);
    let mean_sq = g.reduce_mean(sq, 1);
    let eps_t = g.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
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
    let w_conv = g.reshape(wt, Shape { batch: oc, channels: ic, height: 1, width: 1 });
    g.convolution_2d_1x1_dynamic(a, w_conv)
}

/// K full + Q proj + V proj + V reshape+transpose + Q norm + Q rope.
/// Incrementally building up from k_plus_q_proj to find the breaking point.
pub fn build_kqv_plus_vnorm_qnorm_kernel(w_sq: usize, w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let w_kv = w_ctx + w_sq;

    let hidden = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: w_sq });
    let in_norm_w = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: w_sq });
    let normed = rmsnorm(&mut g, hidden, in_norm_w);

    let context = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: w_ctx });
    let packed = g.concat(&[context, normed], 3);

    // K path (full)
    let wk = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let k_all = conv1x1_proj(&mut g, packed, wk, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let k_4d =
        g.reshape(k_all, Shape { batch: 1, channels: N_KV_HEADS, height: HEAD_DIM, width: w_kv });
    let k_t = g.transpose(k_4d, [0, 2, 1, 3]);
    let k_for_norm =
        g.reshape(k_t, Shape { batch: 1, channels: HEAD_DIM, height: 1, width: N_KV_HEADS * w_kv });
    let k_norm_w =
        g.placeholder(Shape { batch: 1, channels: HEAD_DIM, height: 1, width: N_KV_HEADS * w_kv });
    let k_normed = rmsnorm(&mut g, k_for_norm, k_norm_w);
    let k_norm_4d = g
        .reshape(k_normed, Shape { batch: 1, channels: HEAD_DIM, height: N_KV_HEADS, width: w_kv });
    let k_norm_t = g.transpose(k_norm_4d, [0, 2, 3, 1]);
    let cos_k = g.placeholder(Shape { batch: 1, channels: 1, height: w_kv, width: HEAD_DIM });
    let sin_k = g.placeholder(Shape { batch: 1, channels: 1, height: w_kv, width: HEAD_DIM });
    let _k_rope = apply_rope(&mut g, k_norm_t, cos_k, sin_k, N_KV_HEADS, w_kv, HEAD_DIM);

    // V path (full)
    let wv = g.placeholder(Shape {
        batch: 1,
        channels: HIDDEN,
        height: 1,
        width: N_KV_HEADS * HEAD_DIM,
    });
    let v_all = conv1x1_proj(&mut g, packed, wv, N_KV_HEADS * HEAD_DIM, HIDDEN, w_kv);
    let v_4d =
        g.reshape(v_all, Shape { batch: 1, channels: N_KV_HEADS, height: HEAD_DIM, width: w_kv });
    let _v_4d_t = g.transpose(v_4d, [0, 1, 3, 2]);

    // Q path (full)
    let wq =
        g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: N_HEADS * HEAD_DIM });
    let q_out = conv1x1_proj(&mut g, normed, wq, N_HEADS * HEAD_DIM, HIDDEN, w_sq);
    let q_4d =
        g.reshape(q_out, Shape { batch: 1, channels: N_HEADS, height: HEAD_DIM, width: w_sq });
    let q_t = g.transpose(q_4d, [0, 2, 1, 3]);
    let q_for_norm =
        g.reshape(q_t, Shape { batch: 1, channels: HEAD_DIM, height: 1, width: N_HEADS * w_sq });
    let q_norm_w =
        g.placeholder(Shape { batch: 1, channels: HEAD_DIM, height: 1, width: N_HEADS * w_sq });
    let q_normed = rmsnorm(&mut g, q_for_norm, q_norm_w);
    let q_norm_4d =
        g.reshape(q_normed, Shape { batch: 1, channels: HEAD_DIM, height: N_HEADS, width: w_sq });
    let q_norm_t = g.transpose(q_norm_4d, [0, 2, 3, 1]);
    let cos_q = g.placeholder(Shape { batch: 1, channels: 1, height: w_sq, width: HEAD_DIM });
    let sin_q = g.placeholder(Shape { batch: 1, channels: 1, height: w_sq, width: HEAD_DIM });
    let _q_rope = apply_rope(&mut g, q_norm_t, cos_q, sin_q, N_HEADS, w_sq, HEAD_DIM);

    g
}

pub fn build_fc_norm_kernel(w_ctx: usize) -> Graph {
    let mut g = Graph::new();
    let target_hid =
        g.placeholder(Shape { batch: 1, channels: TARGET_HIDDEN, height: 1, width: w_ctx });
    let fc_w = g.placeholder(Shape { batch: 1, channels: TARGET_HIDDEN, height: 1, width: HIDDEN });
    let norm_w = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: w_ctx });
    let fc_out = conv1x1_proj(&mut g, target_hid, fc_w, HIDDEN, TARGET_HIDDEN, w_ctx);
    let _out = rmsnorm(&mut g, fc_out, norm_w);
    g
}

/// GQA tile K and V: takes 4D K [1, N_KV_HEADS, w_kv, HEAD_DIM] and 4D V [1, N_KV_HEADS, w_kv, HEAD_DIM],
/// tiles KV heads, outputs concat [1, 2*N_HEADS, w_kv, HEAD_DIM].
pub fn build_gqa_tile_kernel(w_kv: usize) -> Graph {
    let mut g = Graph::new();

    let k4 = g.placeholder(Shape { batch: 1, channels: N_KV_HEADS, height: w_kv, width: HEAD_DIM });
    let k_t = tile_kv_heads(&mut g, k4, N_KV_HEADS, GQA_RATIO, w_kv, HEAD_DIM);

    let v4 = g.placeholder(Shape { batch: 1, channels: N_KV_HEADS, height: w_kv, width: HEAD_DIM });
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

    let q = g.placeholder(Shape { batch: 1, channels: N_HEADS, height: w_sq, width: HEAD_DIM });

    let kv =
        g.placeholder(Shape { batch: 1, channels: 2 * N_HEADS, height: w_kv, width: HEAD_DIM });
    let k = g.slice(kv, [0, 0, 0, 0], [1, N_HEADS, w_kv, HEAD_DIM]);
    let v = g.slice(kv, [0, N_HEADS, 0, 0], [1, N_HEADS, w_kv, HEAD_DIM]);

    let scores = g.matrix_multiplication(q, k, false, true);
    let scale = g.constant_with_scalar(
        1.0 / (HEAD_DIM as f32).sqrt(),
        Shape { batch: 1, channels: 1, height: 1, width: 1 },
    );
    let scores_scaled = g.multiplication(scores, scale);

    let attn_mask = g.placeholder(Shape { batch: 1, channels: 1, height: w_sq, width: w_kv });
    let masked_scores = g.addition(scores_scaled, attn_mask);

    let cap_val =
        g.constant_with_scalar(softcap, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let inv_cap =
        g.constant_with_scalar(1.0 / softcap, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let scores_div_cap = g.multiplication(masked_scores, inv_cap);
    let tanh_out = g.tanh(scores_div_cap);
    let softcapped_scores = g.multiplication(cap_val, tanh_out);

    let attn_probs = g.soft_max(softcapped_scores, 3);
    let attn_out_4d = g.matrix_multiplication(attn_probs, v, false, false);

    let attn_t = g.transpose(attn_out_4d, [0, 1, 3, 2]);
    let _out =
        g.reshape(attn_t, Shape { batch: 1, channels: N_HEADS * HEAD_DIM, height: 1, width: w_sq });
    g
}

/// o_proj + residual + softcapping: takes flat attn output [1, NH*HD, 1, w_sq], applies o_proj, adds residual,
/// then applies cap * tanh(output / cap) to bound the residual stream and prevent fp16 overflow.
pub fn build_o_proj_residual_kernel(w_sq: usize, softcap: f32) -> Graph {
    let mut g = Graph::new();
    let attn_flat =
        g.placeholder(Shape { batch: 1, channels: N_HEADS * HEAD_DIM, height: 1, width: w_sq });
    let wo =
        g.placeholder(Shape { batch: 1, channels: N_HEADS * HEAD_DIM, height: 1, width: HIDDEN });
    let o_proj = conv1x1_proj(&mut g, attn_flat, wo, HIDDEN, N_HEADS * HEAD_DIM, w_sq);

    let h_res = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: w_sq });
    let residual = g.addition(h_res, o_proj);

    let inv_cap =
        g.constant_with_scalar(1.0 / softcap, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let cap_val =
        g.constant_with_scalar(softcap, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let divided = g.multiplication(residual, inv_cap);
    let tanh_out = g.tanh(divided);
    let _out = g.multiplication(cap_val, tanh_out);
    g
}

pub fn build_ffn_residual_kernel(w_sq: usize, softcap: f32) -> Graph {
    let mut g = Graph::new();
    let h1 = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: w_sq });
    let post_norm_w = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: w_sq });
    let normed = rmsnorm(&mut g, h1, post_norm_w);

    let w_gate =
        g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: INTERMEDIATE });
    let gate_out = conv1x1_proj(&mut g, normed, w_gate, INTERMEDIATE, HIDDEN, w_sq);
    let w_up = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: INTERMEDIATE });
    let up_out = conv1x1_proj(&mut g, normed, w_up, INTERMEDIATE, HIDDEN, w_sq);

    let sig = g.sigmoid(gate_out);
    let silu = g.multiplication(gate_out, sig);
    let gate = g.multiplication(silu, up_out);

    let w_down =
        g.placeholder(Shape { batch: 1, channels: INTERMEDIATE, height: 1, width: HIDDEN });
    let ffn_out = conv1x1_proj(&mut g, gate, w_down, HIDDEN, INTERMEDIATE, w_sq);

    let residual = g.addition(h1, ffn_out);

    let inv_cap =
        g.constant_with_scalar(1.0 / softcap, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let cap_val =
        g.constant_with_scalar(softcap, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let divided = g.multiplication(residual, inv_cap);
    let tanh_out = g.tanh(divided);
    let _out = g.multiplication(cap_val, tanh_out);
    g
}

pub fn build_final_norm_kernel(w_sq: usize) -> Graph {
    let mut g = Graph::new();
    let h = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: w_sq });
    let norm_w = g.placeholder(Shape { batch: 1, channels: HIDDEN, height: 1, width: w_sq });
    let _out = rmsnorm(&mut g, h, norm_w);
    g
}
