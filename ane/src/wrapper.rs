use std::time::Instant;

use pyo3::prelude::*;

use ane::{Executable, Graph, NSQualityOfService, Shape, Tensor, TensorData};

use crate::dflash;

fn align_width(w: usize) -> usize {
    dflash::align_width(w)
}

#[pyclass]
pub struct ANETensor {
    pub inner: TensorData,
}

#[pymethods]
impl ANETensor {
    #[new]
    fn py_new(batch: usize, channels: usize, height: usize, width: usize) -> PyResult<Self> {
        let w = align_width(width);
        let shape = Shape { batch, channels, height, width: w };
        Ok(Self { inner: TensorData::new(shape) })
    }

    #[staticmethod]
    fn from_f32(
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        data: Vec<f32>,
    ) -> PyResult<Self> {
        let w = align_width(width);
        let shape = Shape { batch, channels, height, width: w };
        Ok(Self { inner: TensorData::with_f32(&data, shape) })
    }

    #[staticmethod]
    fn from_buffer(
        py: Python<'_>,
        batch: usize,
        channels: usize,
        height: usize,
        width: usize,
        buf: &Bound<'_, PyAny>,
    ) -> PyResult<Self> {
        let w = align_width(width);
        let shape = Shape { batch, channels, height, width: w };
        let bytes = buf.call_method0("tobytes")?;
        let raw: &[u8] = bytes.downcast::<pyo3::types::PyBytes>()?.as_bytes();
        let float_count = raw.len() / 4;
        let data: Vec<f32> =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, float_count).to_vec() };
        py.allow_threads(|| Ok(Self { inner: TensorData::with_f32(&data, shape) }))
    }

    fn write_buffer(&self, py: Python<'_>, buf: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = buf.call_method0("tobytes")?;
        let raw: &[u8] = bytes.downcast::<pyo3::types::PyBytes>()?.as_bytes();
        let float_count = raw.len() / 4;
        let data: Vec<f32> =
            unsafe { std::slice::from_raw_parts(raw.as_ptr() as *const f32, float_count).to_vec() };
        py.allow_threads(|| {
            self.inner.copy_from_f32(&data);
        });
        Ok(())
    }

    fn write_f32(&self, data: Vec<f32>) -> PyResult<()> {
        self.inner.copy_from_f32(&data);
        Ok(())
    }

    fn read_f32(&self) -> PyResult<Vec<f32>> {
        let slice = self.inner.as_f32_slice();
        Ok(slice.to_vec())
    }

    #[getter]
    fn shape(&self) -> (usize, usize, usize, usize) {
        let s = self.inner.shape();
        (s.batch, s.channels, s.height, s.width)
    }

    #[getter]
    fn size(&self) -> usize {
        let s = self.inner.shape();
        s.batch * s.channels * s.height * s.width
    }

    fn __repr__(&self) -> String {
        let s = self.inner.shape();
        format!(
            "ANETensor(batch={}, channels={}, height={}, width={})",
            s.batch, s.channels, s.height, s.width
        )
    }
}

#[pyclass]
pub struct ANEKernel {
    executable: Executable,
    name: String,
}

#[pymethods]
impl ANEKernel {
    #[getter]
    fn name(&self) -> &str {
        &self.name
    }

    fn run(&self, inputs: Vec<PyRef<ANETensor>>, outputs: Vec<PyRef<ANETensor>>) -> PyResult<()> {
        let input_refs: Vec<&TensorData> = inputs.iter().map(|t| &t.inner).collect();
        let output_refs: Vec<&TensorData> = outputs.iter().map(|t| &t.inner).collect();
        self.executable.run_cached(&input_refs, &output_refs).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "ANE kernel '{}' run failed: {:?}",
                self.name, e
            ))
        })?;
        Ok(())
    }

    fn run_uncached(
        &self,
        inputs: Vec<PyRef<ANETensor>>,
        outputs: Vec<PyRef<ANETensor>>,
    ) -> PyResult<()> {
        let input_refs: Vec<&TensorData> = inputs.iter().map(|t| &t.inner).collect();
        let output_refs: Vec<&TensorData> = outputs.iter().map(|t| &t.inner).collect();
        self.executable.run(&input_refs, &output_refs).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "ANE kernel '{}' run_uncached failed: {:?}",
                self.name, e
            ))
        })?;
        Ok(())
    }

    fn run_timed(
        &self,
        inputs: Vec<PyRef<ANETensor>>,
        outputs: Vec<PyRef<ANETensor>>,
    ) -> PyResult<f64> {
        let input_refs: Vec<&TensorData> = inputs.iter().map(|t| &t.inner).collect();
        let output_refs: Vec<&TensorData> = outputs.iter().map(|t| &t.inner).collect();
        let start = Instant::now();
        self.executable.run_cached(&input_refs, &output_refs).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "ANE kernel '{}' run failed: {:?}",
                self.name, e
            ))
        })?;
        Ok(start.elapsed().as_secs_f64())
    }

    fn __repr__(&self) -> String {
        format!("ANEKernel('{}')", self.name)
    }
}

#[pyfunction]
pub fn compile_dflash_kernels(
    seq_q: usize,
    ctx_len: usize,
    softcap: f32,
) -> PyResult<Vec<ANEKernel>> {
    let w_sq = dflash::align_width(seq_q);
    let w_ctx = dflash::align_width(ctx_len);
    let w_kv = w_ctx + w_sq;

    let kernel_builders: Vec<(&str, Graph)> = vec![
        ("fc_norm", dflash::build_fc_norm_kernel(w_ctx)),
        ("mega_qkv", dflash::build_kqv_plus_vnorm_qnorm_kernel(w_sq, w_ctx)),
        ("gqa_tile", dflash::build_gqa_tile_kernel(w_kv)),
        ("attn_out", dflash::build_attn_out_kernel(w_sq, w_kv, softcap)),
        ("o_proj_residual", dflash::build_o_proj_residual_kernel(w_sq, softcap)),
        ("ffn_residual", dflash::build_ffn_residual_kernel(w_sq, softcap)),
        ("final_norm", dflash::build_final_norm_kernel(w_sq)),
    ];

    let mut compiled = Vec::new();
    for (name, graph) in kernel_builders {
        let exec = graph.compile(NSQualityOfService::UserInteractive).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "ANE compile '{}' failed: {:?}",
                name, e
            ))
        })?;
        compiled.push(ANEKernel { executable: exec, name: name.to_string() });
    }
    Ok(compiled)
}

#[pyfunction]
pub fn test_input_pack_k(seq_q: usize, ctx_len: usize) -> PyResult<String> {
    let w_sq = dflash::align_width(seq_q);
    let w_ctx = dflash::align_width(ctx_len);

    let results = vec![
        (
            "input_pack_k",
            dflash::build_input_pack_k_kernel(w_sq, w_ctx)
                .compile(NSQualityOfService::UserInteractive),
        ),
        (
            "input_pack_k_full",
            dflash::build_input_pack_k_full_kernel(w_sq, w_ctx)
                .compile(NSQualityOfService::UserInteractive),
        ),
    ];

    let mut out = String::new();
    for (name, result) in results {
        match result {
            Ok(_) => out.push_str(&format!("  {}: OK\n", name)),
            Err(e) => out.push_str(&format!("  {}: FAILED {:?}\n", name, e)),
        }
    }
    Ok(out.trim_end().to_string())
}

#[pyfunction]
pub fn compile_incremental_test_kernels(seq_q: usize, ctx_len: usize) -> PyResult<Vec<ANEKernel>> {
    let w_sq = dflash::align_width(seq_q);
    let w_ctx = dflash::align_width(ctx_len);
    let w_kv = w_ctx + w_sq;

    let kernel_builders: Vec<(&str, Graph)> = vec![
        ("gqa_tile_only", dflash::build_gqa_tile_only_kernel(w_kv)),
        ("gqa_plus_scores", dflash::build_gqa_plus_scores_kernel(w_sq, w_kv)),
        ("sdpa_no_gqa", dflash::build_sdpa_no_gqa_kernel(w_sq, w_kv)),
        ("sdpa_o_proj", dflash::build_sdpa_o_proj_kernel(w_sq, w_kv)),
        ("fused_attn_out", dflash::build_fused_attn_out_kernel(w_sq, w_kv)),
    ];

    let mut compiled = Vec::new();
    for (name, graph) in kernel_builders {
        let exec = graph.compile(NSQualityOfService::UserInteractive).map_err(|e| {
            pyo3::exceptions::PyRuntimeError::new_err(format!(
                "ANE compile '{}' failed: {:?}",
                name, e
            ))
        })?;
        compiled.push(ANEKernel { executable: exec, name: name.to_string() });
    }
    Ok(compiled)
}

#[pyfunction]
pub fn test_rmsnorm(dim: usize, seq: usize) -> PyResult<String> {
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let ms = g.reduce_mean(x, 1);
    let diff = g.subtraction(x, ms);
    let sq = g.multiplication(diff, diff);
    let mean_sq = g.reduce_mean(sq, 1);
    let eps = g.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let meps = g.addition(mean_sq, eps);
    let neg_half =
        g.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let inv_std = g.power(meps, neg_half);
    let _out = g.multiplication(x, inv_std);

    match g.compile(NSQualityOfService::UserInteractive) {
        Ok(_) => Ok(format!("rmsnorm(dim={}, seq={}) compiled OK", dim, seq)),
        Err(e) => Ok(format!("rmsnorm(dim={}, seq={}) FAILED: {:?}", dim, seq, e)),
    }
}

#[pyfunction]
pub fn test_rmsnorm_matmul(dim: usize, oc: usize, seq: usize) -> PyResult<String> {
    let mut g = Graph::new();
    let x = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: seq });
    let ms = g.reduce_mean(x, 1);
    let diff = g.subtraction(x, ms);
    let sq = g.multiplication(diff, diff);
    let mean_sq = g.reduce_mean(sq, 1);
    let eps_t = g.constant_with_scalar(1e-6, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let meps = g.addition(mean_sq, eps_t);
    let neg_half =
        g.constant_with_scalar(-0.5, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let inv_std = g.power(meps, neg_half);
    let normed = g.multiplication(x, inv_std);

    let nr = g.reshape(normed, Shape { batch: 1, channels: 1, height: dim, width: seq });
    let nt = g.transpose(nr, [0, 1, 3, 2]);
    let w = g.placeholder(Shape { batch: 1, channels: dim, height: 1, width: oc });
    let wr = g.reshape(w, Shape { batch: 1, channels: 1, height: dim, width: oc });
    let _out = g.matrix_multiplication(nt, wr, false, false);

    match g.compile(NSQualityOfService::UserInteractive) {
        Ok(_) => Ok(format!("rmsnorm+matmul(dim={}, oc={}, seq={}) compiled OK", dim, oc, seq)),
        Err(e) => {
            Ok(format!("rmsnorm+matmul(dim={}, oc={}, seq={}) FAILED: {:?}", dim, oc, seq, e))
        }
    }
}

#[pyfunction]
pub fn test_matmul(ic: usize, oc: usize, seq: usize) -> PyResult<String> {
    let mut g = Graph::new();
    let acts = g.placeholder(Shape { batch: 1, channels: 1, height: seq, width: ic });
    let wts = g.placeholder(Shape { batch: 1, channels: 1, height: ic, width: oc });
    let _out = g.matrix_multiplication(acts, wts, false, false);

    match g.compile(NSQualityOfService::UserInteractive) {
        Ok(_) => Ok(format!("matmul(ic={}, oc={}, seq={}) compiled OK", ic, oc, seq)),
        Err(e) => Ok(format!("matmul(ic={}, oc={}, seq={}) FAILED: {:?}", ic, oc, seq, e)),
    }
}

#[pyfunction]
pub fn test_swiglu(dim: usize, ffn: usize, seq: usize) -> PyResult<String> {
    let mut g = Graph::new();
    let h1 = g.placeholder(Shape { batch: 1, channels: ffn, height: 1, width: seq });
    let h3 = g.placeholder(Shape { batch: 1, channels: ffn, height: 1, width: seq });
    let sig = g.sigmoid(h1);
    let silu = g.multiplication(h1, sig);
    let gate = g.multiplication(silu, h3);
    let _out = gate;

    match g.compile(NSQualityOfService::UserInteractive) {
        Ok(_) => Ok(format!("swiglu(dim={}, ffn={}, seq={}) compiled OK", dim, ffn, seq)),
        Err(e) => Ok(format!("swiglu(dim={}, ffn={}, seq={}) FAILED: {:?}", dim, ffn, seq, e)),
    }
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

#[pyfunction]
pub fn test_sdpa(
    n_heads: usize,
    head_dim: usize,
    n_kv_heads: usize,
    seq_q: usize,
    seq_k: usize,
) -> PyResult<String> {
    let mut g = Graph::new();
    let w_sq = align_width(seq_q);
    let w_sk = align_width(seq_k);
    let gqa_ratio = n_heads / n_kv_heads;

    let q = g.placeholder(Shape { batch: 1, channels: n_heads, height: head_dim, width: w_sq });
    let q_t = g.transpose(q, [0, 1, 3, 2]);

    let k = g.placeholder(Shape { batch: 1, channels: n_kv_heads, height: head_dim, width: w_sk });
    let k_base = g.transpose(k, [0, 1, 3, 2]);
    let k_t = tile_kv_heads(&mut g, k_base, n_kv_heads, gqa_ratio, seq_k, head_dim);

    let v = g.placeholder(Shape { batch: 1, channels: n_kv_heads, height: head_dim, width: w_sk });
    let v_base = g.transpose(v, [0, 1, 3, 2]);
    let v_t = tile_kv_heads(&mut g, v_base, n_kv_heads, gqa_ratio, seq_k, head_dim);

    let scores = g.matrix_multiplication(q_t, k_t, false, true);
    let scale_val = 1.0 / (head_dim as f32).sqrt();
    let scale =
        g.constant_with_scalar(scale_val, Shape { batch: 1, channels: 1, height: 1, width: 1 });
    let scores_scaled = g.multiplication(scores, scale);
    let attn_probs = g.soft_max(scores_scaled, 3);
    let _out = g.matrix_multiplication(attn_probs, v_t, false, false);

    match g.compile(NSQualityOfService::UserInteractive) {
        Ok(_) => Ok(format!(
            "sdpa(heads={}, head_dim={}, kv_heads={}, q={}, k={}) compiled OK",
            n_heads, head_dim, n_kv_heads, seq_q, seq_k
        )),
        Err(e) => Ok(format!(
            "sdpa(heads={}, head_dim={}, kv_heads={}, q={}, k={}) FAILED: {:?}",
            n_heads, head_dim, n_kv_heads, seq_q, seq_k, e
        )),
    }
}

#[pyfunction]
pub fn test_qkv_progressive(seq_q: usize, ctx_len: usize) -> PyResult<String> {
    use ane::MIN_SPATIAL_WIDTH;
    let w_sq = dflash::align_width(seq_q);
    let w_ctx = dflash::align_width(ctx_len);
    let mut results = Vec::new();

    // Level 1: rmsnorm + concat + 1 conv1x1
    {
        let mut g = Graph::new();
        let hidden =
            g.placeholder(Shape { batch: 1, channels: dflash::HIDDEN, height: 1, width: w_sq });
        let target_hid =
            g.placeholder(Shape { batch: 1, channels: dflash::HIDDEN, height: 1, width: w_ctx });
        let norm_w = g.placeholder(Shape {
            batch: 1,
            channels: dflash::HIDDEN,
            height: 1,
            width: MIN_SPATIAL_WIDTH,
        });
        let normed = dflash::rmsnorm(&mut g, hidden, norm_w);
        let packed = g.concat(&[target_hid, normed], 3);
        let wk = g.placeholder(Shape {
            batch: 1,
            channels: dflash::HIDDEN,
            height: 1,
            width: dflash::N_KV_HEADS * dflash::HEAD_DIM,
        });
        let wk_t = g.transpose(wk, [0, 3, 2, 1]);
        let wk_conv = g.reshape(
            wk_t,
            Shape {
                batch: dflash::N_KV_HEADS * dflash::HEAD_DIM,
                channels: dflash::HIDDEN,
                height: 1,
                width: 1,
            },
        );
        let _out = g.convolution_2d_1x1_dynamic(packed, wk_conv);
        results.push(("1_norm+concat+1conv", g.compile(NSQualityOfService::UserInteractive)));
    }

    // Level 2: + q conv + q_norm
    {
        let mut g = Graph::new();
        let hidden =
            g.placeholder(Shape { batch: 1, channels: dflash::HIDDEN, height: 1, width: w_sq });
        let target_hid =
            g.placeholder(Shape { batch: 1, channels: dflash::HIDDEN, height: 1, width: w_ctx });
        let norm_w = g.placeholder(Shape {
            batch: 1,
            channels: dflash::HIDDEN,
            height: 1,
            width: MIN_SPATIAL_WIDTH,
        });
        let normed = dflash::rmsnorm(&mut g, hidden, norm_w);
        let packed = g.concat(&[target_hid, normed], 3);
        // K conv
        let wk = g.placeholder(Shape {
            batch: 1,
            channels: dflash::HIDDEN,
            height: 1,
            width: dflash::N_KV_HEADS * dflash::HEAD_DIM,
        });
        let wk_t = g.transpose(wk, [0, 3, 2, 1]);
        let wk_conv = g.reshape(
            wk_t,
            Shape {
                batch: dflash::N_KV_HEADS * dflash::HEAD_DIM,
                channels: dflash::HIDDEN,
                height: 1,
                width: 1,
            },
        );
        let _k_out = g.convolution_2d_1x1_dynamic(packed, wk_conv);
        // Q conv
        let wq = g.placeholder(Shape {
            batch: 1,
            channels: dflash::HIDDEN,
            height: 1,
            width: dflash::N_HEADS * dflash::HEAD_DIM,
        });
        let wq_t = g.transpose(wq, [0, 3, 2, 1]);
        let wq_conv = g.reshape(
            wq_t,
            Shape {
                batch: dflash::N_HEADS * dflash::HEAD_DIM,
                channels: dflash::HIDDEN,
                height: 1,
                width: 1,
            },
        );
        let q_out = g.convolution_2d_1x1_dynamic(normed, wq_conv);
        // q_norm WITHOUT reshape - just rmsnorm on the raw conv1x1 output [1, N_HEADS*HEAD_DIM, 1, w_sq]
        let q_norm_w = g.placeholder(Shape {
            batch: 1,
            channels: dflash::N_HEADS * dflash::HEAD_DIM,
            height: 1,
            width: MIN_SPATIAL_WIDTH,
        });
        let _q_normed = dflash::rmsnorm(&mut g, q_out, q_norm_w);
        results.push(("2_norm+2conv+qnorm_flat", g.compile(NSQualityOfService::UserInteractive)));
    }

    // Level 3: norm + Qconv + q_norm (just one path)
    {
        let mut g = Graph::new();
        let hidden =
            g.placeholder(Shape { batch: 1, channels: dflash::HIDDEN, height: 1, width: w_sq });
        let norm_w = g.placeholder(Shape {
            batch: 1,
            channels: dflash::HIDDEN,
            height: 1,
            width: MIN_SPATIAL_WIDTH,
        });
        let normed = dflash::rmsnorm(&mut g, hidden, norm_w);
        let wq = g.placeholder(Shape {
            batch: 1,
            channels: dflash::HIDDEN,
            height: 1,
            width: dflash::N_HEADS * dflash::HEAD_DIM,
        });
        let wq_t = g.transpose(wq, [0, 3, 2, 1]);
        let wq_conv = g.reshape(
            wq_t,
            Shape {
                batch: dflash::N_HEADS * dflash::HEAD_DIM,
                channels: dflash::HIDDEN,
                height: 1,
                width: 1,
            },
        );
        let q_out = g.convolution_2d_1x1_dynamic(normed, wq_conv);
        let q_norm_w = g.placeholder(Shape {
            batch: 1,
            channels: dflash::N_HEADS * dflash::HEAD_DIM,
            height: 1,
            width: MIN_SPATIAL_WIDTH,
        });
        let _q_normed = dflash::rmsnorm(&mut g, q_out, q_norm_w);
        results.push(("3_norm+Qconv+qnorm", g.compile(NSQualityOfService::UserInteractive)));
    }

    // Level 4: K conv + k_norm only
    {
        let mut g = Graph::new();
        let target_hid =
            g.placeholder(Shape { batch: 1, channels: dflash::HIDDEN, height: 1, width: w_ctx });
        let normed =
            g.placeholder(Shape { batch: 1, channels: dflash::HIDDEN, height: 1, width: w_sq });
        let packed = g.concat(&[target_hid, normed], 3);
        let wk = g.placeholder(Shape {
            batch: 1,
            channels: dflash::HIDDEN,
            height: 1,
            width: dflash::N_KV_HEADS * dflash::HEAD_DIM,
        });
        let wk_t = g.transpose(wk, [0, 3, 2, 1]);
        let wk_conv = g.reshape(
            wk_t,
            Shape {
                batch: dflash::N_KV_HEADS * dflash::HEAD_DIM,
                channels: dflash::HIDDEN,
                height: 1,
                width: 1,
            },
        );
        let k_out = g.convolution_2d_1x1_dynamic(packed, wk_conv);
        let k_norm_w = g.placeholder(Shape {
            batch: 1,
            channels: dflash::N_KV_HEADS * dflash::HEAD_DIM,
            height: 1,
            width: MIN_SPATIAL_WIDTH,
        });
        let _k_normed = dflash::rmsnorm(&mut g, k_out, k_norm_w);
        results.push(("4_Kconv+knorm", g.compile(NSQualityOfService::UserInteractive)));
    }

    // Level 5: V conv only
    {
        let mut g = Graph::new();
        let target_hid =
            g.placeholder(Shape { batch: 1, channels: dflash::HIDDEN, height: 1, width: w_ctx });
        let normed =
            g.placeholder(Shape { batch: 1, channels: dflash::HIDDEN, height: 1, width: w_sq });
        let packed = g.concat(&[target_hid, normed], 3);
        let wv = g.placeholder(Shape {
            batch: 1,
            channels: dflash::HIDDEN,
            height: 1,
            width: dflash::N_KV_HEADS * dflash::HEAD_DIM,
        });
        let wv_t = g.transpose(wv, [0, 3, 2, 1]);
        let wv_conv = g.reshape(
            wv_t,
            Shape {
                batch: dflash::N_KV_HEADS * dflash::HEAD_DIM,
                channels: dflash::HIDDEN,
                height: 1,
                width: 1,
            },
        );
        let _v_out = g.convolution_2d_1x1_dynamic(packed, wv_conv);
        results.push(("5_Vconv", g.compile(NSQualityOfService::UserInteractive)));
    }

    let mut out = String::new();
    for (name, result) in results {
        match result {
            Ok(_) => out.push_str(&format!("  {}: OK\n", name)),
            Err(e) => out.push_str(&format!("  {}: FAILED {:?}\n", name, e)),
        }
    }
    Ok(out.trim_end().to_string())
}

/// Compile a test conv1x1 kernel using Approach A (current):
/// Weight placeholder [1, OC, 1, IC] with transpose+reshape
/// Input: [1, IC, 1, SEQ], Weight: [1, OC, 1, IC], Output: [1, OC, 1, SEQ]
#[pyfunction]
pub fn compile_conv1x1_transpose(ic: usize, oc: usize, seq: usize) -> PyResult<Vec<ANEKernel>> {
    let w = dflash::align_width(seq);
    let w_wt = dflash::align_width(ic);

    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: w });
    // Current approach: weight as [1, OC, 1, IC] with transpose
    let weight = g.placeholder(Shape { batch: 1, channels: oc, height: 1, width: w_wt });
    let wt = g.transpose(weight, [0, 3, 2, 1]);
    let w_conv = g.reshape(wt, Shape { batch: oc, channels: ic, height: 1, width: 1 });
    let _out = g.convolution_2d_1x1_dynamic(input, w_conv);

    let exec = g.compile(NSQualityOfService::UserInteractive).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "compile_conv1x1_transpose failed: {:?}",
            e
        ))
    })?;
    Ok(vec![ANEKernel { executable: exec, name: "conv1x1_transpose".to_string() }])
}

/// Compile a test conv1x1 kernel using Approach B (rustane reference):
/// Weight placeholder [1, IC, 1, OC] with concat+slice+transpose+reshape
/// Input: [1, IC, 1, SEQ], Weight: [1, IC, 1, OC], Output: [1, OC, 1, SEQ]
#[pyfunction]
pub fn compile_conv1x1_concat(ic: usize, oc: usize, seq: usize) -> PyResult<Vec<ANEKernel>> {
    let w = dflash::align_width(seq);
    let w_wt = dflash::align_width(oc);

    let mut g = Graph::new();
    let acts = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: w });
    let wts = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: w_wt });

    // Concat then slice (mirrors rustane build_conv_split pattern)
    let packed = g.concat(&[acts, wts], 3);
    let a = g.slice(packed, [0, 0, 0, 0], [1, ic, 1, seq]);
    let w_sliced = g.slice(packed, [0, 0, 0, seq], [1, ic, 1, oc]);

    let wt = g.transpose(w_sliced, [0, 3, 2, 1]);
    let w_conv = g.reshape(wt, Shape { batch: oc, channels: ic, height: 1, width: 1 });
    let _out = g.convolution_2d_1x1_dynamic(a, w_conv);

    let exec = g.compile(NSQualityOfService::UserInteractive).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("compile_conv1x1_concat failed: {:?}", e))
    })?;
    Ok(vec![ANEKernel { executable: exec, name: "conv1x1_concat".to_string() }])
}

/// Compile a test conv1x1 kernel using Approach C (transpose on placeholder, no concat-slice):
/// Weight placeholder [1, IC, 1, OC] with transposed data, direct transpose+reshape.
/// Tests whether transpose on placeholder works at all when using [1, IC, 1, OC] shape.
/// Input: [1, IC, 1, SEQ], Weight: [1, IC, 1, OC], Output: [1, OC, 1, SEQ]
#[pyfunction]
pub fn compile_conv1x1_transpose_b(ic: usize, oc: usize, seq: usize) -> PyResult<Vec<ANEKernel>> {
    let w = dflash::align_width(seq);
    let w_wt = dflash::align_width(oc);

    let mut g = Graph::new();
    let input = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: w });
    // Weight as [1, IC, 1, OC] (transposed shape), then transpose+reshape
    let weight = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: w_wt });
    let wt = g.transpose(weight, [0, 3, 2, 1]);
    let w_conv = g.reshape(wt, Shape { batch: oc, channels: ic, height: 1, width: 1 });
    let _out = g.convolution_2d_1x1_dynamic(input, w_conv);

    let exec = g.compile(NSQualityOfService::UserInteractive).map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!(
            "compile_conv1x1_transpose_b failed: {:?}",
            e
        ))
    })?;
    Ok(vec![ANEKernel { executable: exec, name: "conv1x1_transpose_b".to_string() }])
}

#[pyfunction]
pub fn test_conv1x1(ic: usize, oc: usize, seq: usize) -> PyResult<String> {
    let mut g = Graph::new();
    let acts = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: seq });
    let wts = g.placeholder(Shape { batch: 1, channels: ic, height: 1, width: oc });
    let packed = g.concat(&[acts, wts], 3);
    let a = g.slice(packed, [0, 0, 0, 0], [1, ic, 1, seq]);
    let w = g.slice(packed, [0, 0, 0, seq], [1, ic, 1, oc]);
    let wt = g.transpose(w, [0, 3, 2, 1]);
    let wc = g.reshape(wt, Shape { batch: oc, channels: ic, height: 1, width: 1 });
    let _out = g.convolution_2d_1x1_dynamic(a, wc);

    match g.compile(NSQualityOfService::UserInteractive) {
        Ok(_) => Ok(format!("conv1x1(ic={}, oc={}, seq={}) compiled OK", ic, oc, seq)),
        Err(e) => Ok(format!("conv1x1(ic={}, oc={}, seq={}) FAILED: {:?}", ic, oc, seq, e)),
    }
}

#[pyfunction]
pub fn test_dflash_nlayers(_n_layers: usize, seq_q: usize, ctx_len: usize) -> PyResult<String> {
    let w_sq = dflash::align_width(seq_q);
    let w_ctx = dflash::align_width(ctx_len);
    let w_kv = w_ctx + w_sq;

    let results = vec![
        (
            "fc_norm",
            dflash::build_fc_norm_kernel(w_ctx).compile(NSQualityOfService::UserInteractive),
        ),
        ("q_proj", dflash::build_q_proj_kernel(w_sq).compile(NSQualityOfService::UserInteractive)),
        (
            "k_proj_ctx",
            dflash::build_k_proj_ctx_kernel(w_ctx).compile(NSQualityOfService::UserInteractive),
        ),
        (
            "k_proj_noise",
            dflash::build_k_proj_noise_kernel(w_sq).compile(NSQualityOfService::UserInteractive),
        ),
        (
            "k_norm_4d",
            dflash::build_k_norm_4d_kernel(w_kv).compile(NSQualityOfService::UserInteractive),
        ),
        (
            "v_proj_ctx",
            dflash::build_v_proj_ctx_kernel(w_ctx).compile(NSQualityOfService::UserInteractive),
        ),
        (
            "v_proj_noise",
            dflash::build_v_proj_noise_kernel(w_sq).compile(NSQualityOfService::UserInteractive),
        ),
        ("rope_q", dflash::build_rope_q_kernel(w_sq).compile(NSQualityOfService::UserInteractive)),
        (
            "rope_k",
            dflash::build_rope_k_kernel(w_sq, w_ctx).compile(NSQualityOfService::UserInteractive),
        ),
        (
            "gqa_tile",
            dflash::build_gqa_tile_kernel(w_kv).compile(NSQualityOfService::UserInteractive),
        ),
        (
            "attn_out",
            dflash::build_attn_out_kernel(w_sq, w_kv, 50.0)
                .compile(NSQualityOfService::UserInteractive),
        ),
        (
            "o_proj_residual",
            dflash::build_o_proj_residual_kernel(w_sq, 30000.0)
                .compile(NSQualityOfService::UserInteractive),
        ),
        (
            "ffn_residual",
            dflash::build_ffn_residual_kernel(w_sq, 30000.0)
                .compile(NSQualityOfService::UserInteractive),
        ),
        (
            "final_norm",
            dflash::build_final_norm_kernel(w_sq).compile(NSQualityOfService::UserInteractive),
        ),
    ];

    let mut out = String::new();
    for (name, result) in results {
        match result {
            Ok(_) => out.push_str(&format!("  {}: OK\n", name)),
            Err(e) => out.push_str(&format!("  {}: FAILED {:?}\n", name, e)),
        }
    }
    Ok(out.trim_end().to_string())
}
