use std::ptr;
use std::time::Instant;

use objc2_io_surface::IOSurfaceLockOptions;
use pyo3::prelude::*;

use ane::{Executable, Graph, NSQualityOfService, Shape, TensorData};

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

    /// Write fp16 data directly into the ANE buffer (no f32 conversion).
    ///
    /// `buf` must be a Python buffer whose raw bytes are IEEE 754 fp16 values
    /// (2 bytes per element, little-endian). Equivalent to `write_buffer` but
    /// skips the f32→fp16 NEON conversion — use this when the source data is
    /// already fp16 (e.g. from `mx.float16` arrays).
    fn write_buffer_f16(&self, py: Python<'_>, buf: &Bound<'_, PyAny>) -> PyResult<()> {
        let bytes = buf.call_method0("tobytes")?;
        let raw: &[u8] = bytes.downcast::<pyo3::types::PyBytes>()?.as_bytes();
        let u16_count = raw.len() / 2;
        // Copy fp16 bits into a Vec before releasing the GIL so the Python
        // buffer cannot be mutated while we write to the IOSurface.
        let data: Vec<u16> = unsafe {
            std::slice::from_raw_parts(raw.as_ptr() as *const u16, u16_count).to_vec()
        };
        let surface = self.inner.surface();
        py.allow_threads(|| unsafe {
            surface.lockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
            let dst = std::slice::from_raw_parts_mut(
                surface.baseAddress().as_ptr().cast::<u16>(),
                u16_count,
            );
            dst.copy_from_slice(&data);
            surface.unlockWithOptions_seed(IOSurfaceLockOptions(0), ptr::null_mut());
        });
        Ok(())
    }

    /// Return argmax over the channels (vocab) dimension for each of the first `seq_len`
    /// spatial positions.  Expects buffer shape [1, vocab_size, 1, w_sq] (fp16 IOSurface).
    fn read_argmax(&self, py: Python<'_>, seq_len: usize, vocab_size: usize) -> PyResult<Vec<i32>> {
        let w_sq = self.inner.shape().width;
        let total = vocab_size * w_sq;
        let surface = self.inner.surface();
        let argmax = py.allow_threads(|| {
            let mut result = vec![0i32; seq_len];
            unsafe {
                surface.lockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
                let src = std::slice::from_raw_parts(
                    surface.baseAddress().as_ptr().cast::<u16>(),
                    total,
                );
                let mut f32_data = vec![0.0f32; total];
                ane::neon_convert::f16_to_f32_bulk(src, &mut f32_data);
                surface.unlockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
                // Layout: [1, vocab_size, 1, w_sq] → element[c, w] = f32_data[c * w_sq + w]
                for w in 0..seq_len {
                    let mut best_val = f32::NEG_INFINITY;
                    let mut best_idx = 0i32;
                    for c in 0..vocab_size {
                        let val = f32_data[c * w_sq + w];
                        if val > best_val {
                            best_val = val;
                            best_idx = c as i32;
                        }
                    }
                    result[w] = best_idx;
                }
            }
            result
        });
        Ok(argmax)
    }

    fn read_f32(&self, py: Python<'_>) -> PyResult<Vec<f32>> {
        let element_count = {
            let s = self.inner.shape();
            s.batch * s.channels * s.height * s.width
        };
        let mut result = vec![0.0f32; element_count];
        let surface = self.inner.surface();
        py.allow_threads(|| unsafe {
            surface.lockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
            let src = std::slice::from_raw_parts(
                surface.baseAddress().as_ptr().cast::<u16>(),
                element_count,
            );
            ane::neon_convert::f16_to_f32_bulk(src, &mut result);
            surface.unlockWithOptions_seed(IOSurfaceLockOptions::ReadOnly, ptr::null_mut());
        });
        Ok(result)
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

    fn run(
        &self,
        py: Python<'_>,
        inputs: Vec<PyRef<ANETensor>>,
        outputs: Vec<PyRef<ANETensor>>,
    ) -> PyResult<()> {
        let input_refs: Vec<&TensorData> = inputs.iter().map(|t| &t.inner).collect();
        let output_refs: Vec<&TensorData> = outputs.iter().map(|t| &t.inner).collect();
        py.allow_threads(|| self.executable.run_cached(&input_refs, &output_refs))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "ANE kernel '{}' run failed: {:?}",
                    self.name, e
                ))
            })?;
        Ok(())
    }

    fn run_uncached(
        &self,
        py: Python<'_>,
        inputs: Vec<PyRef<ANETensor>>,
        outputs: Vec<PyRef<ANETensor>>,
    ) -> PyResult<()> {
        let input_refs: Vec<&TensorData> = inputs.iter().map(|t| &t.inner).collect();
        let output_refs: Vec<&TensorData> = outputs.iter().map(|t| &t.inner).collect();
        py.allow_threads(|| self.executable.run(&input_refs, &output_refs))
            .map_err(|e| {
                pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "ANE kernel '{}' run_uncached failed: {:?}",
                    self.name, e
                ))
            })?;
        Ok(())
    }

    fn run_timed(
        &self,
        py: Python<'_>,
        inputs: Vec<PyRef<ANETensor>>,
        outputs: Vec<PyRef<ANETensor>>,
    ) -> PyResult<f64> {
        let input_refs: Vec<&TensorData> = inputs.iter().map(|t| &t.inner).collect();
        let output_refs: Vec<&TensorData> = outputs.iter().map(|t| &t.inner).collect();
        let start = Instant::now();
        py.allow_threads(|| self.executable.run_cached(&input_refs, &output_refs))
            .map_err(|e| {
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

fn compile_kernels(
    dims: &dflash::DFlashDims,
    seq_q: usize,
    ctx_len: usize,
    softcap: f32,
) -> PyResult<Vec<ANEKernel>> {
    let w_sq = dflash::align_width(seq_q);
    let w_ctx = dflash::align_width(ctx_len);
    let w_kv = w_ctx + w_sq;

    let kernel_builders: Vec<(&str, Graph)> = vec![
        ("fc_norm", dflash::build_fc_norm_kernel(dims, w_ctx)),
        ("mega_qkv", dflash::build_mega_qkv_kernel(dims, w_sq, w_ctx)),
        ("gqa_tile", dflash::build_gqa_tile_kernel(dims, w_kv)),
        ("attn_out", dflash::build_attn_out_kernel(dims, w_sq, w_kv, softcap)),
        ("o_proj_residual", dflash::build_o_proj_residual_kernel(dims, w_sq, softcap)),
        ("ffn_residual", dflash::build_ffn_residual_kernel(dims, w_sq, softcap)),
        ("final_norm", dflash::build_final_norm_kernel(dims, w_sq)),
    ];

    let mut compiled = Vec::new();
    for (name, graph) in kernel_builders {
        match graph.compile(NSQualityOfService::UserInteractive) {
            Ok(exec) => {
                compiled.push(ANEKernel { executable: exec, name: name.to_string() });
            }
            Err(e) => {
                return Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
                    "ANE compile '{}' failed: {:?}",
                    name, e
                )));
            }
        }
    }
    Ok(compiled)
}

#[pyfunction]
pub fn compile_dflash_kernels(
    seq_q: usize,
    ctx_len: usize,
    softcap: f32,
) -> PyResult<Vec<ANEKernel>> {
    compile_kernels(&dflash::DIMS_8B, seq_q, ctx_len, softcap)
}

#[pyfunction]
pub fn compile_dflash_kernels_27b(
    seq_q: usize,
    ctx_len: usize,
    softcap: f32,
) -> PyResult<Vec<ANEKernel>> {
    compile_kernels(&dflash::DIMS_27B, seq_q, ctx_len, softcap)
}

/// Compile the fused final-norm + lm_head kernel.
///
/// `is_27b`: True for 27B model (hidden=5120), False for 8B (hidden=4096).
#[pyfunction]
pub fn compile_lm_head_kernel(
    seq_q: usize,
    vocab_size: usize,
    is_27b: bool,
) -> PyResult<ANEKernel> {
    let dims = if is_27b { &dflash::DIMS_27B } else { &dflash::DIMS_8B };
    let w_sq = dflash::align_width(seq_q);
    let graph = dflash::build_final_norm_lm_head_kernel(dims, w_sq, vocab_size);
    match graph.compile(NSQualityOfService::UserInteractive) {
        Ok(exec) => Ok(ANEKernel { executable: exec, name: "final_norm_lm_head".to_string() }),
        Err(e) => Err(pyo3::exceptions::PyRuntimeError::new_err(format!(
            "ANE compile 'final_norm_lm_head' failed: {:?}",
            e
        ))),
    }
}
