use std::time::Instant;

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
        ("mega_qkv", dflash::build_mega_qkv_kernel(w_sq, w_ctx)),
        ("gqa_tile", dflash::build_gqa_tile_kernel(w_kv)),
        ("attn_out", dflash::build_attn_out_kernel(w_sq, w_kv, softcap)),
        ("o_proj_residual", dflash::build_o_proj_residual_kernel(w_sq, softcap)),
        ("ffn_residual", dflash::build_ffn_residual_kernel(w_sq, softcap)),
        ("final_norm", dflash::build_final_norm_kernel(w_sq)),
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
