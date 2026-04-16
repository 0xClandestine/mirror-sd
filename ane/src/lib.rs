use pyo3::prelude::*;

mod dflash;
mod wrapper;

#[pymodule]
fn mirror_sd_ane(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<wrapper::ANETensor>()?;
    m.add_class::<wrapper::ANEKernel>()?;
    m.add_class::<wrapper::Q8LayerWeights>()?;
    m.add_function(wrap_pyfunction!(wrapper::compile_dflash_kernels, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::compile_dflash_kernels_27b, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::compile_lm_head_kernel, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::compile_dflash_kernels_q8, m)?)?;
    Ok(())
}
