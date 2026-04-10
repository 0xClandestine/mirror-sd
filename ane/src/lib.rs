use pyo3::prelude::*;

mod dflash;
mod wrapper;

#[pymodule]
fn mirror_sd_ane(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<wrapper::ANETensor>()?;
    m.add_class::<wrapper::ANEKernel>()?;
    m.add_function(wrap_pyfunction!(wrapper::compile_dflash_kernels, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::test_conv1x1, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::test_rmsnorm, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::test_rmsnorm_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::test_matmul, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::test_swiglu, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::test_sdpa, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::test_dflash_nlayers, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::test_qkv_progressive, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::compile_conv1x1_transpose, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::compile_conv1x1_concat, m)?)?;
    m.add_function(wrap_pyfunction!(wrapper::compile_conv1x1_transpose_b, m)?)?;
    Ok(())
}
