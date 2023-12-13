use pyo3::prelude::*;

mod v5;

/// Formats the sum of two numbers as string.
#[pyfunction]
fn sum_as_string(a: usize, b: usize) -> PyResult<String> {
    Ok((a + b).to_string())
}

/// A Python module implemented in Rust.
#[pymodule]
fn web_rwkv_py(py: Python, module: &PyModule) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(sum_as_string, module)?)?;

    let module_v5 = PyModule::new(py, "v5")?;
    module_v5.add_class::<v5::Model>()?;
    module_v5.add_class::<v5::ModelState>()?;
    module_v5.add_class::<v5::BackedState>()?;
    module.add_submodule(module_v5)?;

    Ok(())
}
