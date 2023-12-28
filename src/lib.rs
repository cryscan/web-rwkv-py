use std::{fs::File, path::PathBuf};

use anyhow::Result;
use memmap2::Mmap;
use half::f16;
use pyo3::{exceptions::PyValueError, prelude::*};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::loader::Loader,
    wgpu,
};
use web_rwkv::num::Scalar;

mod v5;

async fn create_context(max_buffer_size: u32) -> Result<Context> {
    let instance = Instance::new();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let limits = wgpu::Limits {
        max_storage_buffer_binding_size: max_buffer_size,
        ..Default::default()
    };
    let context = ContextBuilder::new(adapter)
        .with_default_pipelines()
        .with_limits(limits)
        .build()
        .await?;
    println!("{:#?}", context.adapter.get_info());
    Ok(context)
}

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelVersion {
    V4,
    V5,
    V6,
}

impl From<web_rwkv::model::ModelVersion> for ModelVersion {
    fn from(value: web_rwkv::model::ModelVersion) -> Self {
        match value {
            web_rwkv::model::ModelVersion::V4 => Self::V4,
            web_rwkv::model::ModelVersion::V5 => Self::V5,
            web_rwkv::model::ModelVersion::V6 => Self::V6,
        }
    }
}

#[pymethods]
impl ModelVersion {
    #[pyo3(name = "__str__")]
    pub fn str(&self) -> String {
        format!("{:?}", self)
    }
}

#[pyclass]
#[derive(Debug, Clone, Copy)]
pub struct ModelInfo {
    #[pyo3(get)]
    pub version: ModelVersion,
    #[pyo3(get)]
    pub num_layer: usize,
    #[pyo3(get)]
    pub num_emb: usize,
    #[pyo3(get)]
    pub num_hidden: usize,
    #[pyo3(get)]
    pub num_vocab: usize,
    #[pyo3(get)]
    pub num_head: usize,
}

impl From<web_rwkv::model::ModelInfo> for ModelInfo {
    fn from(value: web_rwkv::model::ModelInfo) -> Self {
        let web_rwkv::model::ModelInfo {
            version,
            num_layer,
            num_emb,
            num_hidden,
            num_vocab,
            num_head,
        } = value;
        Self {
            version: version.into(),
            num_layer,
            num_emb,
            num_hidden,
            num_vocab,
            num_head,
        }
    }
}

#[pymethods]
impl ModelInfo {
    #[pyo3(name = "__str__")]
    pub fn str(&self) -> String {
        format!("{:#?}", self)
    }

    /// Computes the required storage buffer size.
    pub fn max_buffer_size(&self) -> usize {
        (self.num_emb * self.num_hidden * f16::size()).max(128 << 20)
    }
}

#[pyfunction]
fn peek_info(file: PathBuf) -> PyResult<ModelInfo> {
    let info = || -> Result<ModelInfo> {
        let file = File::open(file)?;
        let map = unsafe { Mmap::map(&file)? };
        Ok(Loader::info(&map)?.into())
    };
    info().map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pymodule]
fn web_rwkv_py(py: Python, module: &PyModule) -> PyResult<()> {
    module.add_class::<ModelVersion>()?;
    module.add_class::<ModelInfo>()?;
    module.add_function(wrap_pyfunction!(peek_info, module)?)?;

    let module_v5 = PyModule::new(py, "v5")?;
    module_v5.add_class::<v5::Model>()?;
    module_v5.add_class::<v5::ModelState>()?;
    module_v5.add_class::<v5::BackedState>()?;
    module_v5.add_function(wrap_pyfunction!(v5::run_one, module_v5)?)?;
    module.add_submodule(module_v5)?;

    Ok(())
}
