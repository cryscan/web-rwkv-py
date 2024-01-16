use std::{fs::File, path::PathBuf};

use anyhow::Result;
use memmap2::Mmap;
use pyo3::{exceptions::PyValueError, prelude::*};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::loader::Loader,
    wgpu,
};

mod v4;
mod v5;
mod v6;

async fn create_context(info: &web_rwkv::model::ModelInfo) -> Result<Context> {
    let instance = Instance::new();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .with_auto_limits(info)
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

    macro_rules! add_module {
        ($ver:ident) => {
            let submodule = PyModule::new(py, stringify!($ver))?;
            submodule.add_class::<$ver::Model>()?;
            submodule.add_class::<$ver::ModelState>()?;
            submodule.add_class::<$ver::BackedState>()?;
            submodule.add_function(wrap_pyfunction!($ver::run_one, submodule)?)?;
            module.add_submodule(submodule)?;
        };
    }

    add_module!(v4);
    add_module!(v5);
    add_module!(v6);

    Ok(())
}
