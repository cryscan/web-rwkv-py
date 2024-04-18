use pyo3::prelude::*;

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ModelVersion {
    V4,
    V5,
    V6,
}

impl From<web_rwkv::runtime::model::ModelVersion> for ModelVersion {
    fn from(value: web_rwkv::runtime::model::ModelVersion) -> Self {
        match value {
            web_rwkv::runtime::model::ModelVersion::V4 => Self::V4,
            web_rwkv::runtime::model::ModelVersion::V5 => Self::V5,
            web_rwkv::runtime::model::ModelVersion::V6 => Self::V6,
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
    #[pyo3(get)]
    pub time_mix_adapter_size: usize,
    #[pyo3(get)]
    pub time_decay_adapter_size: usize,
}

impl From<web_rwkv::runtime::model::ModelInfo> for ModelInfo {
    fn from(value: web_rwkv::runtime::model::ModelInfo) -> Self {
        let web_rwkv::runtime::model::ModelInfo {
            version,
            num_layer,
            num_emb,
            num_hidden,
            num_vocab,
            num_head,
            time_mix_adapter_size,
            time_decay_adapter_size,
        } = value;
        let version = version.into();
        Self {
            version,
            num_layer,
            num_emb,
            num_hidden,
            num_vocab,
            num_head,
            time_mix_adapter_size,
            time_decay_adapter_size,
        }
    }
}
