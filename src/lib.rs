use std::{
    collections::HashMap,
    fs::File,
    path::PathBuf,
    sync::{Arc, Mutex},
};

use anyhow::Result;
use derivative::Derivative;
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use pollster::FutureExt;
use pyo3::{exceptions::PyValueError, prelude::*};
use safetensors::SafeTensors;
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    runtime::{
        infer::{InferInput, InferOutput},
        loader::Loader,
        model::{
            Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelRuntime, ModelVersion, Quant,
            State as ModelState,
        },
        v4, v5, v6, JobRuntime,
    },
    tensor::TensorCpu,
    wgpu,
};

fn err(err: impl ToString) -> PyErr {
    PyValueError::new_err(err.to_string())
}

/// A model with runtime.
#[pyclass]
#[derive(Clone, Derivative)]
#[derivative(Debug)]
pub struct Model {
    tokio: Arc<tokio::runtime::Runtime>,
    runtime: JobRuntime<InferInput, InferOutput<f16>>,

    #[derivative(Debug = "ignore")]
    state: Arc<dyn ModelState + Send + Sync>,
    id: Arc<Mutex<uid::Id<StateId>>>,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = Instance::new();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .with_auto_limits(info)
        .build()
        .await?;
    Ok(context)
}

async fn load_runtime(
    path: PathBuf,
    quant: usize,
    quant_nf4: usize,
) -> Result<(
    JobRuntime<InferInput, InferOutput<f16>>,
    Arc<dyn ModelState + Send + Sync>,
)> {
    let file = File::open(path)?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;

    let context = create_context(&info).await?;
    let quant = (0..quant).map(|layer| (layer, Quant::Int8)).collect_vec();
    let quant_nf4 = (0..quant_nf4)
        .map(|layer| (layer, Quant::NF4))
        .collect_vec();
    let quant = quant.into_iter().chain(quant_nf4).collect();

    match info.version {
        ModelVersion::V4 => {
            let builder = ModelBuilder::new(&context, model).with_quant(quant);
            let builder = Build::<v4::ModelJobBuilder<f16>>::build(builder).await?;
            let state: Arc<dyn ModelState + Send + Sync> = Arc::new(builder.state());
            Ok((JobRuntime::new(builder).await, state))
        }
        ModelVersion::V5 => {
            let builder = ModelBuilder::new(&context, model).with_quant(quant);
            let builder = Build::<v5::ModelJobBuilder<f16>>::build(builder).await?;
            let state: Arc<dyn ModelState + Send + Sync> = Arc::new(builder.state());
            Ok((JobRuntime::new(builder).await, state))
        }
        ModelVersion::V6 => {
            let builder = ModelBuilder::new(&context, model).with_quant(quant);
            let builder = Build::<v6::ModelJobBuilder<f16>>::build(builder).await?;
            let state: Arc<dyn ModelState + Send + Sync> = Arc::new(builder.state());
            Ok((JobRuntime::new(builder).await, state))
        }
    }
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (path, quant=0, quant_nf4=0))]
    pub fn new(path: PathBuf, quant: usize, quant_nf4: usize) -> PyResult<Self> {
        let tokio = Arc::new(tokio::runtime::Runtime::new()?);
        let handle = tokio.spawn(load_runtime(path, quant, quant_nf4));
        let (runtime, state) = handle.block_on().map_err(err)?.map_err(err)?;
        let id = Arc::new(Mutex::new(uid::Id::new()));
        Ok(Self {
            tokio,
            runtime,
            state,
            id,
        })
    }

    #[pyo3(signature = (tokens, state=None, token_chunk_size=128))]
    pub fn run(
        &self,
        tokens: Vec<u16>,
        state: Option<State>,
        token_chunk_size: usize,
    ) -> (Vec<f32>, State) {
        let state = state.unwrap_or_else(|| {
            let backed = self.state.init();
            let id = uid::Id::new();
            State { id, backed }
        });
        todo!()
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateId;

#[pyclass]
#[derive(Debug, Clone)]
pub struct State {
    id: uid::Id<StateId>,
    backed: TensorCpu<f32>,
}

#[pymodule]
fn web_rwkv_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<State>()?;

    Ok(())
}
