use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    sync::{Arc, Weak},
};

use anyhow::{bail, Result};
use derivative::Derivative;
use half::f16;
use itertools::Itertools;
use memmap2::Mmap;
use pollster::FutureExt;
use pyo3::{exceptions::PyValueError, prelude::*};
use safetensors::SafeTensors;
use tokio::sync::RwLock;
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption, InferOutput},
        loader::Loader,
        model::{
            Build, ContextAutoLimits, ModelBuilder, ModelInfo, ModelRuntime, ModelVersion, Quant,
            State as ModelState,
        },
        v4, v5, v6, JobRuntime, Submission,
    },
    tensor::{TensorCpu, TensorInit, TensorShape},
    wgpu,
};

pub mod info;

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
    info: Arc<ModelInfo>,

    #[derivative(Debug = "ignore")]
    state: Arc<dyn ModelState + Send + Sync>,
    backed: Arc<RwLock<CachedState>>,
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
    Arc<ModelInfo>,
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
            let info = Arc::new(builder.info());
            let state: Arc<dyn ModelState + Send + Sync> = Arc::new(builder.state());
            Ok((JobRuntime::new(builder).await, info, state))
        }
        ModelVersion::V5 => {
            let builder = ModelBuilder::new(&context, model).with_quant(quant);
            let builder = Build::<v5::ModelJobBuilder<f16>>::build(builder).await?;
            let info = Arc::new(builder.info());
            let state: Arc<dyn ModelState + Send + Sync> = Arc::new(builder.state());
            Ok((JobRuntime::new(builder).await, info, state))
        }
        ModelVersion::V6 => {
            let builder = ModelBuilder::new(&context, model).with_quant(quant);
            let builder = Build::<v6::ModelJobBuilder<f16>>::build(builder).await?;
            let info = Arc::new(builder.info());
            let state: Arc<dyn ModelState + Send + Sync> = Arc::new(builder.state());
            Ok((JobRuntime::new(builder).await, info, state))
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
        let (runtime, info, state) = handle.block_on().map_err(err)?.map_err(err)?;
        let backed = Arc::new(RwLock::new(CachedState::new()));
        Ok(Self {
            tokio,
            runtime,
            info,
            state,
            backed,
        })
    }

    pub fn info(&self) -> info::ModelInfo {
        self.info.as_ref().clone().into()
    }

    pub fn clone_state(&self, state: &State) -> PyResult<State> {
        let handle = self.tokio.spawn(clone_state(self.clone(), state.clone()));
        let state = handle.block_on().map_err(err)?.map_err(err)?;
        Ok(state)
    }

    #[pyo3(signature = (tokens, state=None, token_chunk_size=128))]
    pub fn run(
        &self,
        tokens: Vec<u16>,
        state: Option<State>,
        token_chunk_size: usize,
    ) -> PyResult<(Vec<f32>, State)> {
        let state = state.unwrap_or_else(|| {
            let tensor = Arc::new(RwLock::new(self.state.init()));
            let id = uid::Id::new();
            State { id, tensor }
        });

        let handle = self.tokio.spawn(run_internal(
            self.clone(),
            tokens,
            state.clone(),
            InferOption::Last,
            token_chunk_size,
        ));
        let output = handle.block_on().map_err(err)?.map_err(err)?;
        assert_eq!(output.shape()[1], 1);
        Ok((output.map(|x| x.to_f32()).to_vec(), state))
    }

    #[pyo3(signature = (tokens, state=None, token_chunk_size=128))]
    pub fn run_full(
        &self,
        tokens: Vec<u16>,
        state: Option<State>,
        token_chunk_size: usize,
    ) -> PyResult<(Vec<Vec<f32>>, State)> {
        let state = state.unwrap_or_else(|| {
            let tensor = Arc::new(RwLock::new(self.state.init()));
            let id = uid::Id::new();
            State { id, tensor }
        });

        let handle = self.tokio.spawn(run_internal(
            self.clone(),
            tokens,
            state.clone(),
            InferOption::Full,
            token_chunk_size,
        ));
        let output = handle.block_on().map_err(err)?.map_err(err)?;
        let output = output.map(|x| x.to_f32()).split(1).map_err(err)?;
        let output = output.into_iter().map(Vec::from).collect();
        Ok((output, state))
    }
}

async fn run_internal(
    model: Model,
    tokens: Vec<u16>,
    state: State,
    option: InferOption,
    token_chunk_size: usize,
) -> Result<TensorCpu<f16>> {
    if tokens.is_empty() {
        bail!("input tokens cannot be empty")
    }

    {
        let mut backed = model.backed.write().await;
        if backed.id != state.id {
            // we need to first back the old cached state
            if let Some(tensor) = backed.tensor.upgrade() {
                let mut tensor = tensor.write().await;
                *tensor = model.state.back(0).await?;
            }

            // then we load the new one
            *backed = state.clone().into();
            let tensor = state.tensor.read().await;
            model.state.load(0, tensor.clone())?;
        }
    }

    let mut inference = Some(InferInput::new(
        vec![InferInputBatch { tokens, option }],
        token_chunk_size,
    ));
    let mut data = vec![];
    let mut num_token = 0;
    loop {
        let input = inference.take().unwrap();
        if input.batches[0].tokens.is_empty() {
            break;
        }

        let (sender, receiver) = tokio::sync::oneshot::channel();
        let submission = Submission { input, sender };

        let _ = model.runtime.send(submission).await;
        let (input, output) = receiver.await?;
        inference = Some(input);

        num_token += output[0].0.shape()[1];
        let mut output = output[0].clone().to_vec();
        data.append(&mut output);
    }

    let num_vocab = model.info.num_vocab;
    let tensor = TensorCpu::from_data([num_vocab, num_token, 1, 1], data)?;
    Ok(tensor)
}

async fn clone_state(model: Model, state: State) -> Result<State> {
    let backed = model.backed.read().await;
    if backed.id != state.id {
        // data is already backed and stored in `state`
        let id = uid::Id::new();
        let tensor = state.tensor.clone();
        Ok(State { id, tensor })
    } else {
        // data is not in `state`, but still on the GPU
        let id = uid::Id::new();
        let tensor = model.state.back(0).await?;
        let tensor = Arc::new(RwLock::new(tensor));
        Ok(State { id, tensor })
    }
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct StateId;

#[pyclass]
#[derive(Debug, Clone)]
pub struct State {
    id: uid::Id<StateId>,
    tensor: Arc<RwLock<TensorCpu<f32>>>,
}

#[derive(Debug, Clone)]
struct CachedState {
    id: uid::Id<StateId>,
    tensor: Weak<RwLock<TensorCpu<f32>>>,
}

impl CachedState {
    pub fn new() -> Self {
        Self {
            id: uid::Id::new(),
            tensor: Weak::new(),
        }
    }
}

impl From<State> for CachedState {
    fn from(State { id, tensor }: State) -> Self {
        let tensor = Arc::downgrade(&tensor);
        Self { id, tensor }
    }
}

fn load_tokenizer(path: PathBuf) -> Result<web_rwkv::tokenizer::Tokenizer> {
    let file = File::open(path)?;
    let mut reader = BufReader::new(file);
    let mut contents = String::new();
    reader.read_to_string(&mut contents)?;
    Ok(web_rwkv::tokenizer::Tokenizer::new(&contents)?)
}

#[pyclass]
pub struct Tokenizer(web_rwkv::tokenizer::Tokenizer);

#[pymethods]
impl Tokenizer {
    #[new]
    pub fn new(path: PathBuf) -> PyResult<Self> {
        Ok(Self(load_tokenizer(path).map_err(err)?))
    }

    pub fn encode(&self, text: &str) -> PyResult<Vec<u16>> {
        self.0.encode(text.as_bytes()).map_err(err)
    }

    pub fn decode(&self, tokens: Vec<u16>) -> PyResult<Vec<u8>> {
        self.0.decode(&tokens).map_err(err)
    }
}

#[pymodule]
fn web_rwkv_py(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<Model>()?;
    module.add_class::<State>()?;
    module.add_class::<Tokenizer>()?;
    module.add_class::<info::ModelInfo>()?;
    module.add_class::<info::ModelVersion>()?;

    Ok(())
}
