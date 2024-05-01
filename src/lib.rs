use std::{
    collections::{HashMap, HashSet},
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    sync::Arc,
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
    backed: Arc<RwLock<StateCache>>,
}

#[derive(Debug, Default)]
struct StateCache {
    active: Option<StateId>,
    map: HashMap<StateId, TensorCpu<f32>>,
}

#[derive(Debug, Default, Clone, Copy, PartialEq, Eq, Hash)]
pub struct _StateId;

#[derive(Debug, Default, Clone, PartialEq, Eq, Hash)]
pub struct StateId(Arc<uid::Id<_StateId>>);

impl StateId {
    pub fn new() -> Self {
        Self(Arc::new(uid::Id::new()))
    }

    pub fn strong_count(&self) -> usize {
        Arc::strong_count(&self.0)
    }
}

impl std::ops::Deref for StateId {
    type Target = uid::Id<_StateId>;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct State {
    id: StateId,
    tensor: TensorCpu<f32>,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = Instance::new();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .auto_limits(info)
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
            let builder = ModelBuilder::new(&context, model).quant(quant);
            let model = Build::<v4::Model>::build(builder).await?;
            let info = Arc::new(model.info.clone());
            let builder = v4::ModelJobBuilder::new(model, 1);
            let state: Arc<dyn ModelState + Send + Sync> = Arc::new(builder.state());
            Ok((JobRuntime::new(builder).await, info, state))
        }
        ModelVersion::V5 => {
            let builder = ModelBuilder::new(&context, model).quant(quant);
            let model = Build::<v5::Model>::build(builder).await?;
            let info = Arc::new(model.info.clone());
            let builder = v5::ModelJobBuilder::new(model, 1);
            let state: Arc<dyn ModelState + Send + Sync> = Arc::new(builder.state());
            Ok((JobRuntime::new(builder).await, info, state))
        }
        ModelVersion::V6 => {
            let builder = ModelBuilder::new(&context, model).quant(quant);
            let model = Build::<v6::Model>::build(builder).await?;
            let info = Arc::new(model.info.clone());
            let builder = v6::ModelJobBuilder::new(model, 1);
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
        let backed = Arc::new(RwLock::new(StateCache::default()));
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

    pub fn init_state(&self) -> State {
        let id = StateId::new();
        let tensor = self.state.init();
        State { id, tensor }
    }

    pub fn clone_state(&self, state: &State) -> PyResult<State> {
        let model = self.clone();
        let state = state.clone();
        let handle = self
            .tokio
            .spawn(async move { model.back_state(state).await });
        let state = handle.block_on().map_err(err)?.map_err(err)?;
        Ok(state)
    }

    #[pyo3(signature = (tokens, state=None, token_chunk_size=128))]
    pub fn run(
        &self,
        tokens: Vec<u16>,
        state: Option<&State>,
        token_chunk_size: usize,
    ) -> PyResult<(Vec<f32>, State)> {
        let handle = {
            let model = self.clone();
            let state = state.cloned().unwrap_or_else(|| self.init_state());
            let option = InferOption::Last;
            self.tokio.spawn(async move {
                let output = model
                    .run_internal(tokens, state, option, token_chunk_size)
                    .await;
                model.clear_cache().await;
                output
            })
        };
        let (output, state) = handle.block_on().map_err(err)?.map_err(err)?;
        let output = output.map(|x| x.to_f32()).to_vec();
        Ok((output, state))
    }

    #[pyo3(signature = (tokens, state=None, token_chunk_size=128))]
    pub fn run_full(
        &self,
        tokens: Vec<u16>,
        state: Option<&State>,
        token_chunk_size: usize,
    ) -> PyResult<(Vec<f32>, State)> {
        let handle = {
            let model = self.clone();
            let state = state.cloned().unwrap_or_else(|| self.init_state());
            let option = InferOption::Full;
            self.tokio.spawn(async move {
                let output = model
                    .run_internal(tokens, state, option, token_chunk_size)
                    .await;
                model.clear_cache().await;
                output
            })
        };
        let (output, state) = handle.block_on().map_err(err)?.map_err(err)?;
        // let output = output.map(|x| x.to_f32()).split(1).map_err(err)?;
        // let output = output.into_iter().map(Vec::from).collect();
        let output = output.map(|x| x.to_f32()).to_vec();
        Ok((output, state))
    }
}

impl Model {
    async fn run_internal(
        &self,
        tokens: Vec<u16>,
        state: State,
        option: InferOption,
        token_chunk_size: usize,
    ) -> Result<(TensorCpu<f16>, State)> {
        if tokens.is_empty() {
            bail!("input tokens cannot be empty")
        }

        let mut backed = self.backed.write().await;
        let state = {
            if Some(state.id.clone()) != backed.active {
                // the input state is not the active state on the gpu
                // we need to back the active gpu state and put it in the cache
                let tensor = self.state.back(0).await?;
                backed.map.insert(state.id.clone(), tensor);

                // then load the input state
                let tensor = match backed.map.get(&state.id) {
                    Some(tensor) => tensor.clone(),
                    None => state.tensor.clone(),
                };
                self.state.load(0, tensor)?;
            }

            let state = State {
                id: StateId::new(),
                tensor: state.tensor,
            };
            backed.active = Some(state.id.clone());
            state
        };

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

            let _ = self.runtime.send(submission).await;
            let (input, output) = receiver.await?;
            inference = Some(input);

            num_token += output[0].0.shape()[1];
            let mut output = output[0].clone().to_vec();
            data.append(&mut output);
        }

        let num_vocab = self.info.num_vocab;
        let tensor = TensorCpu::from_data([num_vocab, num_token, 1, 1], data)?;
        Ok((tensor, state))
    }

    async fn back_state(&self, state: State) -> Result<State> {
        let backed = self.backed.read().await;
        let State { id, tensor } = state;
        let tensor = match (Some(id.clone()) == backed.active, backed.map.get(&id)) {
            (true, _) => self.state.back(0).await?,
            (false, Some(tensor)) => tensor.clone(),
            (false, None) => tensor,
        };
        Ok(State { id, tensor })
    }

    async fn clear_cache(&self) {
        let mut backed = self.backed.write().await;
        let retain: HashSet<_> = backed
            .map
            .keys()
            .filter(|k| k.strong_count() > 1)
            .cloned()
            .collect();
        backed.map.retain(|x, _| retain.contains(x));
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
