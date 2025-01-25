use std::{
    fs::File,
    io::{BufReader, Read},
    path::PathBuf,
    sync::Arc,
};

use anyhow::{bail, Result};
use derivative::Derivative;
use half::f16;
use memmap2::Mmap;
use pyo3::{exceptions::PyValueError, prelude::*};
use safetensors::SafeTensors;
use web_rwkv::{
    context::{Context, ContextBuilder, InstanceExt},
    runtime::{
        infer::{InferInput, InferInputBatch, InferOption, InferOutput},
        loader::Loader,
        model::{
            ContextAutoLimits, ModelBuilder, ModelInfo, ModelVersion, Quant,
            State as ModelState, Bundle
        },
        v4, v5, v6, v7, TokioRuntime,
    },
    tensor::{
        kind::ReadWrite, DeepClone, TensorCpu, TensorGpu, TensorInit, TensorInto, TensorShape,
    },
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
    info: ModelInfo,
    context: Context,
    runtime: TokioRuntime<InferInput, InferOutput>,
    #[derivative(Debug = "ignore")]
    state: Arc<dyn ModelState + Send + Sync>,
}

#[pyclass]
#[derive(Debug, Clone)]
pub enum State {
    Cpu { state: StateCpu },
    Gpu { state: StateGpu },
}

#[pymethods]
impl State {
    pub fn deep_clone(&self) -> Self {
        match self.clone() {
            State::Cpu { state } => State::Cpu { state },
            State::Gpu { state } => {
                let state = StateGpu(state.0.deep_clone());
                State::Gpu { state }
            }
        }
    }

    pub fn device(&self) -> StateDevice {
        match self {
            State::Cpu { .. } => StateDevice::Cpu,
            State::Gpu { .. } => StateDevice::Gpu,
        }
    }

    pub fn to(&self, device: StateDevice) -> Self {
        match (self.clone(), device) {
            (Self::Cpu { state }, StateDevice::Gpu) => {
                let StateCpu(tensor, context) = state;
                let state = StateGpu(tensor.to(&context));
                Self::Gpu { state }
            }
            (Self::Gpu { state }, StateDevice::Cpu) => {
                let context = state.0.context.clone();
                let tensor = state.0.back_in_place();
                let state = StateCpu(tensor, context);
                Self::Cpu { state }
            }
            (state, _) => state,
        }
    }
}

#[pyclass]
#[derive(Debug, Clone)]
pub struct StateCpu(TensorCpu<f32>, Context);

#[pyclass]
#[derive(Debug, Clone)]
pub struct StateGpu(TensorGpu<f32, ReadWrite>);

#[pyclass]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum StateDevice {
    Cpu,
    Gpu,
}

async fn create_context(info: &ModelInfo) -> Result<Context> {
    let instance = web_rwkv::wgpu::Instance::default();
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
    quant_sf4: usize,
) -> Result<(
    Context,
    ModelInfo,
    TokioRuntime<InferInput, InferOutput>,
    Arc<dyn ModelState + Send + Sync>,
)> {
    let file = File::open(path)?;
    let data = unsafe { Mmap::map(&file)? };

    let model = SafeTensors::deserialize(&data)?;
    let info = Loader::info(&model)?;

    let context = create_context(&info).await?;
    let quant = (0..quant)
        .map(|layer| (layer, Quant::Int8))
        .chain((0..quant_nf4).map(|layer| (layer, Quant::NF4)))
        .chain((0..quant_sf4).map(|layer| (layer, Quant::SF4)))
        .collect();

    let builder = ModelBuilder::new(&context, model).quant(quant);

    match info.version {
        ModelVersion::V4 => {
            let model = builder.build_v4().await?;
            let bundle = v4::Bundle::<f16>::new(model, 1);
            let state = Arc::new(bundle.state());
            let runtime = TokioRuntime::new(bundle).await;
            Ok((context, info, runtime, state))
        }
        ModelVersion::V5 => {
            let model = builder.build_v5().await?;
            let bundle = v5::Bundle::<f16>::new(model, 1);
            let state = Arc::new(bundle.state());
            let runtime = TokioRuntime::new(bundle).await;
            Ok((context, info, runtime, state))
        }
        ModelVersion::V6 => {
            let model = builder.build_v6().await?;
            let bundle = v6::Bundle::<f16>::new(model, 1);
            let state = Arc::new(bundle.state());
            let runtime = TokioRuntime::new(bundle).await;
            Ok((context, info, runtime, state))
        }
        ModelVersion::V7 => {
            let model = builder.build_v7().await?;
            let bundle = v7::Bundle::<f16>::new(model, 1);
            let state = Arc::new(bundle.state());
            let runtime = TokioRuntime::new(bundle).await;
            Ok((context, info, runtime, state))
        }
    }
}

#[pymethods]
impl Model {
    #[new]
    #[pyo3(signature = (path, quant=0, quant_nf4=0, quant_sf4=0))]
    pub fn new(path: PathBuf, quant: usize, quant_nf4: usize, quant_sf4: usize) -> PyResult<Self> {
        let tokio = Arc::new(tokio::runtime::Runtime::new()?);
        let (context, info, runtime, state) = tokio
            .block_on(load_runtime(path, quant, quant_nf4, quant_sf4))
            .map_err(err)?;
        Ok(Self {
            tokio,
            context,
            info,
            runtime,
            state,
        })
    }

    pub fn info(&self) -> info::ModelInfo {
        self.info.clone().into()
    }

    pub fn init_state(&self) -> State {
        let state = StateCpu(self.state.init(), self.context.clone());
        State::Cpu { state }
    }

    pub fn clear_state(&self) {
        let _ = self.load_state(&self.init_state());
    }

    pub fn load_state(&self, state: &State) -> PyResult<()> {
        match state.clone() {
            State::Cpu { state } => self.state.load(state.0, 0),
            State::Gpu { state } => self.state.write(state.0, 0),
        }
        .map_err(err)
    }

    #[pyo3(signature = (device=StateDevice::Cpu))]
    pub fn back_state(&self, device: StateDevice) -> PyResult<State> {
        match device {
            StateDevice::Cpu => {
                let tensor = self.tokio.block_on(self.state.back(0)).map_err(err)?;
                let state = StateCpu(tensor, self.context.clone());
                Ok(State::Cpu { state })
            }
            StateDevice::Gpu => {
                let tensor = self.state.read(0).map_err(err)?;
                let state = StateGpu(tensor);
                Ok(State::Gpu { state })
            }
        }
    }

    #[pyo3(signature = (tokens, token_chunk_size=128))]
    pub fn run(&self, tokens: Vec<u16>, token_chunk_size: usize) -> PyResult<Vec<f32>> {
        let model = self.clone();
        let option = InferOption::Last;
        let output = self
            .tokio
            .block_on(model.run_internal(tokens, option, token_chunk_size))
            .map_err(err)?
            .to_vec();
        Ok(output)
    }

    #[pyo3(signature = (tokens, token_chunk_size=128))]
    pub fn run_full(&self, tokens: Vec<u16>, token_chunk_size: usize) -> PyResult<Vec<f32>> {
        let model = self.clone();
        let option = InferOption::Full;
        let output = self
            .tokio
            .block_on(model.run_internal(tokens, option, token_chunk_size))
            .map_err(err)?
            .to_vec();
        Ok(output)
    }
}

impl Model {
    async fn run_internal(
        &self,
        tokens: Vec<u16>,
        option: InferOption,
        token_chunk_size: usize,
    ) -> Result<TensorCpu<f32>> {
        if tokens.is_empty() {
            bail!("input tokens cannot be empty")
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

            let (input, output) = match self.runtime.infer(input).await {
                Ok(output) => output,
                Err(err) => {
                    bail!(err);
                }
            };

            num_token += output[0].0.shape()[1];
            let mut output = output[0].clone().to_vec();
            data.append(&mut output);
            inference.replace(input);
        }

        let num_vocab = self.info.num_vocab;
        let tensor = TensorCpu::from_data([num_vocab, num_token, 1, 1], data)?;
        Ok(tensor)
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
    module.add_class::<StateDevice>()?;
    module.add_class::<Tokenizer>()?;
    module.add_class::<info::ModelInfo>()?;
    module.add_class::<info::ModelVersion>()?;

    Ok(())
}
