use std::{fs::File, path::PathBuf, sync::Arc};

use anyhow::{bail, Result};
use itertools::Itertools;
use memmap2::Mmap;
use pyo3::{exceptions::PyValueError, prelude::*};
use web_rwkv::{
    context::Context,
    model::{
        loader::Loader, run::ModelRun as _, v5, BackedState as _, Lora, ModelBase as _,
        ModelBuilder, ModelState as _, ModelVersion, Quant, StateBuilder,
    },
};
use web_rwkv_derive::Deref;

use crate::create_context;

#[pyclass]
#[derive(Debug, Deref, Clone)]
pub struct Model(Arc<v5::Model<'static>>);

#[pyclass]
#[derive(Debug, Deref, Clone)]
pub struct ModelState(v5::ModelState);

#[pyclass]
#[derive(Debug, Deref, Clone)]
pub struct BackedState(v5::BackedState);

fn load_model(
    context: &Context,
    data: &[u8],
    lora: Option<PathBuf>,
    quant: Option<usize>,
    quant_nf4: Option<usize>,
    turbo: bool,
) -> Result<v5::Model<'static>> {
    let quant = quant
        .map(|layer| (0..layer).map(|layer| (layer, Quant::Int8)).collect_vec())
        .unwrap_or_default();
    let quant_nf4 = quant_nf4
        .map(|layer| (0..layer).map(|layer| (layer, Quant::NF4)).collect_vec())
        .unwrap_or_default();
    let quant = quant.into_iter().chain(quant_nf4).collect();
    let model = ModelBuilder::new(context, data)
        .with_quant(quant)
        .with_turbo(turbo);
    match lora {
        Some(lora) => {
            let file = File::open(lora)?;
            let map = unsafe { Mmap::map(&file)? };
            model
                .add_lora(Lora {
                    data: map.to_vec(),
                    blend: Default::default(),
                })
                .build()
        }
        None => model.build(),
    }
}

#[pymethods]
impl Model {
    #[new]
    pub fn new(
        file: PathBuf,
        turbo: bool,
        lora: Option<PathBuf>,
        quant: Option<usize>,
        quant_nf4: Option<usize>,
    ) -> PyResult<Self> {
        let model = || {
            let file = File::open(file)?;
            let map = unsafe { Mmap::map(&file)? };

            let info = Loader::info(&map)?;
            let context = pollster::block_on(create_context(info.max_buffer_size() as u32))?;
            println!("{:#?}", info);
            match info.version {
                ModelVersion::V5 => {
                    anyhow::Ok(load_model(&context, &map, lora, quant, quant_nf4, turbo)?)
                }
                version => bail!(
                    "model version {:?} is incorrect, should be {:?}",
                    version,
                    ModelVersion::V5
                ),
            }
        };
        model()
            .map(|model| Self(model.into()))
            .map_err(|err| PyValueError::new_err(err.to_string()))
    }
}

#[pymethods]
impl ModelState {
    #[new]
    pub fn new(model: &Model, batch: usize) -> Self {
        let context = model.context();
        let info = model.info();
        Self(
            StateBuilder::new(context, info)
                .with_max_batch(batch)
                .build::<v5::ModelState>(),
        )
    }

    pub fn load(&self, backed: &BackedState) -> PyResult<()> {
        self.0
            .load(backed)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    pub fn load_batch(&self, backed: &BackedState, batch: usize) -> PyResult<()> {
        self.0
            .load_batch(backed, batch)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(())
    }

    pub fn back(&self) -> BackedState {
        let back = pollster::block_on(self.0.back());
        BackedState(back)
    }

    pub fn back_batch(&self, batch: usize) -> PyResult<BackedState> {
        let back = pollster::block_on(self.0.back_batch(batch))
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(BackedState(back))
    }
}

#[pymethods]
impl BackedState {
    #[new]
    pub fn new(model: &Model, batch: usize, data: Vec<Vec<f32>>) -> PyResult<Self> {
        let backed = || {
            let context = model.context();
            let info = model.info();

            let mut backed = StateBuilder::new(context, info)
                .with_max_batch(batch)
                .build_backed::<v5::BackedState>();

            if data.len() != backed.data.len() {
                bail!(
                    "incorrect state chunks: {} vs {}",
                    data.len(),
                    backed.data.len()
                );
            }

            let mut shape_data = vec![];
            for (backed, data) in backed.data.iter().zip(data.into_iter()) {
                let shape = backed.0;
                if data.len() != shape.len() {
                    bail!("incorrect state size: {} vs. {}", data.len(), shape.len());
                }
                shape_data.push((shape, data))
            }

            backed.data = Arc::new(shape_data);
            Ok(backed)
        };
        let backed = backed().map_err(|err| PyValueError::new_err(err.to_string()))?;
        Ok(Self(backed))
    }

    pub fn max_batch(&self) -> usize {
        self.0.max_batch()
    }
}

fn run_one_internal(model: Model, state: ModelState, tokens: Vec<u16>) -> Result<Vec<f32>> {
    if state.max_batch() != 1 {
        bail!("state batch size must be 1")
    }
    if tokens.is_empty() {
        bail!("prompt cannot be empty")
    }

    let mut tokens = vec![tokens];
    let mut logits = vec![None];

    while logits[0].is_none() {
        logits = pollster::block_on(model.run(&mut tokens, &state.0))?;
    }
    Ok(logits[0].clone().unwrap())
}

#[pyfunction]
pub fn run_one(
    model: &Model,
    tokens: Vec<u16>,
    state: Option<ModelState>,
) -> PyResult<(Vec<f32>, ModelState)> {
    let state = state.unwrap_or_else(|| ModelState::new(model, 1));
    let logits = run_one_internal(model.clone(), state.clone(), tokens)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    Ok((logits, state))
}
