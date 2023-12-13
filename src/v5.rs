use std::{fs::File, path::PathBuf, sync::Arc};

use anyhow::{bail, Result};
use itertools::Itertools;
use memmap2::Mmap;
use pyo3::{exceptions::PyValueError, prelude::*};
use web_rwkv::{
    context::{Context, ContextBuilder, Instance},
    model::{loader::Loader, v5, ModelBuilder, ModelVersion, Quant},
    wgpu,
};

#[pyclass]
pub struct Model(Arc<v5::Model<'static>>);

#[pyclass]
pub struct ModelState(Arc<v5::ModelState>);

#[pyclass]
pub struct BackedState(Arc<v5::BackedState>);

async fn create_context() -> Result<Context> {
    let instance = Instance::new();
    let adapter = instance
        .adapter(wgpu::PowerPreference::HighPerformance)
        .await?;
    let context = ContextBuilder::new(adapter)
        .with_default_pipelines()
        .build()
        .await?;
    println!("{:#?}", context.adapter.get_info());
    Ok(context)
}

fn load_model(
    context: &Context,
    data: &[u8],
    // lora: Option<PathBuf>,
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
    model.build()
    // match lora {
    //     Some(lora) => {
    //         let file = File::open(lora)?;
    //         let map = unsafe { Mmap::map(&file)? };
    //         model
    //             .add_lora(Lora {
    //                 data: map.to_vec(),
    //                 blend: Default::default(),
    //             })
    //             .build()
    //     }
    //     None => model.build(),
    // }
}

#[pymethods]
impl Model {
    #[new]
    pub fn new(
        file: PathBuf,
        turbo: bool,
        quant: Option<usize>,
        quant_nf4: Option<usize>,
    ) -> PyResult<Self> {
        let model = || {
            let context = pollster::block_on(create_context())?;
            let file = File::open(file)?;
            let map = unsafe { Mmap::map(&file)? };

            let info = Loader::info(&map)?;
            match info.version {
                ModelVersion::V5 => {
                    anyhow::Ok(load_model(&context, &map, quant, quant_nf4, turbo)?)
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
    pub fn new(_model: &Model) -> Self {
        todo!()
    }
}
