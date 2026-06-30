use pyo3::prelude::*;

mod datasets;
mod encoder_util;
mod errors;
mod io;
mod models;
mod resource;
mod tokenizer;
mod train;
mod train_config;
mod vision;

use datasets::{
    PyDatasetClassification, PyDatasetCorpus, PyDatasetImageEdit, PyDatasetImageGen, PyDatasetQA,
};
use errors::{
    CPUError, CUDAError, DataMismatchError, DataMissingRowError, ModelMismatchError,
};
use io::{
    export, export_classifier_model, export_diffusion_model, import_classifier_model,
    import_diffusion_model, import_model, merge, merge_classifier, merge_diffusion_model,
    quantize, quantize_classifier_model, quantize_diffusion_model,
};
use models::{PyChatbot, PyClassifier, PyDiffusion};
use resource::{limit_percent, limit_resources};
use tokenizer::{PyBytePairEncoder, PyUnigramEncoder};
use train::{RL, SPIN, Train, TrainClassifier, TrainDiffusion};
use train_config::PyTrainConfig;
use vision::{vision_rgb_patch_from_image_path_py, vision_rgb_patches_from_image_path_py};

#[pymodule]
fn _native(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyTrainConfig>()?;
    m.add_class::<PyDatasetQA>()?;
    m.add_class::<PyDatasetCorpus>()?;
    m.add_class::<PyDatasetClassification>()?;
    m.add_class::<PyDatasetImageGen>()?;
    m.add_class::<PyDatasetImageEdit>()?;
    m.add_class::<PyBytePairEncoder>()?;
    m.add_class::<PyUnigramEncoder>()?;
    m.add_class::<PyChatbot>()?;
    m.add_class::<PyClassifier>()?;
    m.add_class::<PyDiffusion>()?;
    m.add_function(wrap_pyfunction!(Train, m)?)?;
    m.add_function(wrap_pyfunction!(TrainClassifier, m)?)?;
    m.add_function(wrap_pyfunction!(TrainDiffusion, m)?)?;
    m.add_function(wrap_pyfunction!(RL, m)?)?;
    m.add_function(wrap_pyfunction!(SPIN, m)?)?;
    m.add_function(wrap_pyfunction!(merge, m)?)?;
    m.add_function(wrap_pyfunction!(merge_classifier, m)?)?;
    m.add_function(wrap_pyfunction!(limit_resources, m)?)?;
    m.add("limit", m.getattr("limit_resources")?)?;
    m.add_function(wrap_pyfunction!(limit_percent, m)?)?;
    m.add_function(wrap_pyfunction!(export, m)?)?;
    m.add_function(wrap_pyfunction!(import_model, m)?)?;
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    m.add_function(wrap_pyfunction!(export_classifier_model, m)?)?;
    m.add_function(wrap_pyfunction!(import_classifier_model, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_classifier_model, m)?)?;
    m.add_function(wrap_pyfunction!(export_diffusion_model, m)?)?;
    m.add_function(wrap_pyfunction!(import_diffusion_model, m)?)?;
    m.add_function(wrap_pyfunction!(merge_diffusion_model, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_diffusion_model, m)?)?;
    m.add_function(wrap_pyfunction!(vision_rgb_patch_from_image_path_py, m)?)?;
    m.add_function(wrap_pyfunction!(vision_rgb_patches_from_image_path_py, m)?)?;
    m.add("vision_rgb_patch_from_image_path", m.getattr("vision_rgb_patch_from_image_path_py")?)?;
    m.add(
        "vision_rgb_patches_from_image_path",
        m.getattr("vision_rgb_patches_from_image_path_py")?,
    )?;
    let py = m.py();
    m.add("CPUError", py.get_type::<CPUError>())?;
    m.add("CUDAError", py.get_type::<CUDAError>())?;
    m.add("DataMismatchError", py.get_type::<DataMismatchError>())?;
    m.add("DataMissingRowError", py.get_type::<DataMissingRowError>())?;
    m.add("ModelMismatchError", py.get_type::<ModelMismatchError>())?;
    Ok(())
}
