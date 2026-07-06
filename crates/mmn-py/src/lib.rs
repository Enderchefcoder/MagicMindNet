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
    dequantize_ggml, export, export_classifier_model, export_diffusion_model, gguf_info_json,
    import_classifier_model, import_diffusion_model, import_model, load_checkpoint,
    load_gguf_bpe_tokenizer, load_gguf_tokenizer, merge, merge_classifier,
    merge_diffusion_model, quantize, quantize_classifier_model, quantize_diffusion_model,
    read_arrays_auto, read_arrays_auto_bytes, read_flax, read_gguf_arrays, read_ggml_legacy,
    read_h5, read_keras,
    read_npy, read_npz, read_pickle_arrays, read_tflite, read_zarr, write_arrays_bytes,
    write_safetensors_sharded,
    read_onnx, read_pt, read_safetensors,
    read_tf_checkpoint, write_h5,
    write_flax, write_gguf_arrays, write_npy, write_npz, write_onnx, write_pickle_arrays,
    write_zarr,
    write_pt, write_safetensors,
    write_tf_checkpoint,
};
use models::{PyChatbot, PyClassifier, PyDiffusion};
use resource::{limit_percent, limit_resources};
use tokenizer::{PyBytePairEncoder, PyGpt2BpeEncoder, PyUnigramEncoder};
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
    m.add_class::<PyGpt2BpeEncoder>()?;
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
    m.add_function(wrap_pyfunction!(load_checkpoint, m)?)?;
    m.add("load", m.getattr("load_checkpoint")?)?;
    m.add_function(wrap_pyfunction!(quantize, m)?)?;
    m.add_function(wrap_pyfunction!(export_classifier_model, m)?)?;
    m.add_function(wrap_pyfunction!(import_classifier_model, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_classifier_model, m)?)?;
    m.add_function(wrap_pyfunction!(export_diffusion_model, m)?)?;
    m.add_function(wrap_pyfunction!(import_diffusion_model, m)?)?;
    m.add_function(wrap_pyfunction!(merge_diffusion_model, m)?)?;
    m.add_function(wrap_pyfunction!(quantize_diffusion_model, m)?)?;
    m.add_function(wrap_pyfunction!(read_npy, m)?)?;
    m.add_function(wrap_pyfunction!(write_npy, m)?)?;
    m.add_function(wrap_pyfunction!(read_npz, m)?)?;
    m.add_function(wrap_pyfunction!(write_npz, m)?)?;
    m.add_function(wrap_pyfunction!(read_pt, m)?)?;
    m.add_function(wrap_pyfunction!(write_pt, m)?)?;
    m.add_function(wrap_pyfunction!(gguf_info_json, m)?)?;
    m.add_function(wrap_pyfunction!(dequantize_ggml, m)?)?;
    m.add_function(wrap_pyfunction!(load_gguf_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(load_gguf_bpe_tokenizer, m)?)?;
    m.add_function(wrap_pyfunction!(read_h5, m)?)?;
    m.add_function(wrap_pyfunction!(read_keras, m)?)?;
    m.add_function(wrap_pyfunction!(read_tf_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(read_onnx, m)?)?;
    m.add_function(wrap_pyfunction!(read_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(read_flax, m)?)?;
    m.add_function(wrap_pyfunction!(read_gguf_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(write_gguf_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(read_arrays_auto, m)?)?;
    m.add_function(wrap_pyfunction!(read_arrays_auto_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(read_tflite, m)?)?;
    m.add_function(wrap_pyfunction!(read_pickle_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(write_safetensors_sharded, m)?)?;
    m.add_function(wrap_pyfunction!(write_arrays_bytes, m)?)?;
    m.add_function(wrap_pyfunction!(read_zarr, m)?)?;
    m.add_function(wrap_pyfunction!(write_zarr, m)?)?;
    m.add_function(wrap_pyfunction!(write_pickle_arrays, m)?)?;
    m.add_function(wrap_pyfunction!(read_ggml_legacy, m)?)?;
    m.add_function(wrap_pyfunction!(write_flax, m)?)?;
    m.add_function(wrap_pyfunction!(write_safetensors, m)?)?;
    m.add_function(wrap_pyfunction!(write_onnx, m)?)?;
    m.add_function(wrap_pyfunction!(write_tf_checkpoint, m)?)?;
    m.add_function(wrap_pyfunction!(write_h5, m)?)?;
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
