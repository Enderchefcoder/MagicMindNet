use mmn_io::{
    detect_checkpoint_kind, export_bin, export_classifier, export_diffusion,
    export_gguf_with_tokenizer, export_hf_classifier_safetensors, export_hf_safetensors,
    export_npz, export_safetensors,
    export_torch_pt, import_bin, import_classifier, import_diffusion, import_gguf,
    import_hf_classifier_safetensors, import_hf_safetensors, import_npz, import_safetensors,
    import_torch_pt, merge_classifiers, merge_diffusion, merge_models, quantize_classifier,
    quantize_diffusion, quantize_model, read_npz_arrays, read_torch_arrays, write_torch_arrays,
    CheckpointKind, NamedArray, TokenizerSidecarRefs,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::path::Path;

use crate::errors::mmn_err_to_py;
use crate::models::{PyChatbot, PyClassifier, PyDiffusion};
use crate::tokenizer::{PyBytePairEncoder, PyUnigramEncoder};

fn bpe_sidecar_name(checkpoint_path: &str) -> String {
    let p = Path::new(checkpoint_path);
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    format!("{stem}.bpe.mmn")
}

fn unigram_sidecar_name(checkpoint_path: &str) -> String {
    let p = Path::new(checkpoint_path);
    let stem = p
        .file_stem()
        .and_then(|s| s.to_str())
        .unwrap_or("model");
    format!("{stem}.unigram.mmn")
}

fn write_tokenizer_sidecars(
    path: &str,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
) -> PyResult<(Option<String>, Option<String>)> {
    if bpe_encoder.is_some() && unigram_encoder.is_some() {
        return Err(PyValueError::new_err(
            "Pass at most one of bpe_encoder or unigram_encoder to export",
        ));
    }
    let parent = Path::new(path).parent().unwrap_or_else(|| Path::new("."));
    if let Some(enc) = bpe_encoder {
        let sidecar_name = bpe_sidecar_name(path);
        let sidecar_path = parent.join(&sidecar_name);
        enc.inner
            .export_json(
                sidecar_path
                    .to_str()
                    .ok_or_else(|| PyValueError::new_err("invalid BPE sidecar path"))?,
            )
            .map_err(mmn_err_to_py)?;
        return Ok((Some(sidecar_name), None));
    }
    if let Some(enc) = unigram_encoder {
        let sidecar_name = unigram_sidecar_name(path);
        let sidecar_path = parent.join(&sidecar_name);
        enc.inner
            .export_json(
                sidecar_path
                    .to_str()
                    .ok_or_else(|| PyValueError::new_err("invalid unigram sidecar path"))?,
            )
            .map_err(mmn_err_to_py)?;
        return Ok((None, Some(sidecar_name)));
    }
    Ok((None, None))
}

/// Export a chatbot checkpoint (shared by `ai.export` and `Chatbot.save`).
pub(crate) fn export_chatbot_to_path(
    model: &mmn_models::Chatbot,
    format: &str,
    path: &str,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
) -> PyResult<()> {
    // GGUF embeds its vocabulary in-file (llama.cpp convention); no sidecars.
    if let Some(quant) = match format {
        "gguf" => Some("f32"),
        "gguf-f16" | "gguf_f16" => Some("f16"),
        "gguf-q8_0" | "gguf_q8_0" => Some("q8_0"),
        "gguf-q4_0" | "gguf_q4_0" => Some("q4_0"),
        "gguf-q4_1" | "gguf_q4_1" => Some("q4_1"),
        "gguf-q5_0" | "gguf_q5_0" => Some("q5_0"),
        "gguf-q5_1" | "gguf_q5_1" => Some("q5_1"),
        "gguf-q2_k" | "gguf_q2_k" => Some("q2_k"),
        "gguf-q3_k" | "gguf_q3_k" => Some("q3_k"),
        "gguf-q4_k" | "gguf_q4_k" => Some("q4_k"),
        "gguf-q5_k" | "gguf_q5_k" => Some("q5_k"),
        "gguf-q6_k" | "gguf_q6_k" => Some("q6_k"),
        "gguf-iq4_nl" | "gguf_iq4_nl" => Some("iq4_nl"),
        "gguf-iq4_xs" | "gguf_iq4_xs" => Some("iq4_xs"),
        _ => None,
    } {
        if bpe_encoder.is_some() {
            return Err(PyValueError::new_err(
                "GGUF embeds unigram vocabularies only; pass unigram_encoder= (or export BPE with safetensors sidecars)",
            ));
        }
        return export_gguf_with_tokenizer(model, path, quant, unigram_encoder.map(|e| &e.inner))
            .map_err(mmn_err_to_py);
    }
    if matches!(format, "bin" | "npz" | "numpy" | "pt" | "pytorch" | "torch")
        && (bpe_encoder.is_some() || unigram_encoder.is_some())
    {
        return Err(PyValueError::new_err(
            "bpe_encoder / unigram_encoder are only supported with safetensors or gguf export",
        ));
    }
    let (bpe_rel, uni_rel) = write_tokenizer_sidecars(path, bpe_encoder, unigram_encoder)?;
    let sidecars = TokenizerSidecarRefs {
        bpe: bpe_rel.as_deref(),
        unigram: uni_rel.as_deref(),
    };
    match format {
        "safetensors" => export_safetensors(model, path, sidecars).map_err(mmn_err_to_py),
        "hf-safetensors" | "hf_safetensors" => {
            export_hf_safetensors(model, path, sidecars).map_err(mmn_err_to_py)
        }
        "bin" => export_bin(model, path).map_err(mmn_err_to_py),
        "npz" | "numpy" => export_npz(model, path).map_err(mmn_err_to_py),
        "pt" | "pytorch" | "torch" => export_torch_pt(model, path).map_err(mmn_err_to_py),
        _ => Err(PyValueError::new_err(format!(
            "Unknown format: {format}. Supported: safetensors, hf-safetensors, bin, gguf, gguf-f16, gguf-q8_0, gguf-q4_0, npz, pt"
        ))),
    }
}

/// Import a chatbot checkpoint (shared by `ai.import_model` and `Chatbot.load`).
pub(crate) fn import_chatbot_from_path(
    format: &str,
    path: &str,
) -> PyResult<mmn_models::Chatbot> {
    match format {
        "safetensors" => import_safetensors(path, 0).map_err(mmn_err_to_py),
        "hf-safetensors" | "hf_safetensors" => import_hf_safetensors(path).map_err(mmn_err_to_py),
        "bin" => import_bin(path).map_err(mmn_err_to_py),
        f if f.starts_with("gguf") => import_gguf(path).map_err(mmn_err_to_py),
        "npz" | "numpy" => import_npz(path).map_err(mmn_err_to_py),
        "pt" | "pytorch" | "torch" => import_torch_pt(path).map_err(mmn_err_to_py),
        "sharded" => mmn_io::import_sharded(path).map_err(mmn_err_to_py),
        _ => Err(PyValueError::new_err(format!(
            "Unknown format: {format}. Supported: safetensors, hf-safetensors, bin, gguf, npz, pt, sharded"
        ))),
    }
}

/// Export a classifier checkpoint (shared by `ai.export_classifier` and `Classifier.save`).
pub(crate) fn export_classifier_to_path(
    model: &mmn_models::Classifier,
    format: &str,
    path: &str,
) -> PyResult<()> {
    match format {
        "safetensors" => export_classifier(model, path).map_err(mmn_err_to_py),
        "hf-safetensors" | "hf_safetensors" => {
            export_hf_classifier_safetensors(model, path).map_err(mmn_err_to_py)
        }
        _ => Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    }
}

/// Import a classifier checkpoint (shared by `ai.import_classifier` and `Classifier.load`).
pub(crate) fn import_classifier_from_path(
    format: &str,
    path: &str,
) -> PyResult<mmn_models::Classifier> {
    match format {
        "safetensors" => import_classifier(path).map_err(mmn_err_to_py),
        "hf-safetensors" | "hf_safetensors" => {
            import_hf_classifier_safetensors(path).map_err(mmn_err_to_py)
        }
        _ => Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    }
}

/// Export a diffusion checkpoint (shared by `ai.export_diffusion` and `Diffusion.save`).
pub(crate) fn export_diffusion_to_path(
    model: &mmn_models::Diffusion,
    format: &str,
    path: &str,
) -> PyResult<()> {
    match format {
        "safetensors" => export_diffusion(model, path).map_err(mmn_err_to_py),
        _ => Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    }
}

/// Import a diffusion checkpoint (shared by `ai.import_diffusion` and `Diffusion.load`).
pub(crate) fn import_diffusion_from_path(
    format: &str,
    path: &str,
) -> PyResult<mmn_models::Diffusion> {
    match format {
        "safetensors" => import_diffusion(path).map_err(mmn_err_to_py),
        _ => Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    }
}

/// Detect the model family stored at `path`, erroring when it is not `expected`.
pub(crate) fn expect_checkpoint_family(
    path: &str,
    expected: &str,
    load_hint: &str,
) -> PyResult<CheckpointKind> {
    let kind = detect_checkpoint_kind(path).map_err(mmn_err_to_py)?;
    if kind.family() == expected {
        Ok(kind)
    } else {
        Err(PyValueError::new_err(format!(
            "{path} stores a {} checkpoint, not a {expected}.\nFix: {load_hint}",
            kind.family()
        )))
    }
}

#[pyfunction]
pub fn merge(model1: &PyChatbot, model2: &PyChatbot) -> PyResult<PyChatbot> {
    let m = merge_models(&model1.inner, &model2.inner).map_err(mmn_err_to_py)?;
    Ok(PyChatbot { inner: m })
}

#[pyfunction]
pub fn merge_classifier(model1: &PyClassifier, model2: &PyClassifier) -> PyResult<PyClassifier> {
    let m = merge_classifiers(&model1.inner, &model2.inner).map_err(mmn_err_to_py)?;
    Ok(PyClassifier { inner: m })
}

/// Save a Chatbot checkpoint (`format`: "safetensors", "hf-safetensors", or "bin").
#[pyfunction]
#[pyo3(signature = (model, format, path, bpe_encoder=None, unigram_encoder=None))]
pub fn export(
    model: &PyChatbot,
    format: &str,
    path: &str,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
) -> PyResult<()> {
    export_chatbot_to_path(&model.inner, format, path, bpe_encoder, unigram_encoder)
}

/// Load a Chatbot from the first path in `files` (strict tensor validation).
#[pyfunction]
pub fn import_model(format: &str, files: Vec<String>) -> PyResult<PyChatbot> {
    let path = files.first().ok_or_else(|| PyValueError::new_err("files required"))?;
    Ok(PyChatbot {
        inner: import_chatbot_from_path(format, path)?,
    })
}

#[pyfunction]
pub fn quantize(model: &mut PyChatbot, quant: &str) -> PyResult<()> {
    quantize_model(&mut model.inner, quant).map_err(mmn_err_to_py)
}

#[pyfunction]
pub fn export_classifier_model(model: &PyClassifier, format: &str, path: &str) -> PyResult<()> {
    export_classifier_to_path(&model.inner, format, path)
}

#[pyfunction]
pub fn import_classifier_model(format: &str, files: Vec<String>) -> PyResult<PyClassifier> {
    let path = files
        .first()
        .ok_or_else(|| PyValueError::new_err("files required"))?;
    Ok(PyClassifier {
        inner: import_classifier_from_path(format, path)?,
    })
}

#[pyfunction]
pub fn quantize_classifier_model(model: &mut PyClassifier, quant: &str) -> PyResult<()> {
    quantize_classifier(&mut model.inner, quant).map_err(mmn_err_to_py)
}

#[pyfunction]
pub fn export_diffusion_model(model: &PyDiffusion, format: &str, path: &str) -> PyResult<()> {
    export_diffusion_to_path(&model.inner, format, path)
}

#[pyfunction]
pub fn import_diffusion_model(format: &str, files: Vec<String>) -> PyResult<PyDiffusion> {
    let path = files
        .first()
        .ok_or_else(|| PyValueError::new_err("files required"))?;
    Ok(PyDiffusion {
        inner: import_diffusion_from_path(format, path)?,
    })
}

#[pyfunction]
pub fn merge_diffusion_model(model1: &PyDiffusion, model2: &PyDiffusion) -> PyResult<PyDiffusion> {
    let m = merge_diffusion(&model1.inner, &model2.inner).map_err(mmn_err_to_py)?;
    Ok(PyDiffusion { inner: m })
}

#[pyfunction]
pub fn quantize_diffusion_model(model: &mut PyDiffusion, quant: &str) -> PyResult<()> {
    quantize_diffusion(&mut model.inner, quant).map_err(mmn_err_to_py)
}

/// Load any MagicMindNet checkpoint and return the right model type.
///
/// Detects the stored family (Chatbot, Classifier, or Diffusion) and format
/// (JSON or binary safetensors, bin stub) automatically, so
/// `ai.load("model.mmn")` is all a beginner needs.
#[pyfunction]
pub fn load_checkpoint(py: Python<'_>, path: &str) -> PyResult<PyObject> {
    let kind = detect_checkpoint_kind(path).map_err(mmn_err_to_py)?;
    match kind {
        CheckpointKind::Chatbot => {
            let inner = import_safetensors(path, 0).map_err(mmn_err_to_py)?;
            Ok(PyChatbot { inner }.into_pyobject(py)?.into_any().unbind())
        }
        CheckpointKind::ChatbotBin => {
            let inner = import_bin(path).map_err(mmn_err_to_py)?;
            Ok(PyChatbot { inner }.into_pyobject(py)?.into_any().unbind())
        }
        CheckpointKind::ChatbotGguf => {
            let inner = import_gguf(path).map_err(mmn_err_to_py)?;
            Ok(PyChatbot { inner }.into_pyobject(py)?.into_any().unbind())
        }
        CheckpointKind::ChatbotNpz => {
            let inner = import_npz(path).map_err(mmn_err_to_py)?;
            Ok(PyChatbot { inner }.into_pyobject(py)?.into_any().unbind())
        }
        CheckpointKind::ChatbotTorch => {
            let inner = import_torch_pt(path).map_err(mmn_err_to_py)?;
            Ok(PyChatbot { inner }.into_pyobject(py)?.into_any().unbind())
        }
        CheckpointKind::ChatbotSharded => {
            let inner = mmn_io::import_sharded(path).map_err(mmn_err_to_py)?;
            Ok(PyChatbot { inner }.into_pyobject(py)?.into_any().unbind())
        }
        CheckpointKind::ChatbotGgmlLegacy => {
            let inner = mmn_io::import_ggml_legacy_chatbot(path).map_err(mmn_err_to_py)?;
            Ok(PyChatbot { inner }.into_pyobject(py)?.into_any().unbind())
        }
        CheckpointKind::Classifier => {
            let inner = import_classifier(path).map_err(mmn_err_to_py)?;
            Ok(PyClassifier { inner }.into_pyobject(py)?.into_any().unbind())
        }
        CheckpointKind::Diffusion => {
            let inner = import_diffusion(path).map_err(mmn_err_to_py)?;
            Ok(PyDiffusion { inner }.into_pyobject(py)?.into_any().unbind())
        }
    }
}

/// Read a NumPy `.npy` file into `(shape, flat f32 values)`.
#[pyfunction]
pub fn read_npy(path: &str) -> PyResult<(Vec<usize>, Vec<f32>)> {
    let bytes = std::fs::read(path)
        .map_err(|e| PyValueError::new_err(format!("cannot read npy {path}: {e}")))?;
    let arr = mmn_io::decode_npy(&bytes).map_err(mmn_err_to_py)?;
    Ok((arr.shape, arr.data))
}

/// Write a NumPy `.npy` file from `(shape, flat f32 values)`.
#[pyfunction]
pub fn write_npy(path: &str, shape: Vec<usize>, data: Vec<f32>) -> PyResult<()> {
    let bytes = mmn_io::encode_npy_f32(&shape, &data).map_err(mmn_err_to_py)?;
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    std::fs::write(path, bytes).map_err(|e| PyValueError::new_err(e.to_string()))
}

/// Read every array in an `.npz` archive as `[(name, shape, values), ...]`.
#[pyfunction]
pub fn read_npz(path: &str) -> PyResult<Vec<NamedArray>> {
    read_npz_arrays(path).map_err(mmn_err_to_py)
}

/// Write named arrays to an `.npz` archive readable by `numpy.load`
/// (`compress=True` matches `np.savez_compressed`).
#[pyfunction]
#[pyo3(signature = (path, arrays, compress=false))]
pub fn write_npz(path: &str, arrays: Vec<NamedArray>, compress: bool) -> PyResult<()> {
    mmn_io::write_npz_arrays_opts(path, &arrays, compress).map_err(mmn_err_to_py)
}

/// Read every tensor in a PyTorch `.pt` state dict as `[(name, shape, values), ...]`.
#[pyfunction]
pub fn read_pt(path: &str) -> PyResult<Vec<NamedArray>> {
    read_torch_arrays(path).map_err(mmn_err_to_py)
}

/// Dequantize a raw GGML block payload (`type_name` e.g. "Q4_K", "IQ2_XXS").
#[pyfunction]
pub fn dequantize_ggml(type_name: &str, data: Vec<u8>, numel: usize) -> PyResult<Vec<f32>> {
    let ty = ggml_type_by_name(type_name)
        .ok_or_else(|| PyValueError::new_err(format!("unknown GGML type {type_name:?}")))?;
    mmn_io::dequantize_ggml(ty, &data, numel).map_err(mmn_err_to_py)
}

fn ggml_type_by_name(name: &str) -> Option<mmn_io::GgmlType> {
    use mmn_io::GgmlType as T;
    Some(match name.to_ascii_uppercase().as_str() {
        "F32" => T::F32,
        "F16" => T::F16,
        "BF16" => T::BF16,
        "F64" => T::F64,
        "Q4_0" => T::Q4_0,
        "Q4_1" => T::Q4_1,
        "Q5_0" => T::Q5_0,
        "Q5_1" => T::Q5_1,
        "Q8_0" => T::Q8_0,
        "Q8_1" => T::Q8_1,
        "Q2_K" => T::Q2K,
        "Q3_K" => T::Q3K,
        "Q4_K" => T::Q4K,
        "Q5_K" => T::Q5K,
        "Q6_K" => T::Q6K,
        "Q8_K" => T::Q8K,
        "IQ4_NL" => T::Iq4Nl,
        "IQ4_XS" => T::Iq4Xs,
        "IQ2_XXS" => T::Iq2Xxs,
        "IQ2_XS" => T::Iq2Xs,
        "IQ2_S" => T::Iq2S,
        "IQ3_XXS" => T::Iq3Xxs,
        "IQ3_S" => T::Iq3S,
        "IQ1_S" => T::Iq1S,
        "IQ1_M" => T::Iq1M,
        "TQ1_0" => T::Tq1_0,
        "TQ2_0" => T::Tq2_0,
        "MXFP4" => T::Mxfp4,
        "NVFP4" => T::Nvfp4,
        "I8" => T::I8,
        "I16" => T::I16,
        "I32" => T::I32,
        "I64" => T::I64,
        _ => return None,
    })
}

/// GGUF file inspection: metadata + tensor summaries as a JSON string.
#[pyfunction]
pub fn gguf_info_json(path: &str) -> PyResult<String> {
    let info = mmn_io::gguf_info_json(path).map_err(mmn_err_to_py)?;
    Ok(info.to_string())
}

/// Extract the SentencePiece vocabulary embedded in a GGUF file.
#[pyfunction]
pub fn load_gguf_tokenizer(path: &str) -> PyResult<crate::tokenizer::PyUnigramEncoder> {
    let inner = mmn_io::import_gguf_tokenizer(path).map_err(mmn_err_to_py)?;
    Ok(crate::tokenizer::PyUnigramEncoder { inner })
}

/// Extract the byte-level BPE ("gpt2") vocabulary embedded in a GGUF file.
#[pyfunction]
pub fn load_gguf_bpe_tokenizer(path: &str) -> PyResult<crate::tokenizer::PyGpt2BpeEncoder> {
    let inner = mmn_io::import_gguf_bpe_tokenizer(path).map_err(mmn_err_to_py)?;
    Ok(crate::tokenizer::PyGpt2BpeEncoder { inner })
}

/// Read every dataset in an HDF5 file as `[(path, shape, values), ...]`.
#[pyfunction]
pub fn read_h5(path: &str) -> PyResult<Vec<NamedArray>> {
    mmn_io::read_h5_arrays(path).map_err(mmn_err_to_py)
}

/// Read weights from a Keras `.keras` / `.weights.h5` / `.h5` file.
#[pyfunction]
pub fn read_keras(path: &str) -> PyResult<Vec<NamedArray>> {
    mmn_io::read_keras_arrays(path).map_err(mmn_err_to_py)
}

/// Read every tensor in a TF checkpoint v2 (`prefix` or `prefix.index`).
#[pyfunction]
pub fn read_tf_checkpoint(path: &str) -> PyResult<Vec<NamedArray>> {
    mmn_io::read_tf_checkpoint_arrays(path).map_err(mmn_err_to_py)
}

/// Read every graph initializer (weight) in an ONNX model.
#[pyfunction]
pub fn read_onnx(path: &str) -> PyResult<Vec<NamedArray>> {
    mmn_io::read_onnx_arrays(path).map_err(mmn_err_to_py)
}

/// Read every tensor in a `.safetensors` file (all spec dtypes, as f32).
#[pyfunction]
pub fn read_safetensors(path: &str) -> PyResult<Vec<NamedArray>> {
    mmn_io::read_safetensors_arrays(path).map_err(mmn_err_to_py)
}

/// Read a Flax msgpack checkpoint into `/`-joined named arrays.
#[pyfunction]
pub fn read_flax(path: &str) -> PyResult<Vec<NamedArray>> {
    mmn_io::read_flax_arrays(path).map_err(mmn_err_to_py)
}

/// Dequantize every tensor in a GGUF file into named f32 arrays.
#[pyfunction]
pub fn read_gguf_arrays(path: &str) -> PyResult<Vec<NamedArray>> {
    mmn_io::read_gguf_arrays(path).map_err(mmn_err_to_py)
}

/// Write named f32 arrays as a GGUF file (F32 tensors).
#[pyfunction]
pub fn write_gguf_arrays(path: &str, arrays: Vec<NamedArray>) -> PyResult<()> {
    mmn_io::write_gguf_arrays(path, &arrays).map_err(mmn_err_to_py)
}

/// Detect and read any supported tensor container; returns
/// `(format_name, arrays)`.
#[pyfunction]
pub fn read_arrays_auto(path: &str) -> PyResult<(String, Vec<NamedArray>)> {
    let (format, arrays) = mmn_io::read_arrays_auto(path).map_err(mmn_err_to_py)?;
    Ok((format.as_str().to_string(), arrays))
}

/// Read every weight tensor in a TFLite flatbuffer model.
#[pyfunction]
pub fn read_tflite(path: &str) -> PyResult<Vec<NamedArray>> {
    mmn_io::read_tflite_arrays(path).map_err(mmn_err_to_py)
}

/// Read a legacy GGML/GGMF/GGJT file; returns
/// `(container_name, hparams, vocab_tokens, arrays)`.
#[pyfunction]
#[allow(clippy::type_complexity)]
pub fn read_ggml_legacy(
    path: &str,
) -> PyResult<(String, Vec<i32>, Vec<(Vec<u8>, f32)>, Vec<NamedArray>)> {
    let file = mmn_io::read_ggml_legacy(path).map_err(mmn_err_to_py)?;
    Ok((
        file.container.as_str().to_string(),
        file.hparams.to_vec(),
        file.vocab,
        file.arrays,
    ))
}

/// Write named f32 arrays as a Flax msgpack pytree.
#[pyfunction]
pub fn write_flax(path: &str, arrays: Vec<NamedArray>) -> PyResult<()> {
    mmn_io::write_flax_arrays(path, &arrays).map_err(mmn_err_to_py)
}

/// Write named f32 arrays as a `.safetensors` file.
#[pyfunction]
pub fn write_safetensors(path: &str, arrays: Vec<NamedArray>) -> PyResult<()> {
    mmn_io::write_safetensors_arrays(path, &arrays).map_err(mmn_err_to_py)
}

/// Write named arrays as an ONNX model (`onnx.load`-compatible).
#[pyfunction]
pub fn write_onnx(path: &str, arrays: Vec<NamedArray>) -> PyResult<()> {
    mmn_io::write_onnx_arrays(path, &arrays).map_err(mmn_err_to_py)
}

/// Write a TF checkpoint v2 (`tf.train.load_checkpoint`-compatible).
#[pyfunction]
pub fn write_tf_checkpoint(path: &str, arrays: Vec<NamedArray>) -> PyResult<()> {
    mmn_io::write_tf_checkpoint_arrays(path, &arrays).map_err(mmn_err_to_py)
}

/// Write named arrays as an HDF5 file (h5py/Keras-readable).
#[pyfunction]
pub fn write_h5(path: &str, arrays: Vec<NamedArray>) -> PyResult<()> {
    mmn_io::write_h5_arrays(path, &arrays).map_err(mmn_err_to_py)
}

/// Write named arrays as a `torch.load`-compatible `.pt` state dict.
#[pyfunction]
pub fn write_pt(path: &str, arrays: Vec<NamedArray>) -> PyResult<()> {
    let bytes = write_torch_arrays(&arrays, None).map_err(mmn_err_to_py)?;
    if let Some(parent) = Path::new(path).parent() {
        if !parent.as_os_str().is_empty() {
            std::fs::create_dir_all(parent)
                .map_err(|e| PyValueError::new_err(e.to_string()))?;
        }
    }
    std::fs::write(path, bytes).map_err(|e| PyValueError::new_err(e.to_string()))
}
