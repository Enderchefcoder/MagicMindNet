use mmn_io::{
    export_bin, export_classifier, export_diffusion, export_hf_classifier_safetensors,
    export_hf_safetensors, export_safetensors, import_bin, import_classifier, import_diffusion,
    merge_diffusion,
    import_hf_classifier_safetensors, import_hf_safetensors, import_safetensors, merge_classifiers,
    merge_models, quantize_classifier, quantize_model, TokenizerSidecarRefs,
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

#[pyfunction]
#[pyo3(signature = (model, format, path, bpe_encoder=None, unigram_encoder=None))]
pub fn export(
    model: &PyChatbot,
    format: &str,
    path: &str,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
) -> PyResult<()> {
    let (bpe_rel, uni_rel) = write_tokenizer_sidecars(path, bpe_encoder, unigram_encoder)?;
    let sidecars = TokenizerSidecarRefs {
        bpe: bpe_rel.as_deref(),
        unigram: uni_rel.as_deref(),
    };
    match format {
        "safetensors" => export_safetensors(&model.inner, path, sidecars).map_err(mmn_err_to_py),
        "hf-safetensors" | "hf_safetensors" => {
            export_hf_safetensors(&model.inner, path, sidecars).map_err(mmn_err_to_py)
        }
        "bin" => {
            if bpe_encoder.is_some() || unigram_encoder.is_some() {
                return Err(PyValueError::new_err(
                    "bpe_encoder / unigram_encoder are only supported with safetensors export",
                ));
            }
            export_bin(&model.inner, path).map_err(mmn_err_to_py)
        }
        _ => Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    }
}

#[pyfunction]
pub fn import_model(format: &str, files: Vec<String>) -> PyResult<PyChatbot> {
    let path = files.first().ok_or_else(|| PyValueError::new_err("files required"))?;
    let m = match format {
        "safetensors" => import_safetensors(path, 0).map_err(mmn_err_to_py)?,
        "hf-safetensors" | "hf_safetensors" => import_hf_safetensors(path).map_err(mmn_err_to_py)?,
        "bin" => import_bin(path).map_err(mmn_err_to_py)?,
        _ => return Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    };
    Ok(PyChatbot { inner: m })
}

#[pyfunction]
pub fn quantize(model: &mut PyChatbot, quant: &str) -> PyResult<()> {
    quantize_model(&mut model.inner, quant).map_err(mmn_err_to_py)
}

#[pyfunction]
pub fn export_classifier_model(model: &PyClassifier, format: &str, path: &str) -> PyResult<()> {
    match format {
        "safetensors" => export_classifier(&model.inner, path).map_err(mmn_err_to_py),
        "hf-safetensors" | "hf_safetensors" => {
            export_hf_classifier_safetensors(&model.inner, path).map_err(mmn_err_to_py)
        }
        _ => Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    }
}

#[pyfunction]
pub fn import_classifier_model(format: &str, files: Vec<String>) -> PyResult<PyClassifier> {
    let path = files
        .first()
        .ok_or_else(|| PyValueError::new_err("files required"))?;
    let m = match format {
        "safetensors" => import_classifier(path).map_err(mmn_err_to_py)?,
        "hf-safetensors" | "hf_safetensors" => {
            import_hf_classifier_safetensors(path).map_err(mmn_err_to_py)?
        }
        _ => return Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    };
    Ok(PyClassifier { inner: m })
}

#[pyfunction]
pub fn quantize_classifier_model(model: &mut PyClassifier, quant: &str) -> PyResult<()> {
    quantize_classifier(&mut model.inner, quant).map_err(mmn_err_to_py)
}

#[pyfunction]
pub fn export_diffusion_model(model: &PyDiffusion, format: &str, path: &str) -> PyResult<()> {
    match format {
        "safetensors" => export_diffusion(&model.inner, path).map_err(mmn_err_to_py),
        _ => Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    }
}

#[pyfunction]
pub fn import_diffusion_model(format: &str, files: Vec<String>) -> PyResult<PyDiffusion> {
    let path = files
        .first()
        .ok_or_else(|| PyValueError::new_err("files required"))?;
    let m = match format {
        "safetensors" => import_diffusion(path).map_err(mmn_err_to_py)?,
        _ => return Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    };
    Ok(PyDiffusion { inner: m })
}

#[pyfunction]
pub fn merge_diffusion_model(model1: &PyDiffusion, model2: &PyDiffusion) -> PyResult<PyDiffusion> {
    let m = merge_diffusion(&model1.inner, &model2.inner).map_err(mmn_err_to_py)?;
    Ok(PyDiffusion { inner: m })
}
