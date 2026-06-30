use mmn_io::{
    export_bin, export_classifier, export_safetensors, import_bin, import_classifier,
    import_safetensors, merge_classifiers, merge_models, quantize_classifier, quantize_model,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;
use crate::models::{PyChatbot, PyClassifier};

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
pub fn export(model: &PyChatbot, format: &str, path: &str) -> PyResult<()> {
    match format {
        "safetensors" => export_safetensors(&model.inner, path).map_err(mmn_err_to_py),
        "bin" => export_bin(&model.inner, path).map_err(mmn_err_to_py),
        _ => Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    }
}

#[pyfunction]
pub fn import_model(format: &str, files: Vec<String>) -> PyResult<PyChatbot> {
    let path = files.first().ok_or_else(|| PyValueError::new_err("files required"))?;
    let m = match format {
        "safetensors" => import_safetensors(path, 0).map_err(mmn_err_to_py)?,
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
        _ => return Err(PyValueError::new_err(format!("Unknown format: {format}"))),
    };
    Ok(PyClassifier { inner: m })
}

#[pyfunction]
pub fn quantize_classifier_model(model: &mut PyClassifier, quant: &str) -> PyResult<()> {
    quantize_classifier(&mut model.inner, quant).map_err(mmn_err_to_py)
}
