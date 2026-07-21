use mmn_data::{DatasetClassification, DatasetMeta, DatasetType};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::errors::{mmn_err_to_py, DataMissingRowError};

/// Labeled text data loaded from a JSON file or an in-memory list of dicts
/// (`data=[{"text": ..., "label": ...}]`).
#[pyclass(name = "DatasetClassification")]
pub struct PyDatasetClassification {
    pub(crate) inner: DatasetClassification,
}

fn classification_samples_from_memory(
    rows: &[HashMap<String, String>],
    text_col: &str,
    tags_col: &str,
) -> PyResult<Vec<(String, String)>> {
    let mut samples = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let text = row.get(text_col).ok_or_else(|| {
            PyErr::new::<DataMissingRowError, _>(format!(
                "data row {i} is missing key {text_col:?}.\nFix: Every dict needs {text_col:?} and {tags_col:?} keys."
            ))
        })?;
        let tag = row.get(tags_col).ok_or_else(|| {
            PyErr::new::<DataMissingRowError, _>(format!(
                "data row {i} is missing key {tags_col:?}.\nFix: Every dict needs {text_col:?} and {tags_col:?} keys."
            ))
        })?;
        samples.push((text.clone(), tag.clone()));
    }
    Ok(samples)
}

#[pymethods]
impl PyDatasetClassification {
    #[new]
    #[pyo3(signature = (file=None, text_col="text", tags_col="label", data=None))]
    pub fn new(
        file: Option<String>,
        text_col: &str,
        tags_col: &str,
        data: Option<Vec<HashMap<String, String>>>,
    ) -> PyResult<Self> {
        match (file, data) {
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "Pass either file=... or data=[...], not both.",
            )),
            (None, None) => Err(PyValueError::new_err(
                "DatasetClassification needs training data.\nFix: Pass file=\"labels.json\" or an in-memory list like data=[{\"text\": \"great\", \"label\": \"positive\"}].",
            )),
            (Some(file), None) => Ok(Self {
                inner: DatasetClassification::load(&file, text_col, tags_col)
                    .map_err(mmn_err_to_py)?,
            }),
            (None, Some(rows)) => {
                let samples = classification_samples_from_memory(&rows, text_col, tags_col)?;
                Ok(Self {
                    inner: DatasetClassification {
                        meta: DatasetMeta {
                            rows: samples.len(),
                            format: "memory".into(),
                            dataset_type: DatasetType::Classification,
                        },
                        samples,
                    },
                })
            }
        }
    }

    #[getter]
    fn rows(&self) -> usize {
        self.inner.meta.rows
    }

    #[getter]
    fn format(&self) -> String {
        self.inner.meta.format.clone()
    }

    #[getter]
    fn type_(&self) -> String {
        "classification".into()
    }

    fn unique_labels(&self) -> Vec<String> {
        self.inner.unique_labels()
    }

    /// Return ``[(text, label), ...]`` for training / hub finetune loops.
    fn as_pairs(&self) -> Vec<(String, String)> {
        self.inner.samples.clone()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetClassification(rows={}, labels={:?}, type='classification')",
            self.inner.meta.rows,
            self.inner.unique_labels()
        )
    }
}
