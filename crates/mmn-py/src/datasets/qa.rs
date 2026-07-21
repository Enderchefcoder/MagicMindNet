use mmn_data::{ChatXmlConfig, DatasetMeta, DatasetQA, DatasetQAConfig, DatasetType, QaSample};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use crate::errors::{mmn_err_to_py, DataMissingRowError};

/// Question/answer training data loaded from JSON, JSONL, Parquet, or an
/// in-memory list of dicts (`data=[{"input": ..., "output": ...}]`).
#[pyclass(name = "DatasetQA")]
pub struct PyDatasetQA {
    pub(crate) inner: DatasetQA,
}

fn qa_samples_from_memory(
    rows: &[HashMap<String, String>],
    user_row: &str,
    ai_row: &str,
    system_row: Option<&str>,
    image_row: Option<&str>,
) -> PyResult<Vec<QaSample>> {
    let mut samples = Vec::with_capacity(rows.len());
    for (i, row) in rows.iter().enumerate() {
        let input = row.get(user_row).ok_or_else(|| {
            PyErr::new::<DataMissingRowError, _>(format!(
                "data row {i} is missing key {user_row:?}.\nFix: Every dict needs {user_row:?} and {ai_row:?} keys."
            ))
        })?;
        let output = row.get(ai_row).ok_or_else(|| {
            PyErr::new::<DataMissingRowError, _>(format!(
                "data row {i} is missing key {ai_row:?}.\nFix: Every dict needs {user_row:?} and {ai_row:?} keys."
            ))
        })?;
        let system = system_row.and_then(|k| row.get(k).cloned());
        let image_paths = image_row
            .and_then(|k| row.get(k).cloned())
            .map(|p| vec![p])
            .unwrap_or_default();
        samples.push(QaSample {
            input: input.clone(),
            output: output.clone(),
            system,
            image_paths,
        });
    }
    Ok(samples)
}

#[pymethods]
impl PyDatasetQA {
    #[new]
    #[pyo3(signature = (file=None, user_row="input", ai_row="output", system_row=None, image_row="image", vision_patch_grid=1, multipleturn=true, tokenizer="ChatXML", cot=true, thinktag="", data=None))]
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        file: Option<String>,
        user_row: &str,
        ai_row: &str,
        system_row: Option<String>,
        image_row: &str,
        vision_patch_grid: usize,
        multipleturn: bool,
        tokenizer: &str,
        cot: bool,
        thinktag: &str,
        data: Option<Vec<HashMap<String, String>>>,
    ) -> PyResult<Self> {
        let _ = (multipleturn, tokenizer);
        let image_row = if image_row.is_empty() {
            None
        } else {
            Some(image_row.to_string())
        };
        match (file, data) {
            (Some(_), Some(_)) => Err(PyValueError::new_err(
                "Pass either file=... or data=[...], not both.",
            )),
            (None, None) => Err(PyValueError::new_err(
                "DatasetQA needs training data.\nFix: Pass file=\"qa.json\" or an in-memory list like data=[{\"input\": \"hi\", \"output\": \"hello\"}].",
            )),
            (Some(file), None) => {
                let inner = DatasetQA::load(DatasetQAConfig {
                    file,
                    user_row: user_row.to_string(),
                    ai_row: ai_row.to_string(),
                    system_row,
                    image_row,
                    vision_patch_grid,
                    multiple_turn: multipleturn,
                    thinktag: thinktag.to_string(),
                    cot,
                })
                .map_err(mmn_err_to_py)?;
                Ok(Self { inner })
            }
            (None, Some(rows)) => {
                let samples = qa_samples_from_memory(
                    &rows,
                    user_row,
                    ai_row,
                    system_row.as_deref(),
                    image_row.as_deref(),
                )?;
                let inner = DatasetQA {
                    meta: DatasetMeta {
                        rows: samples.len(),
                        format: "memory".into(),
                        dataset_type: DatasetType::Qa,
                    },
                    samples,
                    chatxml: ChatXmlConfig::from_thinktag(thinktag, cot),
                    source_dir: std::env::current_dir()
                        .unwrap_or_else(|_| std::path::PathBuf::from(".")),
                    vision_patch_grid: vision_patch_grid.max(1),
                };
                Ok(Self { inner })
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
        "qa".into()
    }

    /// Return ``[(input, output), ...]`` for hub finetune / custom loops.
    fn as_pairs(&self) -> Vec<(String, String)> {
        self.inner
            .samples
            .iter()
            .map(|s| (s.input.clone(), s.output.clone()))
            .collect()
    }

    fn format_sample(&self, index: usize) -> PyResult<String> {
        let s = self
            .inner
            .samples
            .get(index)
            .ok_or_else(|| PyValueError::new_err("sample index out of range"))?;
        let turns = vec![(s.input.clone(), s.output.clone())];
        Ok(self
            .inner
            .chatxml
            .format_conversation(s.system.as_deref(), &turns))
    }

    fn sample_image_path(&self, index: usize) -> PyResult<Option<String>> {
        Ok(self
            .inner
            .samples
            .get(index)
            .and_then(|s| s.image_paths.first().cloned()))
    }

    fn sample_image_paths(&self, index: usize) -> PyResult<Vec<String>> {
        Ok(self
            .inner
            .samples
            .get(index)
            .map(|s| s.image_paths.clone())
            .unwrap_or_default())
    }

    #[getter]
    fn vision_patch_grid(&self) -> usize {
        self.inner.vision_patch_grid
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetQA(rows={}, format={:?}, type='qa')",
            self.inner.meta.rows, self.inner.meta.format
        )
    }
}
