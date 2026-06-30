use mmn_data::DatasetClassification;
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;

#[pyclass(name = "DatasetClassification")]
pub struct PyDatasetClassification {
    pub(crate) inner: DatasetClassification,
}

#[pymethods]
impl PyDatasetClassification {
    #[new]
    pub fn new(file: String, text_col: &str, tags_col: &str) -> PyResult<Self> {
        Ok(Self {
            inner: DatasetClassification::load(&file, text_col, tags_col).map_err(mmn_err_to_py)?,
        })
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

    fn __repr__(&self) -> String {
        format!(
            "DatasetClassification(rows={}, labels={:?}, type='classification')",
            self.inner.meta.rows,
            self.inner.unique_labels()
        )
    }
}
