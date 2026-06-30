use mmn_data::{DatasetQA, DatasetQAConfig};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;

#[pyclass(name = "DatasetQA")]
pub struct PyDatasetQA {
    pub(crate) inner: DatasetQA,
}

#[pymethods]
impl PyDatasetQA {
    #[new]
    #[pyo3(signature = (file, user_row="input", ai_row="output", system_row=None, multipleturn=true, tokenizer="ChatXML", cot=true, thinktag=""))]
    pub fn new(
        file: String,
        user_row: &str,
        ai_row: &str,
        system_row: Option<String>,
        multipleturn: bool,
        tokenizer: &str,
        cot: bool,
        thinktag: &str,
    ) -> PyResult<Self> {
        let _ = (multipleturn, tokenizer);
        let inner = DatasetQA::load(DatasetQAConfig {
            file,
            user_row: user_row.to_string(),
            ai_row: ai_row.to_string(),
            system_row,
            multiple_turn: multipleturn,
            thinktag: thinktag.to_string(),
            cot,
        })
        .map_err(mmn_err_to_py)?;
        Ok(Self { inner })
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

    fn __repr__(&self) -> String {
        format!(
            "DatasetQA(rows={}, format={:?}, type='qa')",
            self.inner.meta.rows, self.inner.meta.format
        )
    }
}
