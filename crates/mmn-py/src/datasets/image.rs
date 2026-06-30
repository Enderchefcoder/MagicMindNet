use mmn_data::{DatasetImageEdit, DatasetImageGen};
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;

#[pyclass(name = "DatasetImageGen")]
pub struct PyDatasetImageGen {
    pub(crate) inner: DatasetImageGen,
}

#[pymethods]
impl PyDatasetImageGen {
    #[new]
    pub fn new(file: String) -> PyResult<Self> {
        Ok(Self {
            inner: DatasetImageGen::load(&file).map_err(mmn_err_to_py)?,
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
        "image_gen".into()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetImageGen(rows={}, format={:?}, type='image_gen')",
            self.inner.meta.rows, self.inner.meta.format
        )
    }
}

#[pyclass(name = "DatasetImageEdit")]
pub struct PyDatasetImageEdit {
    pub(crate) inner: DatasetImageEdit,
}

#[pymethods]
impl PyDatasetImageEdit {
    #[new]
    pub fn new(file: String) -> PyResult<Self> {
        Ok(Self {
            inner: DatasetImageEdit::load(&file).map_err(mmn_err_to_py)?,
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
        "image_edit".into()
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetImageEdit(rows={}, format={:?}, type='image_edit')",
            self.inner.meta.rows, self.inner.meta.format
        )
    }
}
