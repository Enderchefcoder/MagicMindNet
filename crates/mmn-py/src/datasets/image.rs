use mmn_data::{DatasetImageEdit, DatasetImageGen};
use pyo3::exceptions::PyIndexError;
use pyo3::prelude::*;

use crate::errors::mmn_err_to_py;

fn path_to_string(path: std::path::PathBuf) -> String {
    path.to_string_lossy().into_owned()
}

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

    /// Resolve a manifest-relative image path to an absolute filesystem path.
    fn resolve_image_path(&self, rel: String) -> String {
        path_to_string(self.inner.resolve_image_path(&rel))
    }

    /// Absolute path to row `index`'s `image` field.
    fn image_path_at(&self, index: usize) -> PyResult<String> {
        let sample = self.inner.samples.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "image index {index} out of range (rows={})",
                self.inner.meta.rows
            ))
        })?;
        Ok(path_to_string(
            self.inner.resolve_image_path(&sample.image_path),
        ))
    }

    /// Prompt string for row `index`.
    fn prompt_at(&self, index: usize) -> PyResult<String> {
        let sample = self.inner.samples.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "image index {index} out of range (rows={})",
                self.inner.meta.rows
            ))
        })?;
        Ok(sample.prompt.clone())
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

    fn resolve_image_path(&self, rel: String) -> String {
        path_to_string(self.inner.resolve_image_path(&rel))
    }

    fn resolve_mask_path(&self, rel: String) -> String {
        path_to_string(self.inner.resolve_mask_path(&rel))
    }

    fn image_path_at(&self, index: usize) -> PyResult<String> {
        let sample = self.inner.samples.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "image index {index} out of range (rows={})",
                self.inner.meta.rows
            ))
        })?;
        Ok(path_to_string(self.inner.resolve_image_path(&sample.image)))
    }

    fn mask_path_at(&self, index: usize) -> PyResult<String> {
        let sample = self.inner.samples.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "image index {index} out of range (rows={})",
                self.inner.meta.rows
            ))
        })?;
        Ok(path_to_string(
            self.inner.resolve_mask_path(&sample.mask_image),
        ))
    }

    fn prompt_at(&self, index: usize) -> PyResult<String> {
        let sample = self.inner.samples.get(index).ok_or_else(|| {
            PyIndexError::new_err(format!(
                "image index {index} out of range (rows={})",
                self.inner.meta.rows
            ))
        })?;
        Ok(sample.prompt.clone())
    }

    fn __repr__(&self) -> String {
        format!(
            "DatasetImageEdit(rows={}, format={:?}, type='image_edit')",
            self.inner.meta.rows, self.inner.meta.format
        )
    }
}
