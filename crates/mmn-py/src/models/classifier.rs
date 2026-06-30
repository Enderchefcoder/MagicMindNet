use mmn_models::Classifier;
use mmn_train::mean_classification_loss;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyDict;

use crate::datasets::PyDatasetClassification;
use crate::errors::{mmn_err_to_py, DataMismatchError};

#[pyclass(name = "Classifier")]
pub struct PyClassifier {
    pub(crate) inner: Classifier,
}

#[pymethods]
impl PyClassifier {
    #[new]
    #[pyo3(signature = (num_labels, input_dim, seed=None))]
    pub fn new(num_labels: usize, input_dim: usize, seed: Option<u64>) -> Self {
        let labels = (0..num_labels).map(|i| format!("class_{i}")).collect();
        Self {
            inner: Classifier::with_labels_seed(labels, input_dim, seed),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (labels, input_dim, seed=None))]
    fn with_labels(labels: Vec<String>, input_dim: usize, seed: Option<u64>) -> Self {
        Self {
            inner: Classifier::with_labels_seed(labels, input_dim, seed),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (ds, input_dim, seed=None))]
    fn from_classification(
        ds: &PyDatasetClassification,
        input_dim: usize,
        seed: Option<u64>,
    ) -> Self {
        Self {
            inner: Classifier::from_classification_dataset_seed(&ds.inner, input_dim, seed),
        }
    }

    #[getter]
    fn labels(&self) -> Vec<String> {
        self.inner.labels.clone()
    }

    #[getter]
    fn init_seed(&self) -> Option<u64> {
        self.inner.init_seed
    }

    #[getter]
    fn input_dim(&self) -> usize {
        self.inner.input_dim
    }

    #[getter]
    fn num_labels(&self) -> usize {
        self.inner.labels.len()
    }

    fn __repr__(&self) -> String {
        match self.inner.init_seed {
            Some(seed) => format!(
                "Classifier(labels={:?}, input_dim={}, init_seed={seed})",
                self.inner.labels,
                self.inner.input_dim
            ),
            None => format!(
                "Classifier(labels={:?}, input_dim={})",
                self.inner.labels,
                self.inner.input_dim
            ),
        }
    }

    fn predict(&self, text: &str, py: Python<'_>) -> PyResult<PyObject> {
        let m = self.inner.predict_text(text).map_err(mmn_err_to_py)?;
        let dict = PyDict::new(py);
        for (k, v) in m {
            dict.set_item(k, v)?;
        }
        Ok(dict.into())
    }

    fn compute_loss(&self, text: &str, label: &str) -> PyResult<f32> {
        let idx = self
            .inner
            .label_index(label)
            .ok_or_else(|| PyValueError::new_err(format!("unknown label: {label}")))?;
        self.inner.loss_on_label(text, idx).map_err(mmn_err_to_py)
    }

    /// Mean CE over all rows in a `DatasetClassification` (skips unknown tags).
    fn compute_mean_loss(&self, dataset: &Bound<'_, PyAny>) -> PyResult<f32> {
        if let Ok(ds) = dataset.downcast::<PyDatasetClassification>() {
            return mean_classification_loss(&self.inner, &ds.borrow().inner).map_err(mmn_err_to_py);
        }
        Err(PyErr::new::<DataMismatchError, _>(
            "compute_mean_loss on Classifier requires DatasetClassification.\nFix: Use DatasetClassification(file, text_col, tags_col).\nExplanation: QA datasets use Chatbot.compute_mean_loss.".to_string(),
        ))
    }
}
