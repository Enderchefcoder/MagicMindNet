use mmn_data::BytePairEncoder;
use mmn_data::UnigramEncoder;
use pyo3::prelude::*;

use crate::datasets::{PyDatasetCorpus, PyDatasetQA};

#[pyclass(name = "BytePairEncoder")]
pub struct PyBytePairEncoder {
    pub inner: BytePairEncoder,
}

#[pyclass(name = "UnigramEncoder")]
pub struct PyUnigramEncoder {
    pub inner: UnigramEncoder,
}

#[pymethods]
impl PyUnigramEncoder {
    #[staticmethod]
    #[pyo3(signature = (texts, vocab_size=512))]
    fn train(texts: Vec<String>, vocab_size: usize) -> Self {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Self {
            inner: UnigramEncoder::train(&refs, vocab_size),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (dataset, vocab_size=512))]
    fn train_from_qa(dataset: &PyDatasetQA, vocab_size: usize) -> Self {
        let texts: Vec<String> = dataset
            .inner
            .samples
            .iter()
            .flat_map(|s| vec![s.input.clone(), s.output.clone()])
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Self {
            inner: UnigramEncoder::train(&refs, vocab_size),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (dataset, vocab_size=512))]
    fn train_from_corpus(dataset: &PyDatasetCorpus, vocab_size: usize) -> Self {
        let texts: Vec<String> = dataset.inner.rows.iter().map(|r| r.text.clone()).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Self {
            inner: UnigramEncoder::train(&refs, vocab_size),
        }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner
            .export_json(path)
            .map_err(crate::errors::mmn_err_to_py)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = UnigramEncoder::import_json(path).map_err(crate::errors::mmn_err_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn piece_count(&self) -> usize {
        self.inner.piece_count()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}

#[pymethods]
impl PyBytePairEncoder {
    #[staticmethod]
    #[pyo3(signature = (texts, vocab_size=512, num_merges=32))]
    fn train(texts: Vec<String>, vocab_size: usize, num_merges: usize) -> Self {
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Self {
            inner: BytePairEncoder::train(&refs, vocab_size, num_merges),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (dataset, vocab_size=512, num_merges=32))]
    fn train_from_qa(dataset: &PyDatasetQA, vocab_size: usize, num_merges: usize) -> Self {
        let texts: Vec<String> = dataset
            .inner
            .samples
            .iter()
            .flat_map(|s| vec![s.input.clone(), s.output.clone()])
            .collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Self {
            inner: BytePairEncoder::train(&refs, vocab_size, num_merges),
        }
    }

    #[staticmethod]
    #[pyo3(signature = (dataset, vocab_size=512, num_merges=32))]
    fn train_from_corpus(dataset: &PyDatasetCorpus, vocab_size: usize, num_merges: usize) -> Self {
        let texts: Vec<String> = dataset.inner.rows.iter().map(|r| r.text.clone()).collect();
        let refs: Vec<&str> = texts.iter().map(String::as_str).collect();
        Self {
            inner: BytePairEncoder::train(&refs, vocab_size, num_merges),
        }
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }

    fn save(&self, path: &str) -> PyResult<()> {
        self.inner.export_json(path).map_err(crate::errors::mmn_err_to_py)
    }

    #[staticmethod]
    fn load(path: &str) -> PyResult<Self> {
        let inner = BytePairEncoder::import_json(path).map_err(crate::errors::mmn_err_to_py)?;
        Ok(Self { inner })
    }

    #[getter]
    fn merge_count(&self) -> usize {
        self.inner.merge_count()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
}
