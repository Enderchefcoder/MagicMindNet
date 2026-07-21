use mmn_data::BytePairEncoder;
use mmn_data::Gpt2BpeEncoder;
use mmn_data::UnigramEncoder;
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::fs;

use crate::datasets::{PyDatasetCorpus, PyDatasetQA};
use crate::errors::mmn_err_to_py;

#[pyclass(name = "BytePairEncoder")]
pub struct PyBytePairEncoder {
    pub inner: BytePairEncoder,
}

#[pyclass(name = "UnigramEncoder")]
pub struct PyUnigramEncoder {
    pub inner: UnigramEncoder,
}

/// GPT-2-style byte-level BPE over an external vocabulary (GGUF gpt2 vocabs,
/// HF vocab.json + merges.txt). Ids follow the vocabulary order.
#[pyclass(name = "Gpt2BpeEncoder")]
pub struct PyGpt2BpeEncoder {
    pub inner: Gpt2BpeEncoder,
}

#[pymethods]
impl PyGpt2BpeEncoder {
    /// Build from a token list (id order) and "left right" merge rules.
    #[staticmethod]
    #[pyo3(signature = (tokens, merges=Vec::new()))]
    fn from_vocab(tokens: Vec<String>, merges: Vec<String>) -> PyResult<Self> {
        Ok(Self {
            inner: Gpt2BpeEncoder::from_vocab(tokens, &merges).map_err(mmn_err_to_py)?,
        })
    }

    fn encode(&self, text: &str) -> Vec<usize> {
        self.inner.encode(text)
    }

    fn decode(&self, ids: Vec<usize>) -> String {
        self.inner.decode(&ids)
    }

    /// Token string (byte-unicode form) for an id.
    fn token(&self, id: usize) -> PyResult<String> {
        self.inner
            .token(id)
            .map(str::to_string)
            .ok_or_else(|| PyValueError::new_err(format!("token id {id} out of range")))
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    /// Load from a HuggingFace `tokenizer.json` file (BPE model).
    ///
    /// Supports vocab as a `{"token": id}` dict and merges as either a list
    /// of `"left right"` strings or a list of `["left", "right"]` arrays.
    #[staticmethod]
    fn from_hf_tokenizer_json(path: &str) -> PyResult<Self> {
        let text = fs::read_to_string(path)
            .map_err(|e| PyValueError::new_err(format!("cannot read {path}: {e}")))?;
        let root: serde_json::Value = serde_json::from_str(&text)
            .map_err(|e| PyValueError::new_err(format!("JSON parse error in {path}: {e}")))?;

        let model = root
            .get("model")
            .ok_or_else(|| PyValueError::new_err("tokenizer.json missing 'model' key"))?;

        // Vocabulary: {"token": id, ...} — sort by id to build ordered token list.
        let vocab_obj = model
            .get("vocab")
            .ok_or_else(|| PyValueError::new_err("tokenizer.json model missing 'vocab'"))?
            .as_object()
            .ok_or_else(|| PyValueError::new_err("tokenizer.json 'vocab' must be an object"))?;

        let mut id_token: Vec<(usize, String)> = vocab_obj
            .iter()
            .map(|(tok, id_val)| {
                let id = id_val
                    .as_u64()
                    .ok_or_else(|| {
                        PyValueError::new_err(format!("vocab id for {tok:?} is not an integer"))
                    })
                    .map(|v| v as usize)?;
                Ok((id, tok.clone()))
            })
            .collect::<PyResult<_>>()?;
        id_token.sort_by_key(|(id, _)| *id);

        // Fill gaps with empty strings so ids are contiguous.
        let max_id = id_token.last().map(|(id, _)| *id).unwrap_or(0);
        let mut tokens: Vec<String> = vec![String::new(); max_id + 1];
        for (id, tok) in id_token {
            tokens[id] = tok;
        }

        // Merges: list of "left right" strings or ["left", "right"] arrays.
        let merges_val = model
            .get("merges")
            .ok_or_else(|| PyValueError::new_err("tokenizer.json model missing 'merges'"))?;
        let merges_arr = merges_val
            .as_array()
            .ok_or_else(|| PyValueError::new_err("tokenizer.json 'merges' must be an array"))?;

        let mut merges: Vec<String> = Vec::with_capacity(merges_arr.len());
        for (i, item) in merges_arr.iter().enumerate() {
            let rule = if let Some(s) = item.as_str() {
                s.to_string()
            } else if let Some(arr) = item.as_array() {
                if arr.len() != 2 {
                    return Err(PyValueError::new_err(format!(
                        "merges[{i}] array must have exactly 2 elements"
                    )));
                }
                let left = arr[0].as_str().ok_or_else(|| {
                    PyValueError::new_err(format!("merges[{i}][0] must be a string"))
                })?;
                let right = arr[1].as_str().ok_or_else(|| {
                    PyValueError::new_err(format!("merges[{i}][1] must be a string"))
                })?;
                format!("{left} {right}")
            } else {
                return Err(PyValueError::new_err(format!(
                    "merges[{i}] must be a string or a [left, right] array"
                )));
            };
            merges.push(rule);
        }

        Ok(Self {
            inner: Gpt2BpeEncoder::from_vocab(tokens, &merges).map_err(mmn_err_to_py)?,
        })
    }

    fn __repr__(&self) -> String {
        format!("Gpt2BpeEncoder(vocab_size={})", self.inner.vocab_size())
    }
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

    fn prune_pieces_below_logprob(&mut self, min_log_prob: f32) -> PyResult<()> {
        self.inner.prune_pieces_below_logprob(min_log_prob);
        Ok(())
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
