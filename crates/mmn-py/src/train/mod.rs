// The public training entry points intentionally use Python-style names
// (`Train`, `RL`, `SPIN`) because they are exported verbatim to Python.
#![allow(non_snake_case)]

use mmn_train::{
    rl_with_encoder, spin_with_encoder, train_classifier, train_corpus_with_encoder,
    train_diffusion, train_diffusion_edit, train_with_encoder,
};
use pyo3::prelude::*;

use crate::datasets::{PyDatasetClassification, PyDatasetCorpus, PyDatasetQA};
use crate::encoder_util::resolve_text_encoder;
use crate::errors::{mmn_err_to_py, DataMismatchError};
use crate::models::{PyChatbot, PyClassifier, PyDiffusion};
use crate::tokenizer::{PyBytePairEncoder, PyGpt2BpeEncoder, PyUnigramEncoder};
use crate::train_config::PyTrainConfig;

/// Flush Python's buffered stdout so Rust-side progress lines appear in order.
fn flush_python_stdout(py: Python<'_>) {
    let _ = py
        .import("sys")
        .and_then(|sys| sys.getattr("stdout"))
        .and_then(|out| out.call_method0("flush"));
}

/// Train a Chatbot on QA or corpus data (dispatched by dataset type).
/// Returns one mean-loss value per epoch.
pub(crate) fn train_chatbot_dispatch(
    model: &mut PyChatbot,
    dataset: &Bound<'_, PyAny>,
    config: &mmn_train::TrainConfig,
    enc: Option<mmn_data::TextEncoderRef<'_>>,
) -> PyResult<Vec<f32>> {
    if config.verbose {
        flush_python_stdout(dataset.py());
    }
    if let Ok(ds) = dataset.downcast::<PyDatasetQA>() {
        train_with_encoder(&mut model.inner, &ds.borrow().inner, config, enc)
            .map_err(mmn_err_to_py)
    } else if let Ok(ds) = dataset.downcast::<PyDatasetCorpus>() {
        train_corpus_with_encoder(&mut model.inner, &ds.borrow().inner, config, enc)
            .map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "Train requires DatasetQA or DatasetCorpus.\nFix: Use DatasetQA(file, user_row, ai_row) or DatasetCorpus(rowfile, txtfile).\nExplanation: Classification/image datasets use TrainClassifier or other APIs.".to_string(),
        ))
    }
}

/// Train a Classifier on labeled classification data. Returns per-epoch mean losses.
pub(crate) fn train_classifier_dispatch(
    model: &mut PyClassifier,
    dataset: &Bound<'_, PyAny>,
    config: &mmn_train::TrainConfig,
) -> PyResult<Vec<f32>> {
    if config.verbose {
        flush_python_stdout(dataset.py());
    }
    if let Ok(ds) = dataset.downcast::<PyDatasetClassification>() {
        train_classifier(&mut model.inner, &ds.borrow().inner, config).map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "TrainClassifier requires DatasetClassification.\nFix: Use DatasetClassification(file, text_col, tags_col).\nExplanation: QA/Corpus datasets cannot train a Classifier.".to_string(),
        ))
    }
}

/// Train a Diffusion model on image-gen or image-edit data. Returns per-epoch mean losses.
pub(crate) fn train_diffusion_dispatch(
    model: &mut PyDiffusion,
    dataset: &Bound<'_, PyAny>,
    config: &mmn_train::TrainConfig,
) -> PyResult<Vec<f32>> {
    use crate::datasets::{PyDatasetImageEdit, PyDatasetImageGen};
    if config.verbose {
        flush_python_stdout(dataset.py());
    }
    if let Ok(ds) = dataset.downcast::<PyDatasetImageGen>() {
        train_diffusion(&mut model.inner, &ds.borrow().inner, config).map_err(mmn_err_to_py)
    } else if let Ok(ds) = dataset.downcast::<PyDatasetImageEdit>() {
        train_diffusion_edit(&mut model.inner, &ds.borrow().inner, config).map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "TrainDiffusion requires DatasetImageGen or DatasetImageEdit.\nFix: Use DatasetImageGen(file) or DatasetImageEdit(file) with image rows.\nExplanation: QA/Corpus/Classification datasets cannot train Diffusion.".to_string(),
        ))
    }
}

/// Train `model` on `dataset` and return one mean-loss value per epoch.
#[pyfunction]
#[pyo3(signature = (model, dataset, config, bpe_encoder=None, unigram_encoder=None, gpt2_encoder=None))]
pub fn Train(
    model: &mut PyChatbot,
    dataset: &Bound<'_, PyAny>,
    config: &PyTrainConfig,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
    gpt2_encoder: Option<&PyGpt2BpeEncoder>,
) -> PyResult<Vec<f32>> {
    let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
    train_chatbot_dispatch(model, dataset, &config.to_train_config(), enc)
}

/// Train `model` on labeled data and return one mean-loss value per epoch.
#[pyfunction]
pub fn TrainClassifier(
    model: &mut PyClassifier,
    dataset: &Bound<'_, PyAny>,
    config: &PyTrainConfig,
) -> PyResult<Vec<f32>> {
    train_classifier_dispatch(model, dataset, &config.to_train_config())
}

/// Train `model` on image data and return one mean-loss value per epoch.
#[pyfunction]
pub fn TrainDiffusion(
    model: &mut PyDiffusion,
    dataset: &Bound<'_, PyAny>,
    config: &PyTrainConfig,
) -> PyResult<Vec<f32>> {
    train_diffusion_dispatch(model, dataset, &config.to_train_config())
}

#[pyfunction]
#[pyo3(signature = (model, dataset, train_config, reward_amount, punishment_amount, rl_type="policy", bpe_encoder=None, unigram_encoder=None, gpt2_encoder=None))]
#[allow(clippy::too_many_arguments)]
pub fn RL(
    model: &mut PyChatbot,
    dataset: &Bound<'_, PyAny>,
    train_config: &PyTrainConfig,
    reward_amount: f32,
    punishment_amount: f32,
    rl_type: &str,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
    gpt2_encoder: Option<&PyGpt2BpeEncoder>,
) -> PyResult<()> {
    let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
    if let Ok(ds) = dataset.downcast::<PyDatasetQA>() {
        rl_with_encoder(
            &mut model.inner,
            &ds.borrow().inner,
            &train_config.to_train_config(),
            reward_amount,
            punishment_amount,
            rl_type,
            enc,
        )
        .map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "RL requires DatasetQA.\nFix: Use DatasetQA(file, user_row, ai_row).\nExplanation: RL is wired for QA reward heuristics on Chatbot.".to_string(),
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (model, selfplay_epochs, dataset, bpe_encoder=None, unigram_encoder=None, gpt2_encoder=None))]
pub fn SPIN(
    model: &mut PyChatbot,
    selfplay_epochs: usize,
    dataset: &Bound<'_, PyAny>,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
    gpt2_encoder: Option<&PyGpt2BpeEncoder>,
) -> PyResult<()> {
    let enc = resolve_text_encoder(bpe_encoder, unigram_encoder, gpt2_encoder)?;
    if let Ok(ds) = dataset.downcast::<PyDatasetQA>() {
        spin_with_encoder(
            &mut model.inner,
            selfplay_epochs,
            &ds.borrow().inner,
            enc,
        )
        .map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "SPIN requires DatasetQA.\nFix: Use DatasetQA(file, user_row, ai_row).\nExplanation: SPIN alternates Train+RL on QA samples.".to_string(),
        ))
    }
}
