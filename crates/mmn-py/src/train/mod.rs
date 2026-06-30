use mmn_train::{
    rl_with_encoder, spin_with_encoder, train_classifier, train_corpus_with_encoder,
    train_diffusion, train_diffusion_edit, train_with_encoder,
};
use pyo3::prelude::*;

use crate::datasets::{PyDatasetClassification, PyDatasetCorpus, PyDatasetQA};
use crate::encoder_util::resolve_text_encoder;
use crate::errors::{mmn_err_to_py, DataMismatchError};
use crate::models::{PyChatbot, PyClassifier, PyDiffusion};
use crate::tokenizer::{PyBytePairEncoder, PyUnigramEncoder};
use crate::train_config::PyTrainConfig;

#[pyfunction]
#[pyo3(signature = (model, dataset, config, bpe_encoder=None, unigram_encoder=None))]
pub fn Train(
    model: &mut PyChatbot,
    dataset: &Bound<'_, PyAny>,
    config: &PyTrainConfig,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
) -> PyResult<()> {
    let enc = resolve_text_encoder(bpe_encoder, unigram_encoder)?;
    if let Ok(ds) = dataset.downcast::<PyDatasetQA>() {
        train_with_encoder(
            &mut model.inner,
            &ds.borrow().inner,
            &config.to_train_config(),
            enc,
        )
        .map_err(mmn_err_to_py)
    } else if let Ok(ds) = dataset.downcast::<PyDatasetCorpus>() {
        train_corpus_with_encoder(
            &mut model.inner,
            &ds.borrow().inner,
            &config.to_train_config(),
            enc,
        )
        .map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "Train requires DatasetQA or DatasetCorpus.\nFix: Use DatasetQA(file, user_row, ai_row) or DatasetCorpus(rowfile, txtfile).\nExplanation: Classification/image datasets use TrainClassifier or other APIs.".to_string(),
        ))
    }
}

#[pyfunction]
pub fn TrainClassifier(
    model: &mut PyClassifier,
    dataset: &Bound<'_, PyAny>,
    config: &PyTrainConfig,
) -> PyResult<()> {
    if let Ok(ds) = dataset.downcast::<PyDatasetClassification>() {
        train_classifier(&mut model.inner, &ds.borrow().inner, &config.to_train_config())
            .map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "TrainClassifier requires DatasetClassification.\nFix: Use DatasetClassification(file, text_col, tags_col).\nExplanation: QA/Corpus datasets cannot train a Classifier.".to_string(),
        ))
    }
}

#[pyfunction]
pub fn TrainDiffusion(
    model: &mut PyDiffusion,
    dataset: &Bound<'_, PyAny>,
    config: &PyTrainConfig,
) -> PyResult<()> {
    use crate::datasets::{PyDatasetImageEdit, PyDatasetImageGen};
    if let Ok(ds) = dataset.downcast::<PyDatasetImageGen>() {
        train_diffusion(
            &mut model.inner,
            &ds.borrow().inner,
            &config.to_train_config(),
        )
        .map_err(mmn_err_to_py)
    } else if let Ok(ds) = dataset.downcast::<PyDatasetImageEdit>() {
        train_diffusion_edit(
            &mut model.inner,
            &ds.borrow().inner,
            &config.to_train_config(),
        )
        .map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "TrainDiffusion requires DatasetImageGen or DatasetImageEdit.\nFix: Use DatasetImageGen(file) or DatasetImageEdit(file) with image rows.\nExplanation: QA/Corpus/Classification datasets cannot train Diffusion.".to_string(),
        ))
    }
}

#[pyfunction]
#[pyo3(signature = (model, dataset, train_config, reward_amount, punishment_amount, rl_type="policy", bpe_encoder=None, unigram_encoder=None))]
pub fn RL(
    model: &mut PyChatbot,
    dataset: &Bound<'_, PyAny>,
    train_config: &PyTrainConfig,
    reward_amount: f32,
    punishment_amount: f32,
    rl_type: &str,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
) -> PyResult<()> {
    let enc = resolve_text_encoder(bpe_encoder, unigram_encoder)?;
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
#[pyo3(signature = (model, selfplay_epochs, dataset, bpe_encoder=None, unigram_encoder=None))]
pub fn SPIN(
    model: &mut PyChatbot,
    selfplay_epochs: usize,
    dataset: &Bound<'_, PyAny>,
    bpe_encoder: Option<&PyBytePairEncoder>,
    unigram_encoder: Option<&PyUnigramEncoder>,
) -> PyResult<()> {
    let enc = resolve_text_encoder(bpe_encoder, unigram_encoder)?;
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
