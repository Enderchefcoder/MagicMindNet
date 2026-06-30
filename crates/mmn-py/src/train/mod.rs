use mmn_train::{rl, spin, train_classifier, train_corpus_with_bpe, train_with_bpe};
use pyo3::prelude::*;

use crate::datasets::{PyDatasetClassification, PyDatasetCorpus, PyDatasetQA};
use crate::errors::{mmn_err_to_py, DataMismatchError};
use crate::models::{PyChatbot, PyClassifier};
use crate::tokenizer::PyBytePairEncoder;
use crate::train_config::PyTrainConfig;

#[pyfunction]
#[pyo3(signature = (model, dataset, config, bpe_encoder=None))]
pub fn Train(
    model: &mut PyChatbot,
    dataset: &Bound<'_, PyAny>,
    config: &PyTrainConfig,
    bpe_encoder: Option<&PyBytePairEncoder>,
) -> PyResult<()> {
    let bpe = bpe_encoder.map(|e| &e.inner);
    if let Ok(ds) = dataset.downcast::<PyDatasetQA>() {
        train_with_bpe(
            &mut model.inner,
            &ds.borrow().inner,
            &config.to_train_config(),
            bpe,
        )
        .map_err(mmn_err_to_py)
    } else if let Ok(ds) = dataset.downcast::<PyDatasetCorpus>() {
        train_corpus_with_bpe(
            &mut model.inner,
            &ds.borrow().inner,
            &config.to_train_config(),
            bpe,
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
#[pyo3(signature = (model, dataset, train_config, reward_amount, punishment_amount, rl_type="policy"))]
pub fn RL(
    model: &mut PyChatbot,
    dataset: &Bound<'_, PyAny>,
    train_config: &PyTrainConfig,
    reward_amount: f32,
    punishment_amount: f32,
    rl_type: &str,
) -> PyResult<()> {
    if let Ok(ds) = dataset.downcast::<PyDatasetQA>() {
        rl(
            &mut model.inner,
            &ds.borrow().inner,
            &train_config.to_train_config(),
            reward_amount,
            punishment_amount,
            rl_type,
        )
        .map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "RL requires DatasetQA.\nFix: Use DatasetQA(file, user_row, ai_row).\nExplanation: RL is wired for QA reward heuristics on Chatbot.".to_string(),
        ))
    }
}

#[pyfunction]
pub fn SPIN(
    model: &mut PyChatbot,
    selfplay_epochs: usize,
    dataset: &Bound<'_, PyAny>,
) -> PyResult<()> {
    if let Ok(ds) = dataset.downcast::<PyDatasetQA>() {
        spin(
            &mut model.inner,
            selfplay_epochs,
            &ds.borrow().inner,
        )
        .map_err(mmn_err_to_py)
    } else {
        Err(PyErr::new::<DataMismatchError, _>(
            "SPIN requires DatasetQA.\nFix: Use DatasetQA(file, user_row, ai_row).\nExplanation: SPIN alternates Train+RL on QA samples.".to_string(),
        ))
    }
}
