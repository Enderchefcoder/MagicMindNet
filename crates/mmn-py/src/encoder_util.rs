use mmn_data::TextEncoderRef;
use pyo3::prelude::*;

use crate::errors::DataMismatchError;
use crate::tokenizer::{PyBytePairEncoder, PyUnigramEncoder};

pub fn resolve_text_encoder<'a>(
    bpe_encoder: Option<&'a PyBytePairEncoder>,
    unigram_encoder: Option<&'a PyUnigramEncoder>,
) -> PyResult<Option<TextEncoderRef<'a>>> {
    match (bpe_encoder, unigram_encoder) {
        (Some(_), Some(_)) => Err(PyErr::new::<DataMismatchError, _>(
            "Pass only one of bpe_encoder or unigram_encoder.\nFix: Supply a single tokenizer.\nExplanation: Training and generation accept one trained encoder at a time.".to_string(),
        )),
        (Some(b), None) => Ok(Some(TextEncoderRef::Bpe(&b.inner))),
        (None, Some(u)) => Ok(Some(TextEncoderRef::Unigram(&u.inner))),
        (None, None) => Ok(None),
    }
}
