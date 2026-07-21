use mmn_data::TextEncoderRef;
use pyo3::prelude::*;

use crate::errors::DataMismatchError;
use crate::tokenizer::{PyBytePairEncoder, PyGpt2BpeEncoder, PyUnigramEncoder};

pub fn resolve_text_encoder<'a>(
    bpe_encoder: Option<&'a PyBytePairEncoder>,
    unigram_encoder: Option<&'a PyUnigramEncoder>,
    gpt2_encoder: Option<&'a PyGpt2BpeEncoder>,
) -> PyResult<Option<TextEncoderRef<'a>>> {
    let count = bpe_encoder.is_some() as u8
        + unigram_encoder.is_some() as u8
        + gpt2_encoder.is_some() as u8;
    if count > 1 {
        return Err(PyErr::new::<DataMismatchError, _>(
            "Pass only one of bpe_encoder, unigram_encoder, or gpt2_encoder.\nFix: Supply a single tokenizer.\nExplanation: Training and generation accept one trained encoder at a time.".to_string(),
        ));
    }
    if let Some(b) = bpe_encoder {
        return Ok(Some(TextEncoderRef::Bpe(&b.inner)));
    }
    if let Some(u) = unigram_encoder {
        return Ok(Some(TextEncoderRef::Unigram(&u.inner)));
    }
    if let Some(g) = gpt2_encoder {
        return Ok(Some(TextEncoderRef::Gpt2(&g.inner)));
    }
    Ok(None)
}
