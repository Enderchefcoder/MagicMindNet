//! Shared reference wrapper for BPE and unigram tokenizers.

use crate::{BytePairEncoder, UnigramEncoder};

/// Borrowed text encoder used by training and generation.
#[derive(Clone, Copy)]
pub enum TextEncoderRef<'a> {
    Bpe(&'a BytePairEncoder),
    Unigram(&'a UnigramEncoder),
}

impl<'a> TextEncoderRef<'a> {
    pub fn encode(self, text: &str) -> Vec<usize> {
        match self {
            Self::Bpe(e) => e.encode(text),
            Self::Unigram(e) => e.encode(text),
        }
    }

    pub fn decode(self, ids: &[usize]) -> String {
        match self {
            Self::Bpe(e) => e.decode(ids),
            Self::Unigram(e) => e.decode(ids),
        }
    }
}
