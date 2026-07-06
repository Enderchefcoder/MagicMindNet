//! Shared reference wrapper for BPE, unigram, and GPT-2 byte-BPE tokenizers.

use crate::{BytePairEncoder, Gpt2BpeEncoder, UnigramEncoder};

/// Borrowed text encoder used by training and generation.
#[derive(Clone, Copy)]
pub enum TextEncoderRef<'a> {
    Bpe(&'a BytePairEncoder),
    Unigram(&'a UnigramEncoder),
    Gpt2(&'a Gpt2BpeEncoder),
}

impl<'a> TextEncoderRef<'a> {
    pub fn encode(self, text: &str) -> Vec<usize> {
        match self {
            Self::Bpe(e) => e.encode(text),
            Self::Unigram(e) => e.encode(text),
            Self::Gpt2(e) => e.encode(text),
        }
    }

    pub fn decode(self, ids: &[usize]) -> String {
        match self {
            Self::Bpe(e) => e.decode(ids),
            Self::Unigram(e) => e.decode(ids),
            Self::Gpt2(e) => e.decode(ids),
        }
    }
}
