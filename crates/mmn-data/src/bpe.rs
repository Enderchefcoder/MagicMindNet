//! Minimal byte-level BPE tokenizer for small corpora and tests.

use std::collections::HashMap;

/// Byte-pair encoder trained on UTF-8 bytes with merge rules up to `vocab_size`.
#[derive(Clone, Debug)]
pub struct BytePairEncoder {
    merges: Vec<(usize, usize)>,
    vocab_size: usize,
}

impl BytePairEncoder {
    /// Train on `texts` with at most `num_merges` pair merges (capped by `vocab_size - 256`).
    pub fn train(texts: &[&str], vocab_size: usize, num_merges: usize) -> Self {
        let max_merges = vocab_size.saturating_sub(256).min(num_merges);
        let mut words: Vec<Vec<usize>> = texts
            .iter()
            .map(|t| t.bytes().map(|b| b as usize).collect())
            .collect();
        let mut merges = Vec::new();

        for new_id in 0..max_merges {
            let mut pair_counts: HashMap<(usize, usize), usize> = HashMap::new();
            for word in &words {
                for w in word.windows(2) {
                    *pair_counts.entry((w[0], w[1])).or_default() += 1;
                }
            }
            let Some((pair, count)) = pair_counts.into_iter().max_by_key(|(_, c)| *c) else {
                break;
            };
            if count < 2 {
                break;
            }
            merges.push(pair);
            let merged = 256 + new_id;
            for word in &mut words {
                *word = Self::apply_merge(word, pair, merged);
            }
        }

        Self { merges, vocab_size }
    }

    fn apply_merge(word: &[usize], pair: (usize, usize), merged: usize) -> Vec<usize> {
        let mut out = Vec::with_capacity(word.len());
        let mut i = 0;
        while i < word.len() {
            if i + 1 < word.len() && word[i] == pair.0 && word[i + 1] == pair.1 {
                out.push(merged);
                i += 2;
            } else {
                out.push(word[i]);
                i += 1;
            }
        }
        out
    }

    /// Greedy encode with learned merges; token ids are clamped to `vocab_size`.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        let mut ids: Vec<usize> = text.bytes().map(|b| b as usize).collect();
        for (idx, pair) in self.merges.iter().enumerate() {
            ids = Self::apply_merge(&ids, *pair, 256 + idx);
        }
        ids.into_iter().map(|id| id % self.vocab_size).collect()
    }

    pub fn merge_count(&self) -> usize {
        self.merges.len()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn bpe_train_produces_merges_on_repeated_text() {
        let enc = BytePairEncoder::train(&["aaabbb", "aaabbb"], 512, 8);
        assert!(enc.merge_count() > 0);
    }

    #[test]
    fn bpe_encode_is_deterministic() {
        let enc = BytePairEncoder::train(&["hello world", "hello there"], 512, 4);
        let a = enc.encode("hello");
        let b = enc.encode("hello");
        assert_eq!(a, b);
        assert!(!a.is_empty());
    }

    #[test]
    fn bpe_encode_respects_vocab_size() {
        let enc = BytePairEncoder::train(&["test corpus"], 128, 2);
        for id in enc.encode("testing") {
            assert!(id < 128);
        }
    }

    #[test]
    fn bpe_shorter_encoding_on_repeated_pattern() {
        let enc = BytePairEncoder::train(&["aaaa", "aaaa", "aaaa"], 512, 4);
        let raw_len = "aaaa".bytes().count();
        let bpe_len = enc.encode("aaaa").len();
        assert!(bpe_len <= raw_len);
    }
}
