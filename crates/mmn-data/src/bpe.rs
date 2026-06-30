//! Minimal byte-level BPE tokenizer for small corpora and tests.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mmn_core::{MmnError, Result};
use serde::{Deserialize, Serialize};

const BPE_FORMAT: &str = "mmn-bpe-v1";

/// Byte-pair encoder trained on UTF-8 bytes with merge rules up to `vocab_size`.
#[derive(Clone, Debug)]
pub struct BytePairEncoder {
    merges: Vec<(usize, usize)>,
    vocab_size: usize,
}

#[derive(Serialize, Deserialize)]
struct BpeCheckpoint {
    format: String,
    vocab_size: usize,
    merges: Vec<[usize; 2]>,
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

    /// Restore from trained merge rules (used by checkpoint import).
    pub fn from_merges(vocab_size: usize, merges: Vec<(usize, usize)>) -> Result<Self> {
        if vocab_size < 256 {
            return Err(MmnError::Other {
                message: "BPE vocab_size must be at least 256 (byte tokens 0..255)".into(),
            });
        }
        let max_merges = vocab_size - 256;
        if merges.len() > max_merges {
            return Err(MmnError::Other {
                message: format!(
                    "BPE merge count {} exceeds vocab_size {vocab_size} (max {max_merges} merges)",
                    merges.len()
                ),
            });
        }
        Ok(Self { merges, vocab_size })
    }

    /// Write `mmn-bpe-v1` JSON checkpoint (creates parent directories).
    pub fn export_json(&self, path: &str) -> Result<()> {
        let p = Path::new(path);
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| MmnError::Other {
                    message: e.to_string(),
                })?;
            }
        }
        let ckpt = BpeCheckpoint {
            format: BPE_FORMAT.into(),
            vocab_size: self.vocab_size,
            merges: self
                .merges
                .iter()
                .map(|(a, b)| [*a, *b])
                .collect(),
        };
        let json = serde_json::to_string_pretty(&ckpt).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        fs::write(path, json).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })
    }

    /// Load `mmn-bpe-v1` JSON checkpoint.
    pub fn import_json(path: &str) -> Result<Self> {
        let raw = fs::read_to_string(path).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        let ckpt: BpeCheckpoint = serde_json::from_str(&raw).map_err(|e| MmnError::Other {
            message: format!("invalid BPE checkpoint JSON: {e}"),
        })?;
        if ckpt.format != BPE_FORMAT {
            return Err(MmnError::Other {
                message: format!(
                    "expected format {BPE_FORMAT}, got {}",
                    ckpt.format
                ),
            });
        }
        let merges: Vec<(usize, usize)> = ckpt.merges.into_iter().map(|[a, b]| (a, b)).collect();
        Self::from_merges(ckpt.vocab_size, merges)
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

    fn expand_token(id: usize, merges: &[(usize, usize)]) -> Vec<u8> {
        if id < 256 {
            return vec![id as u8];
        }
        let merge_idx = id.wrapping_sub(256);
        if merge_idx >= merges.len() {
            return vec![(id % 256) as u8];
        }
        let (a, b) = merges[merge_idx];
        let mut out = Self::expand_token(a, merges);
        out.extend(Self::expand_token(b, merges));
        out
    }

    /// Decode BPE token ids back to UTF-8 (lossy for invalid byte sequences).
    pub fn decode(&self, ids: &[usize]) -> String {
        let mut bytes = Vec::new();
        for &id in ids {
            let clamped = id % self.vocab_size;
            bytes.extend(Self::expand_token(clamped, &self.merges));
        }
        String::from_utf8_lossy(&bytes).into_owned()
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

    #[test]
    fn bpe_decode_roundtrip_ascii() {
        let enc = BytePairEncoder::train(&["hello hello", "hello world"], 512, 8);
        let text = "hello";
        let ids = enc.encode(text);
        let decoded = enc.decode(&ids);
        assert_eq!(decoded, text);
    }

    #[test]
    fn bpe_json_roundtrip_preserves_encode() {
        let enc = BytePairEncoder::train(&["repeat repeat token", "repeat repeat token"], 512, 12);
        let path = format!(
            "{}/../../target/test_bpe_roundtrip.mmn",
            env!("CARGO_MANIFEST_DIR")
        );
        enc.export_json(&path).unwrap();
        let loaded = BytePairEncoder::import_json(&path).unwrap();
        let sample = "repeat repeat token again";
        assert_eq!(enc.encode(sample), loaded.encode(sample));
        assert_eq!(enc.merge_count(), loaded.merge_count());
        assert_eq!(enc.vocab_size(), loaded.vocab_size());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn bpe_import_rejects_wrong_format() {
        let path = format!(
            "{}/../../target/test_bpe_bad_format.mmn",
            env!("CARGO_MANIFEST_DIR")
        );
        fs::write(&path, r#"{"format":"other","vocab_size":512,"merges":[]}"#).unwrap();
        let err = BytePairEncoder::import_json(&path).unwrap_err();
        assert!(err.to_string().contains("mmn-bpe-v1"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn bpe_import_rejects_vocab_too_small() {
        let path = format!(
            "{}/../../target/test_bpe_small_vocab.mmn",
            env!("CARGO_MANIFEST_DIR")
        );
        fs::write(
            &path,
            r#"{"format":"mmn-bpe-v1","vocab_size":128,"merges":[[97,97]]}"#,
        )
        .unwrap();
        let err = BytePairEncoder::import_json(&path).unwrap_err();
        assert!(err.to_string().contains("256"));
        let _ = fs::remove_file(&path);
    }
}
