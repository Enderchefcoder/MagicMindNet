//! Minimal unigram (SentencePiece-style) tokenizer with Viterbi segmentation.

use std::collections::HashMap;
use std::fs;
use std::path::Path;

use mmn_core::{MmnError, Result};
use serde::{Deserialize, Serialize};

const UNIGRAM_FORMAT: &str = "mmn-unigram-v1";
const BYTE_VOCAB: usize = 256;
const DEFAULT_MAX_PIECE_LEN: usize = 12;
const EM_ITERS: usize = 3;

/// Unigram language-model tokenizer: pieces with log-prob scores and Viterbi encode.
#[derive(Clone, Debug)]
pub struct UnigramEncoder {
    pieces: Vec<Vec<u8>>,
    log_probs: Vec<f32>,
    vocab_size: usize,
}

#[derive(Serialize, Deserialize)]
struct UnigramCheckpoint {
    format: String,
    vocab_size: usize,
    pieces: Vec<String>,
    log_probs: Vec<f32>,
}

impl UnigramEncoder {
    /// Train on `texts` with up to `vocab_size` pieces (256 single-byte tokens + merged substrings).
    pub fn train(texts: &[&str], vocab_size: usize) -> Self {
        Self::train_with_options(texts, vocab_size, DEFAULT_MAX_PIECE_LEN)
    }

    pub fn train_with_options(texts: &[&str], vocab_size: usize, max_piece_len: usize) -> Self {
        let target = vocab_size.max(BYTE_VOCAB);
        let mut pieces: Vec<Vec<u8>> = (0u8..=255).map(|b| vec![b]).collect();
        let mut log_probs = vec![(-(BYTE_VOCAB as f32)).ln(); BYTE_VOCAB];

        let mut sub_counts: HashMap<Vec<u8>, usize> = HashMap::new();
        for text in texts {
            let bytes = text.as_bytes();
            for len in 2..=max_piece_len.min(bytes.len()) {
                for w in bytes.windows(len) {
                    *sub_counts.entry(w.to_vec()).or_default() += 1;
                }
            }
        }
        let mut ranked: Vec<(Vec<u8>, usize)> = sub_counts.into_iter().collect();
        ranked.sort_by(|a, b| b.1.cmp(&a.1));

        for (piece, _) in ranked {
            if pieces.len() >= target {
                break;
            }
            if piece.len() <= 1 {
                continue;
            }
            if pieces.iter().any(|p| p == &piece) {
                continue;
            }
            pieces.push(piece);
            log_probs.push(0.0);
        }

        let mut enc = Self {
            pieces,
            log_probs,
            vocab_size: target,
        };
        enc.run_em(texts, EM_ITERS);
        enc
    }

    fn run_em(&mut self, texts: &[&str], iters: usize) {
        for _ in 0..iters {
            let mut counts = vec![0.0f32; self.pieces.len()];
            for text in texts {
                let ids = self.encode_viterbi(text);
                for id in ids {
                    counts[id] += 1.0;
                }
            }
            let total: f32 = counts.iter().sum::<f32>().max(1.0);
            for (i, c) in counts.iter().enumerate() {
                self.log_probs[i] = (*c / total).max(1e-12).ln();
            }
        }
    }

    fn piece_id(&self, bytes: &[u8]) -> Option<usize> {
        self.pieces.iter().position(|p| p.as_slice() == bytes)
    }

    /// Viterbi segmentation maximizing sum of piece log-probs.
    pub fn encode(&self, text: &str) -> Vec<usize> {
        self.encode_viterbi(text)
    }

    fn encode_viterbi(&self, text: &str) -> Vec<usize> {
        let bytes = text.as_bytes();
        let n = bytes.len();
        if n == 0 {
            return Vec::new();
        }
        let neg_inf = f32::NEG_INFINITY;
        let mut best = vec![neg_inf; n + 1];
        let mut prev_len = vec![0usize; n + 1];
        best[0] = 0.0;

        for i in 0..n {
            if best[i] == neg_inf {
                continue;
            }
            let max_len = (n - i).min(DEFAULT_MAX_PIECE_LEN);
            for len in 1..=max_len {
                let slice = &bytes[i..i + len];
                let Some(id) = self.piece_id(slice) else {
                    continue;
                };
                let score = best[i] + self.log_probs[id];
                if score > best[i + len] {
                    best[i + len] = score;
                    prev_len[i + len] = len;
                }
            }
        }

        let mut ids = Vec::new();
        let mut pos = n;
        while pos > 0 {
            let len = prev_len[pos];
            if len == 0 {
                let b = bytes[pos - 1];
                ids.push(b as usize);
                pos -= 1;
            } else {
                let slice = &bytes[pos - len..pos];
                ids.push(self.piece_id(slice).unwrap_or(slice[0] as usize));
                pos -= len;
            }
        }
        ids.reverse();
        ids
    }

    pub fn decode(&self, ids: &[usize]) -> String {
        let mut out = Vec::new();
        for &id in ids {
            if id < self.pieces.len() {
                out.extend_from_slice(&self.pieces[id]);
            } else if id < BYTE_VOCAB {
                out.push(id as u8);
            }
        }
        String::from_utf8_lossy(&out).into_owned()
    }

    pub fn vocab_size(&self) -> usize {
        self.vocab_size
    }

    pub fn piece_count(&self) -> usize {
        self.pieces.len()
    }

    /// Drop merged pieces (not single-byte tokens) with log-prob below `min_log_prob`.
    pub fn prune_pieces_below_logprob(&mut self, min_log_prob: f32) {
        let mut keep: Vec<usize> = (0..self.pieces.len())
            .filter(|&i| i < BYTE_VOCAB || self.log_probs[i] >= min_log_prob)
            .collect();
        if keep.len() < BYTE_VOCAB {
            keep = (0..BYTE_VOCAB).collect();
        }
        let old_pieces = std::mem::take(&mut self.pieces);
        let old_log_probs = std::mem::take(&mut self.log_probs);
        self.pieces = keep.iter().map(|&i| old_pieces[i].clone()).collect();
        self.log_probs = keep.iter().map(|&i| old_log_probs[i]).collect();
        let total = self.log_probs.iter().map(|p| p.exp()).sum::<f32>();
        if total > 0.0 {
            for lp in &mut self.log_probs {
                *lp = (*lp).exp() / total;
                *lp = lp.ln();
            }
        }
        self.vocab_size = self.vocab_size.max(self.pieces.len());
    }

    pub fn export_json(&self, path: &str) -> Result<()> {
        let p = Path::new(path);
        if let Some(parent) = p.parent() {
            if !parent.as_os_str().is_empty() {
                fs::create_dir_all(parent).map_err(|e| MmnError::Other {
                    message: e.to_string(),
                })?;
            }
        }
        let pieces: Vec<String> = self
            .pieces
            .iter()
            .map(|p| String::from_utf8_lossy(p).into_owned())
            .collect();
        let ckpt = UnigramCheckpoint {
            format: UNIGRAM_FORMAT.into(),
            vocab_size: self.vocab_size,
            pieces,
            log_probs: self.log_probs.clone(),
        };
        let json = serde_json::to_string_pretty(&ckpt).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        fs::write(path, json).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })
    }

    pub fn import_json(path: &str) -> Result<Self> {
        let raw = fs::read_to_string(path).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?;
        let ckpt: UnigramCheckpoint = serde_json::from_str(&raw).map_err(|e| MmnError::Other {
            message: format!("invalid unigram checkpoint JSON: {e}"),
        })?;
        if ckpt.format != UNIGRAM_FORMAT {
            return Err(MmnError::Other {
                message: format!(
                    "expected format {UNIGRAM_FORMAT}, got {}",
                    ckpt.format
                ),
            });
        }
        if ckpt.pieces.len() != ckpt.log_probs.len() {
            return Err(MmnError::Other {
                message: "unigram checkpoint pieces/log_probs length mismatch".into(),
            });
        }
        let pieces: Vec<Vec<u8>> = ckpt.pieces.into_iter().map(|s| s.into_bytes()).collect();
        Ok(Self {
            pieces,
            log_probs: ckpt.log_probs,
            vocab_size: ckpt.vocab_size,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn unigram_roundtrip_utf8() {
        let enc = UnigramEncoder::train(&["hello world", "hello there"], 512);
        assert!(enc.piece_count() >= BYTE_VOCAB);
        let ids = enc.encode("hello");
        let back = enc.decode(&ids);
        assert_eq!(back, "hello");
    }

    #[test]
    fn unigram_prune_drops_low_logprob_pieces() {
        let mut enc = UnigramEncoder::train(&["aaaa bbbb cccc dddd"], 400);
        let before = enc.piece_count();
        enc.prune_pieces_below_logprob(-5.0);
        assert!(enc.piece_count() <= before);
        assert!(enc.piece_count() >= BYTE_VOCAB);
        let ids = enc.encode("aaaa");
        assert!(!ids.is_empty());
    }

    #[test]
    fn unigram_json_roundtrip() {
        let enc = UnigramEncoder::train(&["abc abc abc"], 320);
        let dir = std::env::temp_dir().join("mmn_unigram_test.json");
        let path = dir.to_string_lossy();
        enc.export_json(&path).unwrap();
        let loaded = UnigramEncoder::import_json(&path).unwrap();
        assert_eq!(loaded.encode("abc"), enc.encode("abc"));
        let _ = fs::remove_file(&dir);
    }
}
