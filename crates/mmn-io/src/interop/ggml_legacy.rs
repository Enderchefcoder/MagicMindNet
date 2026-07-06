//! Legacy pre-GGUF llama.cpp containers: GGML (unversioned, March 2023),
//! GGMF v1, and GGJT v1-v3 — the "oldest" model files in the ecosystem.
//!
//! Layout (little-endian): magic u32, [version u32], 7×i32 llama hparams
//! (`n_vocab, n_embd, n_mult, n_head, n_layer, n_rot, ftype/f16`), vocab
//! entries (`len + bytes` (+ `score f32` from GGMF on)), then tensors until
//! EOF: `n_dims u32, name_len u32, ftype u32, ne[n_dims] u32, name`, GGJT
//! pads to 32 bytes, data. Pre-GGJT-v2 quantized blocks use the original
//! f32-scale layouts with consecutive-pair nibble packing; GGJT v2/v3 use
//! the modern block layouts shared with GGUF.

use super::gguf_quant::{dequantize, GgmlType};
use super::NamedArray;
use half::f16;
use mmn_core::MmnError;
use std::fs;

const MAGIC_GGML: u32 = 0x6767_6d6c;
const MAGIC_GGMF: u32 = 0x6767_6d66;
const MAGIC_GGJT: u32 = 0x6767_6a74;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// Container flavor + version detected from the magic.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LegacyGgmlContainer {
    /// Unversioned `ggml` (no vocab scores, no alignment).
    Ggml,
    /// `ggmf` v1 (vocab scores, no alignment).
    Ggmf,
    /// `ggjt` v1-v3 (vocab scores, 32-byte tensor alignment).
    Ggjt(u32),
}

impl LegacyGgmlContainer {
    pub fn as_str(&self) -> &'static str {
        match self {
            LegacyGgmlContainer::Ggml => "ggml",
            LegacyGgmlContainer::Ggmf => "ggmf",
            LegacyGgmlContainer::Ggjt(_) => "ggjt",
        }
    }

    fn has_vocab_scores(&self) -> bool {
        !matches!(self, LegacyGgmlContainer::Ggml)
    }

    fn aligned(&self) -> bool {
        matches!(self, LegacyGgmlContainer::Ggjt(_))
    }

    /// GGJT v2 changed Q4/Q8 block layouts to the modern (GGUF) ones;
    /// everything earlier uses the original f32-scale layouts.
    fn modern_quant_blocks(&self) -> bool {
        matches!(self, LegacyGgmlContainer::Ggjt(v) if *v >= 2)
    }
}

/// True when the buffer starts with a legacy GGML/GGMF/GGJT magic.
pub fn is_ggml_legacy_bytes(bytes: &[u8]) -> bool {
    if bytes.len() < 4 {
        return false;
    }
    let magic = u32::from_le_bytes([bytes[0], bytes[1], bytes[2], bytes[3]]);
    matches!(magic, MAGIC_GGML | MAGIC_GGMF | MAGIC_GGJT)
}

struct Reader<'a> {
    bytes: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Result<&'a [u8], MmnError> {
        let end = self
            .pos
            .checked_add(n)
            .filter(|&e| e <= self.bytes.len())
            .ok_or_else(|| err("legacy ggml file truncated"))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, MmnError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }
}

/// Dequantize the original (pre-GGJT-v2) Q4_0 layout: f32 scale + 16 bytes
/// of consecutive-pair nibbles per 32 values, `v = (q - 8) * d`.
fn dequantize_legacy_q4_0(data: &[u8], numel: usize) -> Result<Vec<f32>, MmnError> {
    const BLOCK: usize = 20;
    let blocks = numel / 32;
    if data.len() != blocks * BLOCK {
        return Err(err("legacy q4_0 data length mismatch"));
    }
    let mut out = Vec::with_capacity(numel);
    for block in data.chunks_exact(BLOCK) {
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        for &q in &block[4..20] {
            out.push(((q & 0x0F) as f32 - 8.0) * d);
            out.push(((q >> 4) as f32 - 8.0) * d);
        }
    }
    Ok(out)
}

/// Dequantize the original Q4_1 layout: f32 scale + f32 min + 16 bytes of
/// consecutive-pair nibbles per 32 values, `v = m + q * d`.
fn dequantize_legacy_q4_1(data: &[u8], numel: usize) -> Result<Vec<f32>, MmnError> {
    const BLOCK: usize = 24;
    let blocks = numel / 32;
    if data.len() != blocks * BLOCK {
        return Err(err("legacy q4_1 data length mismatch"));
    }
    let mut out = Vec::with_capacity(numel);
    for block in data.chunks_exact(BLOCK) {
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        let m = f32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        for &q in &block[8..24] {
            out.push(m + (q & 0x0F) as f32 * d);
            out.push(m + (q >> 4) as f32 * d);
        }
    }
    Ok(out)
}

/// Bytes per tensor + dequantizer for one legacy ftype.
fn legacy_data_len(
    container: LegacyGgmlContainer,
    ftype: u32,
    numel: usize,
    name: &str,
) -> Result<usize, MmnError> {
    match (ftype, container.modern_quant_blocks()) {
        (0, _) => Ok(numel * 4),
        (1, _) => Ok(numel * 2),
        (2, false) => Ok(numel / 32 * 20),
        (3, false) => Ok(numel / 32 * 24),
        (2, true) => GgmlType::Q4_0.data_len(numel),
        (3, true) => GgmlType::Q4_1.data_len(numel),
        (6, true) => GgmlType::Q5_0.data_len(numel),
        (7, true) => GgmlType::Q5_1.data_len(numel),
        (8, true) => GgmlType::Q8_0.data_len(numel),
        (other, _) => Err(err(format!(
            "legacy ggml tensor {name}: ftype {other} not supported for this container \
             (f32/f16/q4_0/q4_1, plus q5/q8 for ggjt v2+)"
        ))),
    }
}

fn legacy_dequantize(
    container: LegacyGgmlContainer,
    ftype: u32,
    data: &[u8],
    numel: usize,
) -> Result<Vec<f32>, MmnError> {
    match (ftype, container.modern_quant_blocks()) {
        (0, _) => Ok(data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect()),
        (1, _) => Ok(data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect()),
        (2, false) => dequantize_legacy_q4_0(data, numel),
        (3, false) => dequantize_legacy_q4_1(data, numel),
        (2, true) => dequantize(GgmlType::Q4_0, data, numel),
        (3, true) => dequantize(GgmlType::Q4_1, data, numel),
        (6, true) => dequantize(GgmlType::Q5_0, data, numel),
        (7, true) => dequantize(GgmlType::Q5_1, data, numel),
        (8, true) => dequantize(GgmlType::Q8_0, data, numel),
        _ => unreachable!("legacy_data_len validates ftypes"),
    }
}

/// Parsed legacy file: container info, llama hparams, vocab, tensors.
pub struct LegacyGgmlFile {
    pub container: LegacyGgmlContainer,
    /// `[n_vocab, n_embd, n_mult, n_head, n_layer, n_rot, ftype]`.
    pub hparams: [i32; 7],
    /// Vocabulary tokens with scores (scores are 0 for unversioned GGML).
    pub vocab: Vec<(Vec<u8>, f32)>,
    pub arrays: Vec<NamedArray>,
}

/// Read a legacy GGML/GGMF/GGJT byte buffer.
pub fn read_ggml_legacy_bytes(bytes: &[u8]) -> Result<LegacyGgmlFile, MmnError> {
    let mut r = Reader { bytes, pos: 0 };
    let magic = r.u32()?;
    let container = match magic {
        MAGIC_GGML => LegacyGgmlContainer::Ggml,
        MAGIC_GGMF => {
            let version = r.u32()?;
            if version != 1 {
                return Err(err(format!("ggmf version {version} not supported (1 is)")));
            }
            LegacyGgmlContainer::Ggmf
        }
        MAGIC_GGJT => {
            let version = r.u32()?;
            if !(1..=3).contains(&version) {
                return Err(err(format!(
                    "ggjt version {version} not supported (1-3 are)"
                )));
            }
            LegacyGgmlContainer::Ggjt(version)
        }
        _ => return Err(err("not a legacy ggml/ggmf/ggjt file")),
    };
    let mut hparams = [0i32; 7];
    for h in hparams.iter_mut() {
        *h = r.u32()? as i32;
    }
    let n_vocab = hparams[0];
    if !(0..=(1 << 22)).contains(&n_vocab) {
        return Err(err(format!("legacy ggml n_vocab {n_vocab} unreasonable")));
    }
    let mut vocab = Vec::with_capacity(n_vocab as usize);
    for _ in 0..n_vocab {
        let len = r.u32()? as usize;
        if len > bytes.len() {
            return Err(err("legacy ggml vocab entry length exceeds file"));
        }
        let token = r.take(len)?.to_vec();
        let score = if container.has_vocab_scores() {
            f32::from_le_bytes(r.take(4)?.try_into().unwrap())
        } else {
            0.0
        };
        vocab.push((token, score));
    }
    let mut arrays = Vec::new();
    while r.pos < bytes.len() {
        let n_dims = r.u32()? as usize;
        let name_len = r.u32()? as usize;
        let ftype = r.u32()?;
        if n_dims > 4 || name_len > 4096 {
            return Err(err("legacy ggml tensor header unreasonable"));
        }
        // ne[0] is fastest-varying (GGML order): row-major shape reverses.
        let mut ne = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            ne.push(r.u32()? as usize);
        }
        let name = String::from_utf8(r.take(name_len)?.to_vec())
            .map_err(|e| err(format!("legacy ggml tensor name not UTF-8: {e}")))?;
        if container.aligned() {
            r.pos = r.pos.div_ceil(32) * 32;
        }
        let numel: usize = ne.iter().product();
        let data_len = legacy_data_len(container, ftype, numel, &name)?;
        let data = r.take(data_len)?;
        let values = legacy_dequantize(container, ftype, data, numel)?;
        let shape: Vec<usize> = ne.iter().rev().copied().collect();
        arrays.push((name, shape, values));
    }
    Ok(LegacyGgmlFile {
        container,
        hparams,
        vocab,
        arrays,
    })
}

/// Read a legacy GGML/GGMF/GGJT file from disk.
pub fn read_ggml_legacy(path: &str) -> Result<LegacyGgmlFile, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read {path}: {e}")))?;
    read_ggml_legacy_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Build a legacy file per the spec (independent of the reader).
    fn build(container: LegacyGgmlContainer, tensors: &[(&str, Vec<usize>, u32, Vec<u8>)]) -> Vec<u8> {
        let mut b: Vec<u8> = Vec::new();
        match container {
            LegacyGgmlContainer::Ggml => b.extend_from_slice(&MAGIC_GGML.to_le_bytes()),
            LegacyGgmlContainer::Ggmf => {
                b.extend_from_slice(&MAGIC_GGMF.to_le_bytes());
                b.extend_from_slice(&1u32.to_le_bytes());
            }
            LegacyGgmlContainer::Ggjt(v) => {
                b.extend_from_slice(&MAGIC_GGJT.to_le_bytes());
                b.extend_from_slice(&v.to_le_bytes());
            }
        }
        // hparams: n_vocab=2, n_embd=8, n_mult=1, n_head=1, n_layer=1, n_rot=2, ftype=0.
        for h in [2i32, 8, 1, 1, 1, 2, 0] {
            b.extend_from_slice(&h.to_le_bytes());
        }
        for token in [b"<s>".as_slice(), b"hi".as_slice()] {
            b.extend_from_slice(&(token.len() as u32).to_le_bytes());
            b.extend_from_slice(token);
            if container.has_vocab_scores() {
                b.extend_from_slice(&(-1.5f32).to_le_bytes());
            }
        }
        for (name, ne, ftype, data) in tensors {
            b.extend_from_slice(&(ne.len() as u32).to_le_bytes());
            b.extend_from_slice(&(name.len() as u32).to_le_bytes());
            b.extend_from_slice(&ftype.to_le_bytes());
            for &d in ne {
                b.extend_from_slice(&(d as u32).to_le_bytes());
            }
            b.extend_from_slice(name.as_bytes());
            if container.aligned() {
                let pad = b.len().div_ceil(32) * 32 - b.len();
                b.extend(std::iter::repeat_n(0u8, pad));
            }
            b.extend_from_slice(data);
        }
        b
    }

    fn f32_bytes(values: &[f32]) -> Vec<u8> {
        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    #[test]
    fn unversioned_ggml_reads_f32_tensor_without_scores() {
        let data = f32_bytes(&[1.0, -2.0, 3.0, 4.0]);
        let bytes = build(
            LegacyGgmlContainer::Ggml,
            &[("tok_embeddings.weight", vec![2, 2], 0, data)],
        );
        let file = read_ggml_legacy_bytes(&bytes).unwrap();
        assert_eq!(file.container, LegacyGgmlContainer::Ggml);
        assert_eq!(file.hparams[1], 8); // n_embd
        assert_eq!(file.vocab.len(), 2);
        assert_eq!(file.vocab[0].0, b"<s>");
        assert_eq!(file.vocab[0].1, 0.0); // no scores in unversioned ggml
        assert_eq!(
            file.arrays,
            vec![(
                "tok_embeddings.weight".to_string(),
                vec![2, 2],
                vec![1.0, -2.0, 3.0, 4.0]
            )]
        );
    }

    #[test]
    fn ggmf_reads_scores_and_f16() {
        let values = [0.5f32, -1.0, 2.0, 0.0];
        let data: Vec<u8> = values
            .iter()
            .flat_map(|&v| f16::from_f32(v).to_le_bytes())
            .collect();
        let bytes = build(
            LegacyGgmlContainer::Ggmf,
            &[("norm.weight", vec![4], 1, data)],
        );
        let file = read_ggml_legacy_bytes(&bytes).unwrap();
        assert_eq!(file.vocab[1].1, -1.5);
        assert_eq!(file.arrays[0].2, values);
    }

    #[test]
    fn ggjt_v1_uses_legacy_q4_0_layout_and_alignment() {
        // One block: d=0.5, nibbles pair-packed: values q=8..(=0.0), q=10 (=1.0)...
        let mut block = Vec::new();
        block.extend_from_slice(&0.5f32.to_le_bytes());
        // 16 bytes: first byte packs q0=10 (low), q1=6 (high) -> 1.0, -1.0.
        block.push(10 | (6 << 4));
        block.extend(std::iter::repeat_n(8 | (8 << 4), 15)); // all zeros
        let bytes = build(
            LegacyGgmlContainer::Ggjt(1),
            &[("layers.0.wq.weight", vec![32], 2, block)],
        );
        let file = read_ggml_legacy_bytes(&bytes).unwrap();
        assert_eq!(file.container, LegacyGgmlContainer::Ggjt(1));
        let values = &file.arrays[0].2;
        assert_eq!(values.len(), 32);
        assert_eq!(values[0], 1.0);
        assert_eq!(values[1], -1.0);
        assert!(values[2..].iter().all(|&v| v == 0.0));
    }

    #[test]
    fn ggjt_v2_uses_modern_q4_0_layout() {
        // Modern q4_0 block: f16 d + 16 bytes, halves layout, q in [0,15]-8.
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        // Element 0 = low nibble of byte 0 (value 9 -> 1.0);
        // element 16 = high nibble of byte 0 (value 7 -> -1.0).
        block.push(9 | (7 << 4));
        block.extend(std::iter::repeat_n(8 | (8 << 4), 15));
        let bytes = build(
            LegacyGgmlContainer::Ggjt(2),
            &[("layers.0.wk.weight", vec![32], 2, block)],
        );
        let file = read_ggml_legacy_bytes(&bytes).unwrap();
        let values = &file.arrays[0].2;
        assert_eq!(values[0], 1.0);
        assert_eq!(values[16], -1.0);
    }

    #[test]
    fn legacy_q4_1_dequantizes() {
        let mut block = Vec::new();
        block.extend_from_slice(&0.25f32.to_le_bytes()); // d
        block.extend_from_slice(&(-2.0f32).to_le_bytes()); // m
        block.push(15 << 4); // q0=0 -> -2.0, q1=15 -> 1.75
        block.extend(std::iter::repeat_n(0u8, 15));
        let bytes = build(
            LegacyGgmlContainer::Ggmf,
            &[("w", vec![32], 3, block)],
        );
        let file = read_ggml_legacy_bytes(&bytes).unwrap();
        let values = &file.arrays[0].2;
        assert_eq!(values[0], -2.0);
        assert_eq!(values[1], 1.75);
    }

    #[test]
    fn bad_versions_and_truncation_error() {
        let mut bytes = build(LegacyGgmlContainer::Ggjt(1), &[]);
        bytes[4..8].copy_from_slice(&9u32.to_le_bytes());
        assert!(read_ggml_legacy_bytes(&bytes)
            .err()
            .unwrap()
            .message()
            .contains("version"));
        let good = build(
            LegacyGgmlContainer::Ggml,
            &[("w", vec![2], 0, f32_bytes(&[1.0, 2.0]))],
        );
        assert!(read_ggml_legacy_bytes(&good[..good.len() - 3]).is_err());
        assert!(!is_ggml_legacy_bytes(b"GGUF"));
        assert!(is_ggml_legacy_bytes(&good));
    }
}
