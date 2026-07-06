//! From-scratch GGML quantization block codecs for GGUF tensors.
//!
//! Implements the on-disk block layouts directly from the format definition —
//! no llama.cpp / ggml code is linked. Supported: F32, F16, BF16, F64,
//! Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K plus plain integer tensors.

use half::{bf16, f16};
use mmn_core::MmnError;

pub const QK: usize = 32; // classic quant block size
pub const QK_K: usize = 256; // k-quant super-block size

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// GGML tensor type ids as stored in GGUF tensor infos.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q4_0,
    Q4_1,
    Q5_0,
    Q5_1,
    Q8_0,
    Q4K,
    Q6K,
    I8,
    I16,
    I32,
    I64,
    F64,
    BF16,
}

impl GgmlType {
    pub fn from_id(id: u32) -> Result<Self, MmnError> {
        Ok(match id {
            0 => GgmlType::F32,
            1 => GgmlType::F16,
            2 => GgmlType::Q4_0,
            3 => GgmlType::Q4_1,
            6 => GgmlType::Q5_0,
            7 => GgmlType::Q5_1,
            8 => GgmlType::Q8_0,
            12 => GgmlType::Q4K,
            14 => GgmlType::Q6K,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            30 => GgmlType::BF16,
            other => {
                return Err(err(format!(
                    "GGUF tensor type id {other} not supported (supported: F32/F16/BF16/F64, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q4_K, Q6_K, ints)"
                )));
            }
        })
    }

    pub fn type_id(&self) -> u32 {
        match self {
            GgmlType::F32 => 0,
            GgmlType::F16 => 1,
            GgmlType::Q4_0 => 2,
            GgmlType::Q4_1 => 3,
            GgmlType::Q5_0 => 6,
            GgmlType::Q5_1 => 7,
            GgmlType::Q8_0 => 8,
            GgmlType::Q4K => 12,
            GgmlType::Q6K => 14,
            GgmlType::I8 => 24,
            GgmlType::I16 => 25,
            GgmlType::I32 => 26,
            GgmlType::I64 => 27,
            GgmlType::F64 => 28,
            GgmlType::BF16 => 30,
        }
    }

    /// (elements per block, bytes per block).
    pub fn block_layout(&self) -> (usize, usize) {
        match self {
            GgmlType::F32 => (1, 4),
            GgmlType::F16 | GgmlType::BF16 => (1, 2),
            GgmlType::Q4_0 => (QK, 2 + QK / 2),
            GgmlType::Q4_1 => (QK, 4 + QK / 2),
            GgmlType::Q5_0 => (QK, 2 + 4 + QK / 2),
            GgmlType::Q5_1 => (QK, 4 + 4 + QK / 2),
            GgmlType::Q8_0 => (QK, 2 + QK),
            GgmlType::Q4K => (QK_K, 2 + 2 + 12 + QK_K / 2),
            GgmlType::Q6K => (QK_K, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2),
            GgmlType::I8 => (1, 1),
            GgmlType::I16 => (1, 2),
            GgmlType::I32 => (1, 4),
            GgmlType::I64 => (1, 8),
            GgmlType::F64 => (1, 8),
        }
    }

    /// Bytes needed to store `numel` elements of this type.
    pub fn data_len(&self, numel: usize) -> Result<usize, MmnError> {
        let (block_elems, block_bytes) = self.block_layout();
        if !numel.is_multiple_of(block_elems) {
            return Err(err(format!(
                "tensor element count {numel} is not a multiple of the {self:?} block size {block_elems}"
            )));
        }
        Ok(numel / block_elems * block_bytes)
    }
}

fn f16_at(data: &[u8], pos: usize) -> f32 {
    f16::from_le_bytes([data[pos], data[pos + 1]]).to_f32()
}

fn dequant_q4_0(data: &[u8], out: &mut Vec<f32>) {
    for block in data.chunks_exact(18) {
        let d = f16_at(block, 0);
        let qs = &block[2..18];
        for &byte in qs {
            out.push(((byte & 0x0F) as i32 - 8) as f32 * d);
        }
        for &byte in qs {
            out.push(((byte >> 4) as i32 - 8) as f32 * d);
        }
    }
}

fn dequant_q4_1(data: &[u8], out: &mut Vec<f32>) {
    for block in data.chunks_exact(20) {
        let d = f16_at(block, 0);
        let m = f16_at(block, 2);
        let qs = &block[4..20];
        for &byte in qs {
            out.push((byte & 0x0F) as f32 * d + m);
        }
        for &byte in qs {
            out.push((byte >> 4) as f32 * d + m);
        }
    }
}

fn dequant_q5_0(data: &[u8], out: &mut Vec<f32>) {
    for block in data.chunks_exact(22) {
        let d = f16_at(block, 0);
        let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
        let qs = &block[6..22];
        for (i, &byte) in qs.iter().enumerate() {
            let high = ((qh >> i) & 1) << 4;
            let x = (byte & 0x0F) as u32 | high;
            out.push((x as i32 - 16) as f32 * d);
        }
        for (i, &byte) in qs.iter().enumerate() {
            let high = ((qh >> (i + 16)) & 1) << 4;
            let x = (byte >> 4) as u32 | high;
            out.push((x as i32 - 16) as f32 * d);
        }
    }
}

fn dequant_q5_1(data: &[u8], out: &mut Vec<f32>) {
    for block in data.chunks_exact(24) {
        let d = f16_at(block, 0);
        let m = f16_at(block, 2);
        let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
        let qs = &block[8..24];
        for (i, &byte) in qs.iter().enumerate() {
            let x = (byte & 0x0F) as u32 | (((qh >> i) & 1) << 4);
            out.push(x as f32 * d + m);
        }
        for (i, &byte) in qs.iter().enumerate() {
            let x = (byte >> 4) as u32 | (((qh >> (i + 16)) & 1) << 4);
            out.push(x as f32 * d + m);
        }
    }
}

fn dequant_q8_0(data: &[u8], out: &mut Vec<f32>) {
    for block in data.chunks_exact(34) {
        let d = f16_at(block, 0);
        for &byte in &block[2..34] {
            out.push((byte as i8) as f32 * d);
        }
    }
}

/// 6-bit packed (scale, min) pairs used by Q4_K super-blocks.
fn scale_min_k4(j: usize, scales: &[u8]) -> (u8, u8) {
    if j < 4 {
        (scales[j] & 63, scales[j + 4] & 63)
    } else {
        (
            (scales[j + 4] & 0x0F) | ((scales[j - 4] >> 6) << 4),
            (scales[j + 4] >> 4) | ((scales[j] >> 6) << 4),
        )
    }
}

fn dequant_q4_k(data: &[u8], out: &mut Vec<f32>) {
    // Super-block: f16 d, f16 dmin, 12 bytes packed scales, 128 bytes nibbles.
    for block in data.chunks_exact(144) {
        let d = f16_at(block, 0);
        let dmin = f16_at(block, 2);
        let scales = &block[4..16];
        let qs = &block[16..144];
        let mut is = 0usize;
        for chunk in qs.chunks_exact(32) {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;
            for &byte in chunk {
                out.push(d1 * (byte & 0x0F) as f32 - min1);
            }
            for &byte in chunk {
                out.push(d2 * (byte >> 4) as f32 - min2);
            }
            is += 2;
        }
    }
}

fn dequant_q6_k(data: &[u8], out: &mut Vec<f32>) {
    // Super-block: 128 bytes ql, 64 bytes qh, 16 signed scales, f16 d.
    for block in data.chunks_exact(210) {
        let ql = &block[0..128];
        let qh = &block[128..192];
        let scales = &block[192..208];
        let d = f16_at(block, 208);
        let start = out.len();
        out.resize(start + QK_K, 0.0);
        for n in 0..2 {
            let ql = &ql[n * 64..];
            let qh = &qh[n * 32..];
            let sc = &scales[n * 8..];
            let y = &mut out[start + n * 128..];
            for l in 0..32 {
                let is = l / 16;
                let q1 = ((ql[l] & 0x0F) as i32 | (((qh[l] as i32) & 3) << 4)) - 32;
                let q2 = ((ql[l + 32] & 0x0F) as i32 | ((((qh[l] as i32) >> 2) & 3) << 4)) - 32;
                let q3 = ((ql[l] >> 4) as i32 | ((((qh[l] as i32) >> 4) & 3) << 4)) - 32;
                let q4 = ((ql[l + 32] >> 4) as i32 | ((((qh[l] as i32) >> 6) & 3) << 4)) - 32;
                y[l] = d * (sc[is] as i8) as f32 * q1 as f32;
                y[l + 32] = d * (sc[2 + is] as i8) as f32 * q2 as f32;
                y[l + 64] = d * (sc[4 + is] as i8) as f32 * q3 as f32;
                y[l + 96] = d * (sc[6 + is] as i8) as f32 * q4 as f32;
            }
        }
    }
}

/// Decode a GGUF tensor payload into `f32` values.
pub fn dequantize(ty: GgmlType, data: &[u8], numel: usize) -> Result<Vec<f32>, MmnError> {
    let expected = ty.data_len(numel)?;
    if data.len() < expected {
        return Err(err(format!(
            "GGUF tensor data truncated: need {expected} bytes for {numel} {ty:?} elements, got {}",
            data.len()
        )));
    }
    let data = &data[..expected];
    let mut out = Vec::with_capacity(numel);
    match ty {
        GgmlType::F32 => {
            for chunk in data.chunks_exact(4) {
                out.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
            }
        }
        GgmlType::F16 => {
            for chunk in data.chunks_exact(2) {
                out.push(f16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        GgmlType::BF16 => {
            for chunk in data.chunks_exact(2) {
                out.push(bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32());
            }
        }
        GgmlType::F64 => {
            for chunk in data.chunks_exact(8) {
                out.push(f64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32);
            }
        }
        GgmlType::I8 => out.extend(data.iter().map(|&b| (b as i8) as f32)),
        GgmlType::I16 => {
            for chunk in data.chunks_exact(2) {
                out.push(i16::from_le_bytes([chunk[0], chunk[1]]) as f32);
            }
        }
        GgmlType::I32 => {
            for chunk in data.chunks_exact(4) {
                out.push(i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32);
            }
        }
        GgmlType::I64 => {
            for chunk in data.chunks_exact(8) {
                out.push(i64::from_le_bytes([
                    chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
                ]) as f32);
            }
        }
        GgmlType::Q4_0 => dequant_q4_0(data, &mut out),
        GgmlType::Q4_1 => dequant_q4_1(data, &mut out),
        GgmlType::Q5_0 => dequant_q5_0(data, &mut out),
        GgmlType::Q5_1 => dequant_q5_1(data, &mut out),
        GgmlType::Q8_0 => dequant_q8_0(data, &mut out),
        GgmlType::Q4K => dequant_q4_k(data, &mut out),
        GgmlType::Q6K => dequant_q6_k(data, &mut out),
    }
    if out.len() != numel {
        return Err(err(format!(
            "GGUF dequantize produced {} values, expected {numel}",
            out.len()
        )));
    }
    Ok(out)
}

/// Quantize `f32` values into Q8_0 blocks (used by the GGUF writer).
pub fn quantize_q8_0(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    if !values.len().is_multiple_of(QK) {
        return Err(err(format!(
            "Q8_0 quantization needs a multiple of {QK} values, got {}",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(values.len() / QK * 34);
    for block in values.chunks_exact(QK) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let d = amax / 127.0;
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for &v in block {
            out.push((v * inv).round().clamp(-127.0, 127.0) as i8 as u8);
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn q8_0_roundtrip_close() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let packed = quantize_q8_0(&values).unwrap();
        let back = dequantize(GgmlType::Q8_0, &packed, 64).unwrap();
        for (a, b) in values.iter().zip(&back) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }

    #[test]
    fn q4_0_known_block() {
        // d = 1.0, all nibbles = 0b1001 (9): value = (9 - 8) * 1.0 = 1.0.
        let mut block = f16::from_f32(1.0).to_le_bytes().to_vec();
        block.extend(std::iter::repeat_n(0x99u8, 16));
        let out = dequantize(GgmlType::Q4_0, &block, 32).unwrap();
        assert!(out.iter().all(|&v| (v - 1.0).abs() < 1e-3));
    }

    #[test]
    fn q4_1_known_block() {
        // d = 0.5, m = 2.0, nibble 3 -> 3 * 0.5 + 2.0 = 3.5.
        let mut block = f16::from_f32(0.5).to_le_bytes().to_vec();
        block.extend_from_slice(&f16::from_f32(2.0).to_le_bytes());
        block.extend(std::iter::repeat_n(0x33u8, 16));
        let out = dequantize(GgmlType::Q4_1, &block, 32).unwrap();
        assert!(out.iter().all(|&v| (v - 3.5).abs() < 1e-2));
    }

    #[test]
    fn q5_0_high_bit_extends_range() {
        // qh all ones -> every value has bit 4 set: x = 16 + nibble.
        let mut block = f16::from_f32(1.0).to_le_bytes().to_vec();
        block.extend_from_slice(&u32::MAX.to_le_bytes());
        block.extend(std::iter::repeat_n(0x00u8, 16));
        let out = dequantize(GgmlType::Q5_0, &block, 32).unwrap();
        // x = 16 | 0 = 16 -> (16 - 16) * d = 0.
        assert!(out.iter().all(|&v| v.abs() < 1e-6));
    }

    #[test]
    fn q5_1_known_block() {
        // d = 1, m = 0, qh zero, nibbles 0x44 -> value 4.
        let mut block = f16::from_f32(1.0).to_le_bytes().to_vec();
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
        block.extend_from_slice(&0u32.to_le_bytes());
        block.extend(std::iter::repeat_n(0x44u8, 16));
        let out = dequantize(GgmlType::Q5_1, &block, 32).unwrap();
        assert!(out.iter().all(|&v| (v - 4.0).abs() < 1e-3));
    }

    #[test]
    fn q4_k_uniform_block() {
        // d=1, dmin=0, scales: sc=1/m=0 for all sub-blocks, nibbles 0x55 -> 5.
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = 1; // sc for sub-blocks 0..4
            scales[j + 4] = 0; // min
            scales[j + 8] = 1; // low nibble sc=1, high nibble min=0 for 4..8
        }
        block.extend_from_slice(&scales);
        block.extend(std::iter::repeat_n(0x55u8, 128));
        let out = dequantize(GgmlType::Q4K, &block, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!(out.iter().all(|&v| (v - 5.0).abs() < 1e-3), "got {:?}", &out[..8]);
    }

    #[test]
    fn q6_k_uniform_block() {
        // ql nibbles 2, qh zero, scales 1, d=1 -> q = 2 - 32 = -30 -> value -30.
        let mut block = Vec::new();
        block.extend(std::iter::repeat_n(0x22u8, 128));
        block.extend(std::iter::repeat_n(0x00u8, 64));
        block.extend(std::iter::repeat_n(1u8, 16));
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let out = dequantize(GgmlType::Q6K, &block, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!(out.iter().all(|&v| (v + 30.0).abs() < 1e-3), "got {:?}", &out[..4]);
    }

    #[test]
    fn f16_bf16_f64_decode() {
        let h = f16::from_f32(1.25).to_le_bytes();
        assert_eq!(dequantize(GgmlType::F16, &h, 1).unwrap(), vec![1.25]);
        let b = bf16::from_f32(-2.0).to_le_bytes();
        assert_eq!(dequantize(GgmlType::BF16, &b, 1).unwrap(), vec![-2.0]);
        let d = 3.5f64.to_le_bytes();
        assert_eq!(dequantize(GgmlType::F64, &d, 1).unwrap(), vec![3.5]);
    }

    #[test]
    fn truncated_data_errors() {
        assert!(dequantize(GgmlType::F32, &[0, 0], 1).is_err());
        assert!(dequantize(GgmlType::Q8_0, &[0; 10], 32).is_err());
    }

    #[test]
    fn misaligned_block_count_errors() {
        // 33 elements cannot fill Q4_0 blocks of 32.
        assert!(GgmlType::Q4_0.data_len(33).is_err());
        assert_eq!(GgmlType::Q4_0.data_len(64).unwrap(), 36);
        assert_eq!(GgmlType::Q4K.data_len(512).unwrap(), 288);
        assert_eq!(GgmlType::Q6K.data_len(256).unwrap(), 210);
    }

    #[test]
    fn unknown_type_id_errors() {
        assert!(GgmlType::from_id(999).is_err());
        assert_eq!(GgmlType::from_id(8).unwrap(), GgmlType::Q8_0);
        assert_eq!(GgmlType::Q6K.type_id(), 14);
    }
}
