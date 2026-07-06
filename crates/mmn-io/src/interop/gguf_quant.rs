//! From-scratch GGML quantization block codecs for GGUF tensors.
//!
//! Implements the on-disk block layouts directly from the format definition —
//! no llama.cpp / ggml code is linked. Supported: F32, F16, BF16, F64, ints,
//! the classic quants (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/Q8_1), the full k-quant
//! family (Q2_K/Q3_K/Q4_K/Q5_K/Q6_K/Q8_K), the lookup quants
//! (IQ4_NL/IQ4_XS), the codebook-grid IQ family (IQ1_S/IQ1_M, IQ2_XXS/XS/S,
//! IQ3_XXS/S — see [`super::gguf_quant_iq`]), ternary BitNet quants
//! (TQ1_0/TQ2_0), MXFP4, and NVFP4.

use super::gguf_quant_iq::{
    dequant_iq1_m, dequant_iq1_s, dequant_iq2_s, dequant_iq2_xs, dequant_iq2_xxs,
    dequant_iq3_s, dequant_iq3_xxs, dequant_nvfp4,
};
use half::{bf16, f16};
use mmn_core::MmnError;

pub const QK: usize = 32; // classic quant block size
pub const QK_K: usize = 256; // k-quant super-block size

/// Non-linear 4-bit codebook shared by IQ4_NL and IQ4_XS.
pub(crate) const KVALUES_IQ4NL: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

/// Doubled E2M1 values used by MXFP4 (positive then negative half).
const KVALUES_MXFP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

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
    Q8_1,
    Q2K,
    Q3K,
    Q4K,
    Q5K,
    Q6K,
    Q8K,
    Iq4Nl,
    Iq4Xs,
    Iq2Xxs,
    Iq2Xs,
    Iq2S,
    Iq3Xxs,
    Iq3S,
    Iq1S,
    Iq1M,
    Tq1_0,
    Tq2_0,
    Mxfp4,
    Nvfp4,
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
            9 => GgmlType::Q8_1,
            10 => GgmlType::Q2K,
            11 => GgmlType::Q3K,
            12 => GgmlType::Q4K,
            13 => GgmlType::Q5K,
            14 => GgmlType::Q6K,
            15 => GgmlType::Q8K,
            16 => GgmlType::Iq2Xxs,
            17 => GgmlType::Iq2Xs,
            18 => GgmlType::Iq3Xxs,
            19 => GgmlType::Iq1S,
            20 => GgmlType::Iq4Nl,
            21 => GgmlType::Iq3S,
            22 => GgmlType::Iq2S,
            23 => GgmlType::Iq4Xs,
            29 => GgmlType::Iq1M,
            34 => GgmlType::Tq1_0,
            35 => GgmlType::Tq2_0,
            39 => GgmlType::Mxfp4,
            40 => GgmlType::Nvfp4,
            24 => GgmlType::I8,
            25 => GgmlType::I16,
            26 => GgmlType::I32,
            27 => GgmlType::I64,
            28 => GgmlType::F64,
            30 => GgmlType::BF16,
            4 | 5 => {
                return Err(err(format!(
                    "GGUF tensor type id {id} (Q4_2/Q4_3) was removed from ggml and cannot be read"
                )));
            }
            31..=33 | 36..=38 => {
                return Err(err(format!(
                    "GGUF tensor type id {id} (repacked Q4_0_x_x / IQ4_NL_4_4) was removed from ggml; re-export the model without runtime repacking"
                )));
            }
            other => {
                return Err(err(format!(
                    "GGUF tensor type id {other} unknown (supported: F32/F16/BF16/F64, Q4_0..Q8_1, Q2_K..Q8_K, IQ1_S/M, IQ2_XXS/XS/S, IQ3_XXS/S, IQ4_NL/XS, TQ1_0, TQ2_0, MXFP4, NVFP4, ints)"
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
            GgmlType::Q8_1 => 9,
            GgmlType::Q2K => 10,
            GgmlType::Q3K => 11,
            GgmlType::Q4K => 12,
            GgmlType::Q5K => 13,
            GgmlType::Q6K => 14,
            GgmlType::Q8K => 15,
            GgmlType::Iq4Nl => 20,
            GgmlType::Iq4Xs => 23,
            GgmlType::Iq2Xxs => 16,
            GgmlType::Iq2Xs => 17,
            GgmlType::Iq3Xxs => 18,
            GgmlType::Iq1S => 19,
            GgmlType::Iq3S => 21,
            GgmlType::Iq2S => 22,
            GgmlType::Iq1M => 29,
            GgmlType::Tq1_0 => 34,
            GgmlType::Tq2_0 => 35,
            GgmlType::Mxfp4 => 39,
            GgmlType::Nvfp4 => 40,
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
            GgmlType::Q8_1 => (QK, 4 + QK),
            GgmlType::Q2K => (QK_K, QK_K / 16 + QK_K / 4 + 4),
            GgmlType::Q3K => (QK_K, QK_K / 8 + QK_K / 4 + 12 + 2),
            GgmlType::Q4K => (QK_K, 2 + 2 + 12 + QK_K / 2),
            GgmlType::Q5K => (QK_K, 2 + 2 + 12 + QK_K / 8 + QK_K / 2),
            GgmlType::Q6K => (QK_K, QK_K / 2 + QK_K / 4 + QK_K / 16 + 2),
            GgmlType::Q8K => (QK_K, 4 + QK_K + QK_K / 16 * 2),
            GgmlType::Iq4Nl => (QK, 2 + QK / 2),
            GgmlType::Iq4Xs => (QK_K, 2 + 2 + QK_K / 64 + QK_K / 2),
            GgmlType::Iq2Xxs => (QK_K, 2 + QK_K / 4),
            GgmlType::Iq2Xs => (QK_K, 2 + QK_K / 4 + QK_K / 32),
            GgmlType::Iq2S => (QK_K, 2 + QK_K / 4 + QK_K / 16),
            GgmlType::Iq3Xxs => (QK_K, 2 + 3 * QK_K / 8),
            GgmlType::Iq3S => (QK_K, 2 + QK_K / 4 + QK_K / 32 + QK_K / 8 + QK_K / 64),
            GgmlType::Iq1S => (QK_K, 2 + QK_K / 8 + QK_K / 16),
            GgmlType::Iq1M => (QK_K, QK_K / 8 + QK_K / 16 + QK_K / 32),
            GgmlType::Tq1_0 => (QK_K, (QK_K - 4 * QK_K / 64) / 5 + QK_K / 64 + 2),
            GgmlType::Tq2_0 => (QK_K, QK_K / 4 + 2),
            GgmlType::Mxfp4 => (QK, 1 + QK / 2),
            GgmlType::Nvfp4 => (64, 4 + QK),
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

fn dequant_q8_1(data: &[u8], out: &mut Vec<f32>) {
    // { f16 d; f16 s (= d * sum, dot-product helper); i8 qs[32] }
    for block in data.chunks_exact(36) {
        let d = f16_at(block, 0);
        for &byte in &block[4..36] {
            out.push((byte as i8) as f32 * d);
        }
    }
}

fn dequant_q2_k(data: &[u8], out: &mut Vec<f32>) {
    // Super-block: u8 scales[16] (4-bit sc | 4-bit min), u8 qs[64], f16 d, f16 dmin.
    for block in data.chunks_exact(84) {
        let scales = &block[0..16];
        let qs = &block[16..80];
        let d = f16_at(block, 80);
        let dmin = f16_at(block, 82);
        let mut is = 0usize;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            for shift in [0u32, 2, 4, 6] {
                for base in [0usize, 16] {
                    let sc = scales[is];
                    is += 1;
                    let dl = d * (sc & 0x0F) as f32;
                    let ml = dmin * (sc >> 4) as f32;
                    for l in 0..16 {
                        out.push(dl * ((q[base + l] >> shift) & 3) as f32 - ml);
                    }
                }
            }
        }
    }
}

/// Unpack Q3_K's 12 packed bytes into 16 signed 6-bit scales.
fn q3_k_scales(packed: &[u8]) -> [i8; 16] {
    let mut aux = [0u32; 4];
    for (i, chunk) in packed.chunks_exact(4).enumerate() {
        aux[i] = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
    }
    const KMASK1: u32 = 0x0303_0303;
    const KMASK2: u32 = 0x0F0F_0F0F;
    let tmp = aux[2];
    aux[2] = ((aux[0] >> 4) & KMASK2) | (((tmp >> 4) & KMASK1) << 4);
    aux[3] = ((aux[1] >> 4) & KMASK2) | (((tmp >> 6) & KMASK1) << 4);
    aux[0] = (aux[0] & KMASK2) | ((tmp & KMASK1) << 4);
    aux[1] = (aux[1] & KMASK2) | (((tmp >> 2) & KMASK1) << 4);
    let mut scales = [0i8; 16];
    for (i, a) in aux.iter().enumerate() {
        for (j, &b) in a.to_le_bytes().iter().enumerate() {
            scales[i * 4 + j] = b as i8;
        }
    }
    scales
}

fn dequant_q3_k(data: &[u8], out: &mut Vec<f32>) {
    // Super-block: u8 hmask[32], u8 qs[64], u8 scales[12] (6-bit packed), f16 d.
    for block in data.chunks_exact(110) {
        let hmask = &block[0..32];
        let qs = &block[32..96];
        let scales = q3_k_scales(&block[96..108]);
        let d_all = f16_at(block, 108);
        let mut is = 0usize;
        let mut m: u8 = 1;
        for half in 0..2 {
            let q = &qs[half * 32..half * 32 + 32];
            for shift in [0u32, 2, 4, 6] {
                for base in [0usize, 16] {
                    let dl = d_all * (scales[is] as f32 - 32.0);
                    is += 1;
                    for l in 0..16 {
                        let low = ((q[base + l] >> shift) & 3) as i32;
                        let high = if hmask[base + l] & m != 0 { 0 } else { 4 };
                        out.push(dl * (low - high) as f32);
                    }
                }
                m <<= 1;
            }
        }
    }
}

fn dequant_q5_k(data: &[u8], out: &mut Vec<f32>) {
    // Super-block: f16 d, f16 dmin, u8 scales[12], u8 qh[32], u8 qs[128].
    for block in data.chunks_exact(176) {
        let d = f16_at(block, 0);
        let dmin = f16_at(block, 2);
        let scales = &block[4..16];
        let qh = &block[16..48];
        let qs = &block[48..176];
        let mut is = 0usize;
        let mut u1: u8 = 1;
        let mut u2: u8 = 2;
        for chunk in qs.chunks_exact(32) {
            let (sc1, m1) = scale_min_k4(is, scales);
            let (sc2, m2) = scale_min_k4(is + 1, scales);
            let d1 = d * sc1 as f32;
            let min1 = dmin * m1 as f32;
            let d2 = d * sc2 as f32;
            let min2 = dmin * m2 as f32;
            for (l, &byte) in chunk.iter().enumerate() {
                let high = if qh[l] & u1 != 0 { 16 } else { 0 };
                out.push(d1 * ((byte & 0x0F) as i32 + high) as f32 - min1);
            }
            for (l, &byte) in chunk.iter().enumerate() {
                let high = if qh[l] & u2 != 0 { 16 } else { 0 };
                out.push(d2 * ((byte >> 4) as i32 + high) as f32 - min2);
            }
            is += 2;
            u1 <<= 2;
            u2 <<= 2;
        }
    }
}

fn dequant_q8_k(data: &[u8], out: &mut Vec<f32>) {
    // Super-block: f32 d, i8 qs[256], i16 bsums[16] (dot-product helper).
    for block in data.chunks_exact(292) {
        let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
        for &byte in &block[4..4 + QK_K] {
            out.push((byte as i8) as f32 * d);
        }
    }
}

fn dequant_iq4_nl(data: &[u8], out: &mut Vec<f32>) {
    // Block of 32: f16 d, u8 qs[16] with a non-linear 4-bit codebook.
    for block in data.chunks_exact(18) {
        let d = f16_at(block, 0);
        let qs = &block[2..18];
        for &byte in qs {
            out.push(d * KVALUES_IQ4NL[(byte & 0x0F) as usize] as f32);
        }
        for &byte in qs {
            out.push(d * KVALUES_IQ4NL[(byte >> 4) as usize] as f32);
        }
    }
}

fn dequant_iq4_xs(data: &[u8], out: &mut Vec<f32>) {
    // Super-block: f16 d, u16 scales_h, u8 scales_l[4], u8 qs[128].
    for block in data.chunks_exact(136) {
        let d = f16_at(block, 0);
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l = &block[4..8];
        let qs = &block[8..136];
        for ib in 0..QK_K / 32 {
            let low = (scales_l[ib / 2] >> (4 * (ib % 2))) & 0x0F;
            let high = ((scales_h >> (2 * ib)) & 3) as u8;
            let ls = (low | (high << 4)) as i32;
            let dl = d * (ls - 32) as f32;
            let q = &qs[ib * 16..ib * 16 + 16];
            for &byte in q {
                out.push(dl * KVALUES_IQ4NL[(byte & 0x0F) as usize] as f32);
            }
            for &byte in q {
                out.push(dl * KVALUES_IQ4NL[(byte >> 4) as usize] as f32);
            }
        }
    }
}

/// Extract one balanced-ternary digit ({-1, 0, 1}) via ggml's fixed-point trick.
fn tq1_digit(byte: u8, pow3: u8) -> f32 {
    let q = byte.wrapping_mul(pow3);
    let xi = ((q as u16 * 3) >> 8) as i32;
    (xi - 1) as f32
}

fn dequant_tq1_0(data: &[u8], out: &mut Vec<f32>) {
    // BitNet ternary: u8 qs[48] (5 trits/byte), u8 qh[4] (4 trits/byte), f16 d.
    const POW3: [u8; 6] = [1, 3, 9, 27, 81, 243];
    for block in data.chunks_exact(54) {
        let qs = &block[0..48];
        let qh = &block[48..52];
        let d = f16_at(block, 52);
        for chunk in [&qs[0..32], &qs[32..48]] {
            for &p in POW3.iter().take(5) {
                for &byte in chunk {
                    out.push(tq1_digit(byte, p) * d);
                }
            }
        }
        for &p in POW3.iter().take(4) {
            for &byte in qh {
                out.push(tq1_digit(byte, p) * d);
            }
        }
    }
}

fn dequant_tq2_0(data: &[u8], out: &mut Vec<f32>) {
    // BitNet ternary, 2 bits per element: u8 qs[64], f16 d.
    for block in data.chunks_exact(66) {
        let qs = &block[0..64];
        let d = f16_at(block, 64);
        for chunk in qs.chunks_exact(32) {
            for shift in [0u32, 2, 4, 6] {
                for &byte in chunk {
                    let q = ((byte >> shift) & 3) as i32;
                    out.push((q - 1) as f32 * d);
                }
            }
        }
    }
}

/// E8M0 exponent-only scale halved (MXFP4 codebook values are doubled).
fn e8m0_to_f32_half(e: u8) -> f32 {
    (2.0f64).powi(e as i32 - 128) as f32
}

fn dequant_mxfp4(data: &[u8], out: &mut Vec<f32>) {
    // Block of 32: u8 e (E8M0 scale), u8 qs[16] of FP4 (E2M1) nibbles.
    for block in data.chunks_exact(17) {
        let d = e8m0_to_f32_half(block[0]);
        let qs = &block[1..17];
        for &byte in qs {
            out.push(d * KVALUES_MXFP4[(byte & 0x0F) as usize] as f32);
        }
        for &byte in qs {
            out.push(d * KVALUES_MXFP4[(byte >> 4) as usize] as f32);
        }
    }
}

/// 6-bit packed (scale, min) pairs used by Q4_K/Q5_K super-blocks
/// (shared with the k-quant encoders).
pub(crate) fn scale_min_k4_pub(j: usize, scales: &[u8]) -> (u8, u8) {
    scale_min_k4(j, scales)
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
        GgmlType::Q8_1 => dequant_q8_1(data, &mut out),
        GgmlType::Q2K => dequant_q2_k(data, &mut out),
        GgmlType::Q3K => dequant_q3_k(data, &mut out),
        GgmlType::Q4K => dequant_q4_k(data, &mut out),
        GgmlType::Q5K => dequant_q5_k(data, &mut out),
        GgmlType::Q6K => dequant_q6_k(data, &mut out),
        GgmlType::Q8K => dequant_q8_k(data, &mut out),
        GgmlType::Iq4Nl => dequant_iq4_nl(data, &mut out),
        GgmlType::Iq4Xs => dequant_iq4_xs(data, &mut out),
        GgmlType::Iq2Xxs => dequant_iq2_xxs(data, &mut out),
        GgmlType::Iq2Xs => dequant_iq2_xs(data, &mut out),
        GgmlType::Iq2S => dequant_iq2_s(data, &mut out),
        GgmlType::Iq3Xxs => dequant_iq3_xxs(data, &mut out),
        GgmlType::Iq3S => dequant_iq3_s(data, &mut out),
        GgmlType::Iq1S => dequant_iq1_s(data, &mut out),
        GgmlType::Iq1M => dequant_iq1_m(data, &mut out),
        GgmlType::Tq1_0 => dequant_tq1_0(data, &mut out),
        GgmlType::Tq2_0 => dequant_tq2_0(data, &mut out),
        GgmlType::Mxfp4 => dequant_mxfp4(data, &mut out),
        GgmlType::Nvfp4 => dequant_nvfp4(data, &mut out),
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

/// Quantize `f32` values into Q4_0 blocks (used by the GGUF writer).
pub fn quantize_q4_0(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    if !values.len().is_multiple_of(QK) {
        return Err(err(format!(
            "Q4_0 quantization needs a multiple of {QK} values, got {}",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(values.len() / QK * 18);
    for block in values.chunks_exact(QK) {
        // ggml picks the signed max so one code lands exactly on it.
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -8.0;
        let inv = if d == 0.0 { 0.0 } else { 1.0 / d };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        for l in 0..QK / 2 {
            let x0 = (block[l] * inv + 8.5).clamp(0.0, 15.0) as u8;
            let x1 = (block[l + QK / 2] * inv + 8.5).clamp(0.0, 15.0) as u8;
            out.push(x0 | (x1 << 4));
        }
    }
    Ok(out)
}

/// Encode `f32` values as raw little-endian F16 (used by the GGUF writer).
pub fn encode_f16(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 2);
    for &v in values {
        out.extend_from_slice(&f16::from_f32(v).to_le_bytes());
    }
    out
}

fn require_block_multiple(values: &[f32], block: usize, what: &str) -> Result<(), MmnError> {
    if !values.len().is_multiple_of(block) {
        return Err(block_multiple_err(what, block, values.len()));
    }
    Ok(())
}

/// Shared block-alignment error (used by the IQ encoders too).
pub(crate) fn block_multiple_err(what: &str, block: usize, got: usize) -> MmnError {
    err(format!(
        "{what} quantization needs a multiple of {block} values, got {got}"
    ))
}

/// Quantize into Q4_1 blocks (min + scale), matching ggml byte-for-byte.
pub fn quantize_q4_1(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    require_block_multiple(values, QK, "Q4_1")?;
    let mut out = Vec::with_capacity(values.len() / QK * 20);
    for block in values.chunks_exact(QK) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in block {
            min = min.min(v);
            max = max.max(v);
        }
        let d = (max - min) / 15.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(min).to_le_bytes());
        for j in 0..QK / 2 {
            let x0 = (block[j] - min) * id;
            let x1 = (block[j + QK / 2] - min) * id;
            // ggml rounds with (int8_t)(x + 0.5f): truncation after +0.5.
            let xi0 = ((x0 + 0.5) as i32).min(15) as u8;
            let xi1 = ((x1 + 0.5) as i32).min(15) as u8;
            out.push(xi0 | (xi1 << 4));
        }
    }
    Ok(out)
}

/// Quantize into Q5_0 blocks (signed range + high bits), ggml-exact.
pub fn quantize_q5_0(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    require_block_multiple(values, QK, "Q5_0")?;
    let mut out = Vec::with_capacity(values.len() / QK * 22);
    for block in values.chunks_exact(QK) {
        let mut amax = 0.0f32;
        let mut max = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        let d = max / -16.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        let mut qh = 0u32;
        let mut qs = [0u8; QK / 2];
        for (j, slot) in qs.iter_mut().enumerate() {
            let x0 = block[j] * id;
            let x1 = block[j + QK / 2] * id;
            let xi0 = ((x0 + 16.5) as i32).min(31) as u8;
            let xi1 = ((x1 + 16.5) as i32).min(31) as u8;
            *slot = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK / 2);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

/// Quantize into Q5_1 blocks (min + scale + high bits), ggml-exact.
pub fn quantize_q5_1(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    require_block_multiple(values, QK, "Q5_1")?;
    let mut out = Vec::with_capacity(values.len() / QK * 24);
    for block in values.chunks_exact(QK) {
        let mut min = f32::MAX;
        let mut max = f32::MIN;
        for &v in block {
            min = min.min(v);
            max = max.max(v);
        }
        let d = (max - min) / 31.0;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(min).to_le_bytes());
        let mut qh = 0u32;
        let mut qs = [0u8; QK / 2];
        for (j, slot) in qs.iter_mut().enumerate() {
            let x0 = (block[j] - min) * id;
            let x1 = (block[j + QK / 2] - min) * id;
            let xi0 = (x0 + 0.5) as u8;
            let xi1 = (x1 + 0.5) as u8;
            *slot = (xi0 & 0x0F) | ((xi1 & 0x0F) << 4);
            qh |= (((xi0 & 0x10) >> 4) as u32) << j;
            qh |= (((xi1 & 0x10) >> 4) as u32) << (j + QK / 2);
        }
        out.extend_from_slice(&qh.to_le_bytes());
        out.extend_from_slice(&qs);
    }
    Ok(out)
}

/// Quantize into MXFP4 blocks (E8M0 scale + FP4 codebook), reference-exact.
pub fn quantize_mxfp4(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    require_block_multiple(values, QK, "MXFP4")?;
    let mut out = Vec::with_capacity(values.len() / QK * 17);
    for block in values.chunks_exact(QK) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let e: u8 = if amax > 0.0 {
            (amax.log2().floor() as i32 - 2 + 127) as u8
        } else {
            0
        };
        let d = e8m0_to_f32_half(e);
        out.push(e);
        // Nearest codebook value (first index wins ties, like np.argmin).
        let quantize_one = |v: f32| -> u8 {
            let mut best = 0usize;
            let mut best_err = f32::INFINITY;
            for (i, &k) in KVALUES_MXFP4.iter().enumerate() {
                let err_i = (d * k as f32 - v).abs();
                if err_i < best_err {
                    best_err = err_i;
                    best = i;
                }
            }
            best as u8
        };
        for j in 0..QK / 2 {
            let lo = quantize_one(block[j]);
            let hi = quantize_one(block[j + QK / 2]);
            out.push(lo | (hi << 4));
        }
    }
    Ok(out)
}

/// Quantize into ternary TQ1_0 blocks (5 trits/byte), reference-exact.
pub fn quantize_tq1_0(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    require_block_multiple(values, QK_K, "TQ1_0")?;
    let mut out = Vec::with_capacity(values.len() / QK_K * 54);
    for block in values.chunks_exact(QK_K) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let id = if amax != 0.0 { 1.0 / amax } else { 0.0 };
        // Trits in 0..=2 (round half away from zero, then +1).
        let trit = |v: f32| -> u16 { ((v * id).round() as i32 + 1) as u16 };
        let scale = |q: u16| -> u8 { (q as u32 * 256).div_ceil(243) as u8 };
        // Elements 0..160: 32 bytes, weights [81,27,9,3,1] over strides of 32.
        for m in 0..32 {
            let q: u16 = (0..5).map(|k| trit(block[k * 32 + m]) * [81, 27, 9, 3, 1][k]).sum();
            out.push(scale(q));
        }
        // Elements 160..240: 16 bytes over strides of 16.
        for m in 0..16 {
            let q: u16 = (0..5)
                .map(|k| trit(block[160 + k * 16 + m]) * [81, 27, 9, 3, 1][k])
                .sum();
            out.push(scale(q));
        }
        // Elements 240..256: 4 bytes over strides of 4, weights [81,27,9,3].
        for m in 0..4 {
            let q: u16 = (0..4)
                .map(|k| trit(block[240 + k * 4 + m]) * [81, 27, 9, 3][k])
                .sum();
            out.push(scale(q));
        }
        out.extend_from_slice(&f16::from_f32(amax).to_le_bytes());
    }
    Ok(out)
}

/// Quantize into ternary TQ2_0 blocks ({-1,0,1} at 2 bits), ggml-exact.
pub fn quantize_tq2_0(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    require_block_multiple(values, QK_K, "TQ2_0")?;
    let mut out = Vec::with_capacity(values.len() / QK_K * 66);
    for block in values.chunks_exact(QK_K) {
        let amax = block.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        let id = if amax != 0.0 { 1.0 / amax } else { 0.0 };
        let mut qs = [0u8; QK_K / 4];
        for (chunk_idx, chunk) in block.chunks_exact(128).enumerate() {
            for m in 0..32 {
                let mut q = 0u8;
                for n in 0..4 {
                    // lroundf: round half away from zero.
                    let xi = (chunk[m + n * 32] * id).round() as i32 + 1;
                    q += ((xi & 3) as u8) << (2 * n);
                }
                qs[chunk_idx * 32 + m] = q;
            }
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&f16::from_f32(amax).to_le_bytes());
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
    fn q8_1_known_block() {
        // d = 0.5, s ignored, all quants = 4 -> 2.0.
        let mut block = f16::from_f32(0.5).to_le_bytes().to_vec();
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
        block.extend(std::iter::repeat_n(4u8, 32));
        let out = dequantize(GgmlType::Q8_1, &block, 32).unwrap();
        assert!(out.iter().all(|&v| (v - 2.0).abs() < 1e-3));
    }

    #[test]
    fn q2_k_uniform_block() {
        // scales: sc=2 (low nibble), min=1 (high nibble); qs all 0b01 -> q=1.
        // d=1, dmin=0.5 -> value = 1*2*1 - 0.5*1 = 1.5.
        let mut block = Vec::new();
        block.extend(std::iter::repeat_n(0x12u8, 16)); // min=1, sc=2
        block.extend(std::iter::repeat_n(0b0101_0101u8, 64));
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(0.5).to_le_bytes());
        assert_eq!(block.len(), 84);
        let out = dequantize(GgmlType::Q2K, &block, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!(out.iter().all(|&v| (v - 1.5).abs() < 1e-3), "got {:?}", &out[..4]);
    }

    #[test]
    fn q3_k_uniform_block() {
        // hmask all ones -> high bit set -> no -4; qs all 0b10 -> q=2.
        // scales packed so every 6-bit scale = 34 -> dl = d * (34-32) = 2.
        // aux packing: low 4 bits in scales[0..8], high 2 bits in scales[8..12].
        let mut scales = [0u8; 12];
        for s in scales.iter_mut().take(8) {
            *s = 0x22; // low nibbles = 2 for scales 0..16
        }
        for s in scales.iter_mut().skip(8) {
            *s = 0b1010_1010; // every 2-bit high field = 0b10 -> +32
        }
        let mut block = Vec::new();
        block.extend(std::iter::repeat_n(0xFFu8, 32)); // hmask
        block.extend(std::iter::repeat_n(0b1010_1010u8, 64)); // qs 2-bit = 2
        block.extend_from_slice(&scales);
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        assert_eq!(block.len(), 110);
        let out = dequantize(GgmlType::Q3K, &block, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        // dl = 1 * (34 - 32) = 2; value = dl * (2 - 0) = 4.
        assert!(out.iter().all(|&v| (v - 4.0).abs() < 1e-3), "got {:?}", &out[..4]);
    }

    #[test]
    fn q5_k_uniform_block() {
        // d=1, dmin=0, scales sc=1/min=0, qh zero, nibbles 0x33 -> value 3.
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = 1;
            scales[j + 4] = 0;
            scales[j + 8] = 1;
        }
        block.extend_from_slice(&scales);
        block.extend(std::iter::repeat_n(0x00u8, 32)); // qh
        block.extend(std::iter::repeat_n(0x33u8, 128)); // qs
        assert_eq!(block.len(), 176);
        let out = dequantize(GgmlType::Q5K, &block, QK_K).unwrap();
        assert!(out.iter().all(|&v| (v - 3.0).abs() < 1e-3), "got {:?}", &out[..4]);
    }

    #[test]
    fn q5_k_high_bit_adds_16() {
        // Same as above but qh all ones: every value gets +16 -> 19.
        let mut block = Vec::new();
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        block.extend_from_slice(&f16::from_f32(0.0).to_le_bytes());
        let mut scales = [0u8; 12];
        for j in 0..4 {
            scales[j] = 1;
            scales[j + 8] = 1;
        }
        block.extend_from_slice(&scales);
        block.extend(std::iter::repeat_n(0xFFu8, 32));
        block.extend(std::iter::repeat_n(0x33u8, 128));
        let out = dequantize(GgmlType::Q5K, &block, QK_K).unwrap();
        assert!(out.iter().all(|&v| (v - 19.0).abs() < 1e-3), "got {:?}", &out[..4]);
    }

    #[test]
    fn q8_k_uniform_block() {
        let mut block = 2.0f32.to_le_bytes().to_vec();
        block.extend(std::iter::repeat_n(3u8, QK_K));
        block.extend(std::iter::repeat_n(0u8, 32)); // bsums (unused)
        assert_eq!(block.len(), 292);
        let out = dequantize(GgmlType::Q8K, &block, QK_K).unwrap();
        assert!(out.iter().all(|&v| (v - 6.0).abs() < 1e-4));
    }

    #[test]
    fn iq4_nl_codebook_lookup() {
        // nibble 8 -> kvalue 1; nibble 0 -> kvalue -127.
        let mut block = f16::from_f32(2.0).to_le_bytes().to_vec();
        block.extend(std::iter::repeat_n(0x08u8, 16)); // low=8, high=0
        let out = dequantize(GgmlType::Iq4Nl, &block, 32).unwrap();
        for v in &out[..16] {
            assert!((v - 2.0).abs() < 1e-3, "low nibble -> 1 * 2");
        }
        for v in &out[16..] {
            assert!((v + 254.0).abs() < 1e-1, "high nibble -> -127 * 2");
        }
    }

    #[test]
    fn iq4_xs_uniform_block() {
        // ls = 33 for every sub-block: low nibble 1, high bits 1 -> 1 | (1<<4)? No:
        // low = 1, high = 1 -> ls = 1 | 16 = 17 -> dl = d * (17-32) = -15 * d.
        let mut block = f16::from_f32(1.0).to_le_bytes().to_vec();
        block.extend_from_slice(&0b0101_0101_0101_0101u16.to_le_bytes()); // scales_h: all 2-bit = 1
        block.extend(std::iter::repeat_n(0x11u8, 4)); // scales_l: all nibbles 1
        block.extend(std::iter::repeat_n(0x88u8, 128)); // all nibbles 8 -> kvalue 1
        assert_eq!(block.len(), 136);
        let out = dequantize(GgmlType::Iq4Xs, &block, QK_K).unwrap();
        assert!(out.iter().all(|&v| (v + 15.0).abs() < 1e-3), "got {:?}", &out[..4]);
    }

    #[test]
    fn tq1_0_zero_and_positive_trits() {
        // qs = 0 encodes all trits 0 -> q value (0*3)>>8 = 0 -> -1 * d... check:
        // byte 0: q = 0, xi = 0, value = (0-1)*d = -d for every element.
        let mut block = vec![0u8; 52];
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        assert_eq!(block.len(), 54);
        let out = dequantize(GgmlType::Tq1_0, &block, QK_K).unwrap();
        assert_eq!(out.len(), QK_K);
        assert!(out.iter().all(|&v| (v + 1.0).abs() < 1e-4));
    }

    #[test]
    fn tq2_0_all_codes() {
        // qs byte 0b10_01_00_11: shifts 0,2,4,6 give 3,0,1,2 -> values 2,-1,0,1.
        let mut block = vec![0b1001_0011u8; 64];
        block.extend_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let out = dequantize(GgmlType::Tq2_0, &block, QK_K).unwrap();
        // Element order: 32 values at shift 0 (=3-1=2), then shift 2 (0-1=-1), ...
        assert!((out[0] - 2.0).abs() < 1e-4);
        assert!((out[32] + 1.0).abs() < 1e-4);
        assert!((out[64] - 0.0).abs() < 1e-4);
        assert!((out[96] - 1.0).abs() < 1e-4);
    }

    #[test]
    fn mxfp4_codebook_and_scale() {
        // e = 129 -> scale 2^(129-128) = 2; nibble 5 -> kvalue 6 -> 12.
        let mut block = vec![129u8];
        block.extend(std::iter::repeat_n(0x55u8, 16));
        let out = dequantize(GgmlType::Mxfp4, &block, 32).unwrap();
        assert!(out.iter().all(|&v| (v - 12.0).abs() < 1e-3), "got {:?}", &out[..4]);
        // Negative half of the codebook: nibble 0xD -> -6 -> -12.
        let mut block = vec![129u8];
        block.extend(std::iter::repeat_n(0xDDu8, 16));
        let out = dequantize(GgmlType::Mxfp4, &block, 32).unwrap();
        assert!(out.iter().all(|&v| (v + 12.0).abs() < 1e-3));
    }

    #[test]
    fn mxfp4_quantize_roundtrip_close() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 30.0) * 0.1).collect();
        let packed = quantize_mxfp4(&values).unwrap();
        assert_eq!(packed.len(), 34);
        let back = dequantize(GgmlType::Mxfp4, &packed, 64).unwrap();
        // FP4 is coarse: verify within one codebook step of the scale.
        for (a, b) in values.iter().zip(&back) {
            assert!((a - b).abs() < 1.1, "{a} vs {b}");
        }
    }

    #[test]
    fn tq1_0_quantize_roundtrip_exact_on_ternary() {
        let values: Vec<f32> = (0..256).map(|i| ((i % 3) as f32) - 1.0).collect();
        let packed = quantize_tq1_0(&values).unwrap();
        assert_eq!(packed.len(), 54);
        let back = dequantize(GgmlType::Tq1_0, &packed, 256).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn tq2_0_quantize_roundtrip_exact_on_ternary() {
        let values: Vec<f32> = (0..256).map(|i| (((i * 7) % 3) as f32) - 1.0).collect();
        let packed = quantize_tq2_0(&values).unwrap();
        let back = dequantize(GgmlType::Tq2_0, &packed, 256).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn q4_1_q5_0_q5_1_quantize_roundtrip_close() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 30.0) * 0.05).collect();
        for (packed, ty, tol) in [
            (quantize_q4_1(&values).unwrap(), GgmlType::Q4_1, 0.15),
            (quantize_q5_0(&values).unwrap(), GgmlType::Q5_0, 0.12),
            (quantize_q5_1(&values).unwrap(), GgmlType::Q5_1, 0.07),
        ] {
            let back = dequantize(ty, &packed, 64).unwrap();
            for (a, b) in values.iter().zip(&back) {
                assert!((a - b).abs() < tol, "{ty:?}: {a} vs {b}");
            }
        }
    }

    #[test]
    fn q4_0_quantize_roundtrip_close() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let packed = quantize_q4_0(&values).unwrap();
        assert_eq!(packed.len(), 36);
        let back = dequantize(GgmlType::Q4_0, &packed, 64).unwrap();
        for (a, b) in values.iter().zip(&back) {
            assert!((a - b).abs() < 0.25, "{a} vs {b}");
        }
    }

    #[test]
    fn f16_encode_roundtrip() {
        let values = vec![0.5f32, -1.25, 3.0];
        let bytes = encode_f16(&values);
        let back = dequantize(GgmlType::F16, &bytes, 3).unwrap();
        assert_eq!(back, values);
    }

    #[test]
    fn removed_type_ids_error_clearly() {
        for id in [4u32, 5, 31, 32, 33, 36, 37, 38] {
            let e = GgmlType::from_id(id).err().unwrap();
            assert!(e.message().contains("removed"), "{id}: {}", e.message());
        }
    }

    #[test]
    fn iq_grid_type_ids_resolve() {
        assert_eq!(GgmlType::from_id(16).unwrap(), GgmlType::Iq2Xxs);
        assert_eq!(GgmlType::from_id(17).unwrap(), GgmlType::Iq2Xs);
        assert_eq!(GgmlType::from_id(18).unwrap(), GgmlType::Iq3Xxs);
        assert_eq!(GgmlType::from_id(19).unwrap(), GgmlType::Iq1S);
        assert_eq!(GgmlType::from_id(21).unwrap(), GgmlType::Iq3S);
        assert_eq!(GgmlType::from_id(22).unwrap(), GgmlType::Iq2S);
        assert_eq!(GgmlType::from_id(29).unwrap(), GgmlType::Iq1M);
        assert_eq!(GgmlType::from_id(40).unwrap(), GgmlType::Nvfp4);
    }

    #[test]
    fn every_supported_type_id_roundtrips_through_from_id() {
        // Guards against dropping ids during refactors (IQ4_XS regression).
        for ty in [
            GgmlType::F32,
            GgmlType::F16,
            GgmlType::BF16,
            GgmlType::F64,
            GgmlType::Q4_0,
            GgmlType::Q4_1,
            GgmlType::Q5_0,
            GgmlType::Q5_1,
            GgmlType::Q8_0,
            GgmlType::Q8_1,
            GgmlType::Q2K,
            GgmlType::Q3K,
            GgmlType::Q4K,
            GgmlType::Q5K,
            GgmlType::Q6K,
            GgmlType::Q8K,
            GgmlType::Iq4Nl,
            GgmlType::Iq4Xs,
            GgmlType::Iq2Xxs,
            GgmlType::Iq2Xs,
            GgmlType::Iq2S,
            GgmlType::Iq3Xxs,
            GgmlType::Iq3S,
            GgmlType::Iq1S,
            GgmlType::Iq1M,
            GgmlType::Tq1_0,
            GgmlType::Tq2_0,
            GgmlType::Mxfp4,
            GgmlType::Nvfp4,
            GgmlType::I8,
            GgmlType::I16,
            GgmlType::I32,
            GgmlType::I64,
        ] {
            assert_eq!(GgmlType::from_id(ty.type_id()).unwrap(), ty, "{ty:?}");
        }
    }

    #[test]
    fn iq_block_layouts_match_spec_sizes() {
        assert_eq!(GgmlType::Iq2Xxs.block_layout(), (256, 66));
        assert_eq!(GgmlType::Iq2Xs.block_layout(), (256, 74));
        assert_eq!(GgmlType::Iq2S.block_layout(), (256, 82));
        assert_eq!(GgmlType::Iq3Xxs.block_layout(), (256, 98));
        assert_eq!(GgmlType::Iq3S.block_layout(), (256, 110));
        assert_eq!(GgmlType::Iq1S.block_layout(), (256, 50));
        assert_eq!(GgmlType::Iq1M.block_layout(), (256, 56));
        assert_eq!(GgmlType::Nvfp4.block_layout(), (64, 36));
    }

    #[test]
    fn iq_dequant_dispatch_produces_full_blocks() {
        for ty in [
            GgmlType::Iq2Xxs,
            GgmlType::Iq2Xs,
            GgmlType::Iq2S,
            GgmlType::Iq3Xxs,
            GgmlType::Iq3S,
            GgmlType::Iq1S,
            GgmlType::Iq1M,
        ] {
            let (elems, bytes) = ty.block_layout();
            let data = vec![0u8; bytes];
            let out = dequantize(ty, &data, elems).unwrap();
            assert_eq!(out.len(), elems, "{ty:?}");
            assert!(out.iter().all(|v| v.is_finite()), "{ty:?}");
        }
        let out = dequantize(GgmlType::Nvfp4, &[0u8; 36], 64).unwrap();
        assert_eq!(out.len(), 64);
    }

    #[test]
    fn k_quant_block_layouts_match_ggml_sizes() {
        assert_eq!(GgmlType::Q2K.block_layout(), (256, 84));
        assert_eq!(GgmlType::Q3K.block_layout(), (256, 110));
        assert_eq!(GgmlType::Q5K.block_layout(), (256, 176));
        assert_eq!(GgmlType::Q8K.block_layout(), (256, 292));
        assert_eq!(GgmlType::Iq4Nl.block_layout(), (32, 18));
        assert_eq!(GgmlType::Iq4Xs.block_layout(), (256, 136));
        assert_eq!(GgmlType::Tq1_0.block_layout(), (256, 54));
        assert_eq!(GgmlType::Tq2_0.block_layout(), (256, 66));
        assert_eq!(GgmlType::Mxfp4.block_layout(), (32, 17));
        assert_eq!(GgmlType::Q8_1.block_layout(), (32, 36));
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
