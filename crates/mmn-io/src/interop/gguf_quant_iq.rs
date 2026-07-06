//! IQ-family GGML dequantizers (QuIP#-style codebook quants) and NVFP4.
//!
//! Ported from the GGUF quantization spec: each 256-element super-block
//! combines codebook grid rows (see [`super::gguf_iq_grids`]) with packed
//! sign bits and per-group scales. `IQ1_*` add a per-row delta instead of
//! signs. NVFP4 is 16-element FP4 (E2M1) blocks under unsigned-E4M3 scales.

use super::gguf_iq_grids::{ksigns, IqGrid, IQ1_S, IQ2_S, IQ2_XS, IQ2_XXS, IQ3_S, IQ3_XXS};
use half::f16;
use std::sync::OnceLock;

pub const QK_K: usize = 256;
const IQ1_DELTA: f32 = 0.125;

fn grid_values(cell: &'static OnceLock<Vec<f32>>, grid: &IqGrid) -> &'static [f32] {
    cell.get_or_init(|| grid.decode())
}

fn iq2_xxs_grid() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    grid_values(&CELL, &IQ2_XXS)
}

fn iq2_xs_grid() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    grid_values(&CELL, &IQ2_XS)
}

fn iq2_s_grid() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    grid_values(&CELL, &IQ2_S)
}

fn iq3_xxs_grid() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    grid_values(&CELL, &IQ3_XXS)
}

fn iq3_s_grid() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    grid_values(&CELL, &IQ3_S)
}

fn iq1_grid() -> &'static [f32] {
    static CELL: OnceLock<Vec<f32>> = OnceLock::new();
    grid_values(&CELL, &IQ1_S)
}

fn f16_at(data: &[u8], pos: usize) -> f32 {
    f16::from_le_bytes([data[pos], data[pos + 1]]).to_f32()
}

fn u16_at(data: &[u8], pos: usize) -> u16 {
    u16::from_le_bytes([data[pos], data[pos + 1]])
}

fn u32_at(data: &[u8], pos: usize) -> u32 {
    u32::from_le_bytes([data[pos], data[pos + 1], data[pos + 2], data[pos + 3]])
}

fn sign_of(bits: u8, j: usize) -> f32 {
    if bits >> j & 1 != 0 {
        -1.0
    } else {
        1.0
    }
}

/// IQ2_XXS: `{ f16 d; u16 qs[32] }` — 4 u16 per 32-element group.
pub fn dequant_iq2_xxs(data: &[u8], out: &mut Vec<f32>) {
    let grid = iq2_xxs_grid();
    for block in data.chunks_exact(66) {
        let d = f16_at(block, 0);
        for g in 0..QK_K / 32 {
            let base = 2 + g * 8;
            let aux0 = u32_at(block, base);
            let aux1 = u32_at(block, base + 4);
            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let row = (aux0 >> (8 * l)) as u8 as usize;
                let signs = ksigns((aux1 >> (7 * l)) as usize & 127);
                for j in 0..8 {
                    out.push(db * grid[row * 8 + j] * sign_of(signs, j));
                }
            }
        }
    }
}

/// IQ2_XS: `{ f16 d; u16 qs[32]; u8 scales[8] }` — 9-bit grid + 7-bit signs.
pub fn dequant_iq2_xs(data: &[u8], out: &mut Vec<f32>) {
    let grid = iq2_xs_grid();
    for block in data.chunks_exact(74) {
        let d = f16_at(block, 0);
        let scales = &block[66..74];
        for entry in 0..32 {
            let q = u16_at(block, 2 + entry * 2);
            let nibble = (scales[entry / 4] >> (4 * ((entry / 2) % 2))) & 0x0F;
            let db = d * (0.5 + nibble as f32) * 0.25;
            let row = (q & 511) as usize;
            let signs = ksigns((q >> 9) as usize);
            for j in 0..8 {
                out.push(db * grid[row * 8 + j] * sign_of(signs, j));
            }
        }
    }
}

/// IQ2_S: `{ f16 d; u8 qs[32]; u8 signs[32]; u8 qh[8]; u8 scales[8] }`.
pub fn dequant_iq2_s(data: &[u8], out: &mut Vec<f32>) {
    let grid = iq2_s_grid();
    for block in data.chunks_exact(82) {
        let d = f16_at(block, 0);
        let qs = &block[2..34];
        let signs = &block[34..66];
        let qh = &block[66..74];
        let scales = &block[74..82];
        for entry in 0..32 {
            let high = (qh[entry / 4] >> (2 * (entry % 4))) & 3;
            let row = qs[entry] as usize | ((high as usize) << 8);
            let nibble = (scales[entry / 4] >> (4 * ((entry / 2) % 2))) & 0x0F;
            let db = d * (0.5 + nibble as f32) * 0.25;
            let sign_bits = signs[entry];
            for j in 0..8 {
                out.push(db * grid[row * 8 + j] * sign_of(sign_bits, j));
            }
        }
    }
}

/// IQ3_XXS: `{ f16 d; u8 qs[64]; u32 sas[8] }` — 4-wide grid rows.
pub fn dequant_iq3_xxs(data: &[u8], out: &mut Vec<f32>) {
    let grid = iq3_xxs_grid();
    for block in data.chunks_exact(98) {
        let d = f16_at(block, 0);
        let qs = &block[2..66];
        for g in 0..QK_K / 32 {
            let sas = u32_at(block, 66 + g * 4);
            let db = d * (0.5 + (sas >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = ksigns((sas >> (7 * l)) as usize & 127);
                for j in 0..8 {
                    let row = qs[g * 8 + l * 2 + j / 4] as usize;
                    out.push(db * grid[row * 4 + j % 4] * sign_of(signs, j));
                }
            }
        }
    }
}

/// IQ3_S: `{ f16 d; u8 qs[64]; u8 qh[8]; u8 signs[32]; u8 scales[4] }`.
pub fn dequant_iq3_s(data: &[u8], out: &mut Vec<f32>) {
    let grid = iq3_s_grid();
    for block in data.chunks_exact(110) {
        let d = f16_at(block, 0);
        let qs = &block[2..66];
        let qh = &block[66..74];
        let signs = &block[74..106];
        let scales = &block[106..110];
        for g in 0..QK_K / 32 {
            let nibble = (scales[g / 2] >> (4 * (g % 2))) & 0x0F;
            let db = d * (1.0 + 2.0 * nibble as f32);
            for l in 0..4 {
                let sign_bits = signs[g * 4 + l];
                for j in 0..8 {
                    let r = g * 8 + l * 2 + j / 4;
                    let row = qs[r] as usize | (((qh[r / 8] >> (r % 8)) as usize & 1) << 8);
                    out.push(db * grid[row * 4 + j % 4] * sign_of(sign_bits, j));
                }
            }
        }
    }
}

/// IQ1_S: `{ f16 d; u8 qs[32]; u16 qh[8] }` — grid of {-1,0,1} plus delta.
pub fn dequant_iq1_s(data: &[u8], out: &mut Vec<f32>) {
    let grid = iq1_grid();
    for block in data.chunks_exact(50) {
        let d = f16_at(block, 0);
        let qs = &block[2..34];
        for g in 0..QK_K / 32 {
            let h = u16_at(block, 34 + g * 2);
            let dl = d * (2.0 * ((h >> 12) & 7) as f32 + 1.0);
            let delta = if h & 0x8000 != 0 { -IQ1_DELTA } else { IQ1_DELTA };
            for l in 0..4 {
                let row = qs[g * 4 + l] as usize | (((h >> (3 * l)) as usize & 7) << 8);
                for j in 0..8 {
                    out.push(dl * (grid[row * 8 + j] + delta));
                }
            }
        }
    }
}

/// IQ1_M: `{ u8 qs[32]; u8 qh[16]; u16 scales[4] }` — f16 scale packed in scales.
pub fn dequant_iq1_m(data: &[u8], out: &mut Vec<f32>) {
    let grid = iq1_grid();
    for block in data.chunks_exact(56) {
        let qs = &block[0..32];
        let qh = &block[32..48];
        let sc: [u16; 4] = [
            u16_at(block, 48),
            u16_at(block, 50),
            u16_at(block, 52),
            u16_at(block, 54),
        ];
        // The f16 super-scale hides in the top nibbles of the four u16s.
        let d_bits = ((sc[0] & 0xF000) >> 12)
            | ((sc[1] & 0xF000) >> 8)
            | ((sc[2] & 0xF000) >> 4)
            | (sc[3] & 0xF000);
        let d = f16::from_bits(d_bits).to_f32();
        for r in 0..32 {
            // 16 3-bit scales (shifts 0,3,6,9 within each u16); one per 16 elems.
            let half = r / 2;
            let scale = (sc[half / 4] >> (3 * (half % 4))) & 0x07;
            let dl = d * (2.0 * scale as f32 + 1.0);
            let nib = (qh[r / 2] >> (4 * (r % 2))) & 0x0F;
            let row = qs[r] as usize | (((nib & 7) as usize) << 8);
            let delta = if nib & 8 != 0 { -IQ1_DELTA } else { IQ1_DELTA };
            for j in 0..8 {
                out.push(dl * (grid[row * 8 + j] + delta));
            }
        }
    }
}

/// Unsigned E4M3 (bias 7) to f32, halved (NVFP4 codebook values are doubled).
fn ue4m3_to_f32_half(x: u8) -> f32 {
    if x == 0 || x == 0x7F {
        return 0.0;
    }
    let exp = (x >> 3) & 0x0F;
    let man = (x & 0x07) as f32;
    let raw = if exp == 0 {
        man * (2.0f32).powi(-9)
    } else {
        (1.0 + man / 8.0) * (2.0f32).powi(exp as i32 - 7)
    };
    raw * 0.5
}

/// Doubled E2M1 values (same codebook as MXFP4).
const KVALUES_FP4: [i8; 16] = [0, 1, 2, 3, 4, 6, 8, 12, 0, -1, -2, -3, -4, -6, -8, -12];

/// NVFP4: 64-element super-block `{ u8 e4m3_scales[4]; u8 qs[32] }`.
pub fn dequant_nvfp4(data: &[u8], out: &mut Vec<f32>) {
    for block in data.chunks_exact(36) {
        let scales = &block[0..4];
        let qs = &block[4..36];
        for sub in 0..4 {
            let d = ue4m3_to_f32_half(scales[sub]);
            let q = &qs[sub * 8..sub * 8 + 8];
            for &byte in q {
                out.push(d * KVALUES_FP4[(byte & 0x0F) as usize] as f32);
            }
            for &byte in q {
                out.push(d * KVALUES_FP4[(byte >> 4) as usize] as f32);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn grids_decode_to_expected_shapes() {
        assert_eq!(iq2_xxs_grid().len(), 256 * 8);
        assert_eq!(iq2_xs_grid().len(), 512 * 8);
        assert_eq!(iq2_s_grid().len(), 1024 * 8);
        assert_eq!(iq3_xxs_grid().len(), 256 * 4);
        assert_eq!(iq3_s_grid().len(), 512 * 4);
        assert_eq!(iq1_grid().len(), 2048 * 8);
        // IQ2 grids hold only the mapped magnitudes {8, 25, 43}.
        assert!(iq2_xxs_grid().iter().all(|v| [8.0, 25.0, 43.0].contains(v)));
        // IQ1 grids hold ternary values.
        assert!(iq1_grid().iter().all(|v| [-1.0, 0.0, 1.0].contains(v)));
    }

    #[test]
    fn ksigns_matches_parity_construction() {
        // First entries of the reference table: 0x00, 0x81, 0x82, 0x03.
        assert_eq!(ksigns(0), 0x00);
        assert_eq!(ksigns(1), 0x81);
        assert_eq!(ksigns(2), 0x82);
        assert_eq!(ksigns(3), 0x03);
        assert_eq!(ksigns(127), 0xFF);
    }

    #[test]
    fn iq2_xxs_zero_block_is_finite() {
        let mut block = vec![0u8; 66];
        block[..2].copy_from_slice(&f16::from_f32(1.0).to_le_bytes());
        let mut out = Vec::new();
        dequant_iq2_xxs(&block, &mut out);
        assert_eq!(out.len(), 256);
        assert!(out.iter().all(|v| v.is_finite()));
        // db = 1 * (0.5 + 0) * 0.25 = 0.125; grid row 0 elems from the table.
        assert!((out[0] - 0.125 * iq2_xxs_grid()[0]).abs() < 1e-4);
    }

    #[test]
    fn iq1_s_delta_and_scale() {
        let mut block = vec![0u8; 50];
        block[..2].copy_from_slice(&f16::from_f32(2.0).to_le_bytes());
        let mut out = Vec::new();
        dequant_iq1_s(&block, &mut out);
        assert_eq!(out.len(), 256);
        // h = 0: dl = 2 * (2*0+1) = 2, delta = +0.125, grid row 0.
        assert!((out[0] - 2.0 * (iq1_grid()[0] + 0.125)).abs() < 1e-4);
    }

    #[test]
    fn nvfp4_scale_decoding() {
        assert_eq!(ue4m3_to_f32_half(0), 0.0);
        assert_eq!(ue4m3_to_f32_half(0x7F), 0.0);
        // exp=7, man=0 -> (1+0)*2^0 * 0.5 = 0.5.
        assert!((ue4m3_to_f32_half(7 << 3) - 0.5).abs() < 1e-6);
        let mut block = vec![0u8; 36];
        block[0] = 7 << 3; // first sub-block scale 0.5
        block[4] = 0x05; // low nibble 5 -> kvalue 6 -> 3.0
        let mut out = Vec::new();
        dequant_nvfp4(&block, &mut out);
        assert_eq!(out.len(), 64);
        assert!((out[0] - 3.0).abs() < 1e-6);
        assert_eq!(out[63], 0.0);
    }
}
