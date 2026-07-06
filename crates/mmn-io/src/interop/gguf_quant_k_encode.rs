//! K-quant encoders (Q4_K / Q5_K / Q6_K), ported from the GGML reference
//! quantization: per-sub-block scale/min search (`make_qx_quants` /
//! `make_qkx2_quants`) under 6-bit super-block scales.
//!
//! Notably the reference llama.cpp Python package cannot encode k-quants at
//! all — this is a from-scratch port of the C reference algorithms.

use half::f16;
use mmn_core::MmnError;

pub const QK_K: usize = 256;
const GROUP_MAX_EPS: f32 = 1e-15;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// GGML's `nearest_int`: round half to even (the float-add trick).
fn nearest_int(x: f32) -> i32 {
    x.round_ties_even() as i32
}

/// Signed-range sub-block quantization with iscale refinement (rmse_type=1).
fn make_qx_quants(nmax: i32, x: &[f32], levels: &mut [i8]) -> f32 {
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        levels.fill(0);
        return 0.0;
    }
    let mut iscale = -(nmax as f32) / max;
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for (i, &v) in x.iter().enumerate() {
        let l = nearest_int(iscale * v).clamp(-nmax, nmax - 1);
        levels[i] = (l + nmax) as i8;
        let w = v * v;
        sumlx += w * v * l as f32;
        suml2 += w * (l * l) as f32;
    }
    let mut scale = if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;
    for is in -9i32..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        let mut sumlx = 0.0f32;
        let mut suml2 = 0.0f32;
        for &v in x {
            let l = nearest_int(iscale * v).clamp(-nmax, nmax - 1);
            let w = v * v;
            sumlx += w * v * l as f32;
            suml2 += w * (l * l) as f32;
        }
        if suml2 > 0.0 && sumlx * sumlx > best * suml2 {
            for (i, &v) in x.iter().enumerate() {
                let l = nearest_int(iscale * v).clamp(-nmax, nmax - 1);
                levels[i] = (l + nmax) as i8;
            }
            scale = sumlx / suml2;
            best = scale * sumlx;
        }
    }
    scale
}

/// Unsigned-range quantization with joint scale+min least-squares search.
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    levels: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32) {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = weights[0] * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        levels.fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        levels[i] = l as u8;
        let diff = scale * l as f32 + min - x[i];
        best_mad += weights[i] * diff * diff;
    }
    let mut laux = vec![0u8; n];
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = l as u8;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * (l as f32) * (l as f32);
            sum_xl += w * l as f32 * x[i];
        }
        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let diff = this_scale * laux[i] as f32 + this_min - x[i];
                mad += weights[i] * diff * diff;
            }
            if mad < best_mad {
                levels.copy_from_slice(&laux);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Pack 6-bit (scale, min) pairs into the 12-byte Q4_K/Q5_K scales field.
fn pack_scales_k4(ls: &[u8; 8], lm: &[u8; 8]) -> [u8; 12] {
    let mut scales = [0u8; 12];
    for j in 0..8 {
        if j < 4 {
            scales[j] = ls[j];
            scales[j + 4] = lm[j];
        } else {
            scales[j + 4] = (ls[j] & 0x0F) | ((lm[j] & 0x0F) << 4);
            scales[j - 4] |= (ls[j] >> 4) << 6;
            scales[j] |= (lm[j] >> 4) << 6;
        }
    }
    scales
}

/// Shared Q4_K/Q5_K super-block preparation: per-32 sub-block scale/min
/// search + 6-bit packing. Returns (d, dmin, packed scales, levels).
fn qk45_prepare(block: &[f32], nmax: i32, rmin: f32, rdelta: f32, nstep: i32) -> (f32, f32, [u8; 12], [u8; QK_K]) {
    let mut levels = [0u8; QK_K];
    let mut scales = [0.0f32; 8];
    let mut mins = [0.0f32; 8];
    let mut weights = [0.0f32; 32];
    for j in 0..8 {
        let sub = &block[32 * j..32 * j + 32];
        let sum_x2: f32 = sub.iter().map(|v| v * v).sum();
        let av_x = (sum_x2 / 32.0).sqrt();
        for (w, &v) in weights.iter_mut().zip(sub) {
            *w = av_x + v.abs();
        }
        let (scale, min) = make_qkx2_quants(
            nmax,
            sub,
            &weights,
            &mut levels[32 * j..32 * j + 32],
            rmin,
            rdelta,
            nstep,
        );
        scales[j] = scale;
        mins[j] = min;
    }
    let max_scale = scales.iter().fold(0.0f32, |m, &v| m.max(v));
    let max_min = mins.iter().fold(0.0f32, |m, &v| m.max(v));
    let inv_scale = if max_scale > 0.0 { 63.0 / max_scale } else { 0.0 };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };
    let mut ls = [0u8; 8];
    let mut lm = [0u8; 8];
    for j in 0..8 {
        ls[j] = nearest_int(inv_scale * scales[j]).clamp(0, 63) as u8;
        lm[j] = nearest_int(inv_min * mins[j]).clamp(0, 63) as u8;
    }
    let packed = pack_scales_k4(&ls, &lm);
    let d = f16::from_f32(max_scale / 63.0).to_f32();
    let dmin = f16::from_f32(max_min / 63.0).to_f32();
    // Requantize levels against the rounded 6-bit scales.
    for j in 0..8 {
        let (sc, m) = super::gguf_quant::scale_min_k4_pub(j, &packed);
        let dj = d * sc as f32;
        if dj == 0.0 {
            continue;
        }
        let dm = dmin * m as f32;
        for ii in 0..32 {
            let l = nearest_int((block[32 * j + ii] + dm) / dj).clamp(0, nmax);
            levels[32 * j + ii] = l as u8;
        }
    }
    (d, dmin, packed, levels)
}

/// Signed sub-block quantization with iterative RMSE refinement (Q3_K).
fn make_q3_quants(nmax: i32, x: &[f32], levels: &mut [i8]) -> f32 {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &v in x {
        if v.abs() > amax {
            amax = v.abs();
            max = v;
        }
    }
    if amax < GROUP_MAX_EPS {
        levels.fill(0);
        return 0.0;
    }
    let iscale = -(nmax as f32) / max;
    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for (i, &v) in x.iter().enumerate() {
        let l = nearest_int(iscale * v).clamp(-nmax, nmax - 1);
        levels[i] = l as i8;
        let w = v * v;
        sumlx += w * v * l as f32;
        suml2 += w * (l * l) as f32;
    }
    for _ in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let v = x[i];
            let w = v * v;
            let slx = sumlx - w * v * levels[i] as f32;
            if slx > 0.0 {
                let mut sl2 = suml2 - w * (levels[i] as i32 * levels[i] as i32) as f32;
                let new_l = nearest_int(v * sl2 / slx).clamp(-nmax, nmax - 1);
                if new_l as i8 != levels[i] {
                    let slx2 = slx + w * v * new_l as f32;
                    sl2 += w * (new_l * new_l) as f32;
                    if sl2 > 0.0 && slx2 * slx2 * suml2 > sumlx * sumlx * sl2 {
                        levels[i] = new_l as i8;
                        sumlx = slx2;
                        suml2 = sl2;
                        n_changed += 1;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }
    for l in levels.iter_mut() {
        *l += nmax as i8;
    }
    if suml2 > 0.0 {
        sumlx / suml2
    } else {
        0.0
    }
}

/// Pack 2-bit levels in ggml's 128-element interleave (shared Q2_K/Q3_K).
fn pack_2bit(levels: &[u8; QK_K], out: &mut Vec<u8>) {
    for j in (0..QK_K).step_by(128) {
        for l in 0..32 {
            out.push(
                levels[j + l]
                    | (levels[j + l + 32] << 2)
                    | (levels[j + l + 64] << 4)
                    | (levels[j + l + 96] << 6),
            );
        }
    }
}

/// Quantize `f32` values into Q2_K super-blocks (84 bytes / 256 values).
pub fn quantize_q2_k(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    if !values.len().is_multiple_of(QK_K) {
        return Err(err(format!(
            "Q2_K quantization needs a multiple of {QK_K} values, got {}",
            values.len()
        )));
    }
    const Q4SCALE: f32 = 15.0;
    let mut out = Vec::with_capacity(values.len() / QK_K * 84);
    for block in values.chunks_exact(QK_K) {
        let mut levels = [0u8; QK_K];
        let mut scales = [0.0f32; 16];
        let mut mins = [0.0f32; 16];
        let mut weights = [0.0f32; 16];
        for (j, sub) in block.chunks_exact(16).enumerate() {
            for (w, &v) in weights.iter_mut().zip(sub) {
                *w = v.abs();
            }
            let (scale, min) = make_qkx2_quants_mad(
                3,
                sub,
                &weights,
                &mut levels[16 * j..16 * j + 16],
                -0.5,
                0.1,
                15,
            );
            scales[j] = scale;
            mins[j] = min;
        }
        let max_scale = scales.iter().fold(0.0f32, |m, &v| m.max(v));
        let max_min = mins.iter().fold(0.0f32, |m, &v| m.max(v));
        let mut scale_bytes = [0u8; 16];
        let d = if max_scale > 0.0 {
            let iscale = Q4SCALE / max_scale;
            for (j, &s) in scales.iter().enumerate() {
                scale_bytes[j] = nearest_int(iscale * s) as u8;
            }
            f16::from_f32(max_scale / Q4SCALE).to_f32()
        } else {
            0.0
        };
        let dmin = if max_min > 0.0 {
            let iscale = Q4SCALE / max_min;
            for (j, &m) in mins.iter().enumerate() {
                scale_bytes[j] |= (nearest_int(iscale * m) as u8) << 4;
            }
            f16::from_f32(max_min / Q4SCALE).to_f32()
        } else {
            0.0
        };
        for j in 0..16 {
            let dj = d * (scale_bytes[j] & 0x0F) as f32;
            if dj == 0.0 {
                continue;
            }
            let dm = dmin * (scale_bytes[j] >> 4) as f32;
            for ii in 0..16 {
                let l = nearest_int((block[16 * j + ii] + dm) / dj).clamp(0, 3);
                levels[16 * j + ii] = l as u8;
            }
        }
        out.extend_from_slice(&scale_bytes);
        pack_2bit(&levels, &mut out);
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
    }
    Ok(out)
}

/// Quantize `f32` values into Q3_K super-blocks (110 bytes / 256 values).
pub fn quantize_q3_k(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    if !values.len().is_multiple_of(QK_K) {
        return Err(err(format!(
            "Q3_K quantization needs a multiple of {QK_K} values, got {}",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(values.len() / QK_K * 110);
    for block in values.chunks_exact(QK_K) {
        let mut levels = [0i8; QK_K];
        let mut scales = [0.0f32; 16];
        let mut max_scale = 0.0f32;
        let mut amax_scale = 0.0f32;
        for (j, sub) in block.chunks_exact(16).enumerate() {
            scales[j] = make_q3_quants(4, sub, &mut levels[16 * j..16 * j + 16]);
            if scales[j].abs() > amax_scale {
                amax_scale = scales[j].abs();
                max_scale = scales[j];
            }
        }
        let mut packed_scales = [0u8; 12];
        let d = if max_scale != 0.0 {
            let iscale = -32.0 / max_scale;
            for (j, &s) in scales.iter().enumerate() {
                let l = (nearest_int(iscale * s).clamp(-32, 31) + 32) as u8;
                if j < 8 {
                    packed_scales[j] |= l & 0x0F;
                } else {
                    packed_scales[j - 8] |= (l & 0x0F) << 4;
                }
                let high = l >> 4;
                packed_scales[8 + j % 4] |= high << (2 * (j / 4));
            }
            f16::from_f32(1.0 / iscale).to_f32()
        } else {
            0.0
        };
        for j in 0..16 {
            let sc_low = if j < 8 {
                packed_scales[j] & 0x0F
            } else {
                packed_scales[j - 8] >> 4
            };
            let sc = (sc_low | (((packed_scales[8 + j % 4] >> (2 * (j / 4))) & 3) << 4)) as i32 - 32;
            let dj = d * sc as f32;
            if dj == 0.0 {
                continue;
            }
            for ii in 0..16 {
                let l = nearest_int(block[16 * j + ii] / dj).clamp(-4, 3);
                levels[16 * j + ii] = (l + 4) as i8;
            }
        }
        let mut hmask = [0u8; QK_K / 8];
        let mut two_bit = [0u8; QK_K];
        let mut m = 0usize;
        let mut hm: u8 = 1;
        for (ii, &l) in levels.iter().enumerate() {
            let mut l = l as u8;
            if l > 3 {
                hmask[m] |= hm;
                l -= 4;
            }
            two_bit[ii] = l;
            m += 1;
            if m == QK_K / 8 {
                m = 0;
                hm <<= 1;
            }
        }
        out.extend_from_slice(&hmask);
        pack_2bit(&two_bit, &mut out);
        out.extend_from_slice(&packed_scales);
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    }
    Ok(out)
}

/// Quantize `f32` values into Q8_K super-blocks (292 bytes / 256 values).
pub fn quantize_q8_k(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    if !values.len().is_multiple_of(QK_K) {
        return Err(err(format!(
            "Q8_K quantization needs a multiple of {QK_K} values, got {}",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(values.len() / QK_K * 292);
    for block in values.chunks_exact(QK_K) {
        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &v in block {
            if v.abs() > amax {
                amax = v.abs();
                max = v;
            }
        }
        if amax == 0.0 {
            out.extend(std::iter::repeat_n(0u8, 292));
            continue;
        }
        let iscale = -128.0 / max;
        let mut qs = [0i8; QK_K];
        for (i, &v) in block.iter().enumerate() {
            qs[i] = nearest_int(iscale * v).min(127) as i8;
        }
        out.extend_from_slice(&(1.0 / iscale).to_le_bytes());
        out.extend_from_slice(&qs.map(|b| b as u8));
        for sub in qs.chunks_exact(16) {
            let sum: i16 = sub.iter().map(|&b| b as i16).sum();
            out.extend_from_slice(&sum.to_le_bytes());
        }
    }
    Ok(out)
}

/// `make_qkx2_quants` with mean-absolute-deviation error (Q2_K variant).
#[allow(clippy::too_many_arguments)]
fn make_qkx2_quants_mad(
    nmax: i32,
    x: &[f32],
    weights: &[f32],
    levels: &mut [u8],
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32) {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = weights[0] * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        sum_w += weights[i];
        sum_x += weights[i] * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    if max == min {
        levels.fill(0);
        return (0.0, -min);
    }
    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut best_mad = 0.0f32;
    for i in 0..n {
        let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
        levels[i] = l as u8;
        let diff = (scale * l as f32 + min - x[i]).abs();
        best_mad += weights[i] * diff;
    }
    let mut laux = vec![0u8; n];
    for is in 0..=nstep {
        iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
        let mut sum_l = 0.0f32;
        let mut sum_l2 = 0.0f32;
        let mut sum_xl = 0.0f32;
        for i in 0..n {
            let l = nearest_int(iscale * (x[i] - min)).clamp(0, nmax);
            laux[i] = l as u8;
            let w = weights[i];
            sum_l += w * l as f32;
            sum_l2 += w * (l as f32) * (l as f32);
            sum_xl += w * l as f32 * x[i];
        }
        let det = sum_w * sum_l2 - sum_l * sum_l;
        if det > 0.0 {
            let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / det;
            let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / det;
            if this_min > 0.0 {
                this_min = 0.0;
                this_scale = sum_xl / sum_l2;
            }
            let mut mad = 0.0f32;
            for i in 0..n {
                let diff = (this_scale * laux[i] as f32 + this_min - x[i]).abs();
                mad += weights[i] * diff;
            }
            if mad < best_mad {
                levels.copy_from_slice(&laux);
                best_mad = mad;
                scale = this_scale;
                min = this_min;
            }
        }
    }
    (scale, -min)
}

/// Quantize `f32` values into Q4_K super-blocks (144 bytes / 256 values).
pub fn quantize_q4_k(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    if !values.len().is_multiple_of(QK_K) {
        return Err(err(format!(
            "Q4_K quantization needs a multiple of {QK_K} values, got {}",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(values.len() / QK_K * 144);
    for block in values.chunks_exact(QK_K) {
        let (d, dmin, scales, levels) = qk45_prepare(block, 15, -1.0, 0.1, 20);
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
        out.extend_from_slice(&scales);
        for j in (0..QK_K).step_by(64) {
            for l in 0..32 {
                out.push(levels[j + l] | (levels[j + l + 32] << 4));
            }
        }
    }
    Ok(out)
}

/// Quantize `f32` values into Q5_K super-blocks (176 bytes / 256 values).
pub fn quantize_q5_k(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    if !values.len().is_multiple_of(QK_K) {
        return Err(err(format!(
            "Q5_K quantization needs a multiple of {QK_K} values, got {}",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(values.len() / QK_K * 176);
    for block in values.chunks_exact(QK_K) {
        let (d, dmin, scales, levels) = qk45_prepare(block, 31, -0.5, 0.1, 15);
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
        out.extend_from_slice(&f16::from_f32(dmin).to_le_bytes());
        out.extend_from_slice(&scales);
        let mut qh = [0u8; QK_K / 8];
        let mut ql = [0u8; QK_K / 2];
        let mut m1: u8 = 1;
        let mut m2: u8 = 2;
        let mut ql_pos = 0usize;
        for n in (0..QK_K).step_by(64) {
            for j in 0..32 {
                let mut l1 = levels[n + j];
                if l1 > 15 {
                    l1 -= 16;
                    qh[j] |= m1;
                }
                let mut l2 = levels[n + j + 32];
                if l2 > 15 {
                    l2 -= 16;
                    qh[j] |= m2;
                }
                ql[ql_pos + j] = l1 | (l2 << 4);
            }
            m1 <<= 2;
            m2 <<= 2;
            ql_pos += 32;
        }
        out.extend_from_slice(&qh);
        out.extend_from_slice(&ql);
    }
    Ok(out)
}

/// Quantize `f32` values into Q6_K super-blocks (210 bytes / 256 values).
pub fn quantize_q6_k(values: &[f32]) -> Result<Vec<u8>, MmnError> {
    if !values.len().is_multiple_of(QK_K) {
        return Err(err(format!(
            "Q6_K quantization needs a multiple of {QK_K} values, got {}",
            values.len()
        )));
    }
    let mut out = Vec::with_capacity(values.len() / QK_K * 210);
    for block in values.chunks_exact(QK_K) {
        let mut levels = [0i8; QK_K];
        let mut scales = [0.0f32; 16];
        let mut max_scale = 0.0f32;
        let mut max_abs_scale = 0.0f32;
        for (ib, sub) in block.chunks_exact(16).enumerate() {
            let scale = make_qx_quants(32, sub, &mut levels[16 * ib..16 * ib + 16]);
            scales[ib] = scale;
            if scale.abs() > max_abs_scale {
                max_abs_scale = scale.abs();
                max_scale = scale;
            }
        }
        if max_abs_scale < GROUP_MAX_EPS {
            out.extend(std::iter::repeat_n(0u8, 210));
            continue;
        }
        let iscale = -128.0 / max_scale;
        let d = f16::from_f32(1.0 / iscale).to_f32();
        let mut scale_bytes = [0i8; 16];
        for (ib, &s) in scales.iter().enumerate() {
            scale_bytes[ib] = nearest_int(iscale * s).min(127) as i8;
        }
        for j in 0..QK_K {
            let dj = d * scale_bytes[j / 16] as f32;
            if dj == 0.0 {
                continue;
            }
            let l = nearest_int(block[j] / dj).clamp(-32, 31);
            levels[j] = (l + 32) as i8;
        }
        // Pack: ql lower 4 bits, qh upper 2 bits in the dequant layout.
        let mut ql = [0u8; QK_K / 2];
        let mut qh = [0u8; QK_K / 4];
        let mut ql_pos = 0usize;
        let mut qh_pos = 0usize;
        for j in (0..QK_K).step_by(128) {
            for l in 0..32 {
                let q1 = levels[j + l] as u8 & 0x0F;
                let q2 = levels[j + l + 32] as u8 & 0x0F;
                let q3 = levels[j + l + 64] as u8 & 0x0F;
                let q4 = levels[j + l + 96] as u8 & 0x0F;
                ql[ql_pos + l] = q1 | (q3 << 4);
                ql[ql_pos + l + 32] = q2 | (q4 << 4);
                qh[qh_pos + l] = ((levels[j + l] as u8) >> 4)
                    | (((levels[j + l + 32] as u8) >> 4) << 2)
                    | (((levels[j + l + 64] as u8) >> 4) << 4)
                    | (((levels[j + l + 96] as u8) >> 4) << 6);
            }
            ql_pos += 64;
            qh_pos += 32;
        }
        out.extend_from_slice(&ql);
        out.extend_from_slice(&qh);
        out.extend_from_slice(&scale_bytes.map(|b| b as u8));
        out.extend_from_slice(&f16::from_f32(d).to_le_bytes());
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::gguf_quant::{dequantize, GgmlType};
    use super::*;

    fn pseudo_random(n: usize, seed: u64) -> Vec<f32> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((state >> 33) as f32 / (1u64 << 31) as f32 - 0.5) * 2.0
            })
            .collect()
    }

    fn rel_rmse(a: &[f32], b: &[f32]) -> f32 {
        let range = a.iter().fold(0.0f32, |m, &v| m.max(v.abs())).max(1e-9);
        let mse: f32 = a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum::<f32>()
            / a.len() as f32;
        mse.sqrt() / range
    }

    #[test]
    fn q2_k_roundtrip_error_reasonable() {
        let values = pseudo_random(512, 13);
        let packed = quantize_q2_k(&values).unwrap();
        assert_eq!(packed.len(), 2 * 84);
        let back = dequantize(GgmlType::Q2K, &packed, 512).unwrap();
        let e = rel_rmse(&values, &back);
        // ~2.6 bits/weight on uniform random data is inherently coarse.
        assert!(e < 0.16, "Q2_K rel RMSE {e}");
    }

    #[test]
    fn q3_k_roundtrip_error_reasonable() {
        let values = pseudo_random(512, 17);
        let packed = quantize_q3_k(&values).unwrap();
        assert_eq!(packed.len(), 2 * 110);
        let back = dequantize(GgmlType::Q3K, &packed, 512).unwrap();
        let e = rel_rmse(&values, &back);
        assert!(e < 0.08, "Q3_K rel RMSE {e}");
    }

    #[test]
    fn q8_k_roundtrip_error_tiny() {
        let values = pseudo_random(512, 19);
        let packed = quantize_q8_k(&values).unwrap();
        assert_eq!(packed.len(), 2 * 292);
        let back = dequantize(GgmlType::Q8K, &packed, 512).unwrap();
        let e = rel_rmse(&values, &back);
        assert!(e < 0.006, "Q8_K rel RMSE {e}");
    }

    #[test]
    fn full_kquant_quality_ladder() {
        // Reconstruction error must improve monotonically with bit width.
        let values = pseudo_random(1024, 23);
        let errs: Vec<f32> = [
            (quantize_q2_k(&values).unwrap(), GgmlType::Q2K),
            (quantize_q3_k(&values).unwrap(), GgmlType::Q3K),
            (quantize_q4_k(&values).unwrap(), GgmlType::Q4K),
            (quantize_q5_k(&values).unwrap(), GgmlType::Q5K),
            (quantize_q6_k(&values).unwrap(), GgmlType::Q6K),
            (quantize_q8_k(&values).unwrap(), GgmlType::Q8K),
        ]
        .into_iter()
        .map(|(packed, ty)| {
            let back = dequantize(ty, &packed, 1024).unwrap();
            rel_rmse(&values, &back)
        })
        .collect();
        for pair in errs.windows(2) {
            assert!(pair[1] < pair[0], "quality ladder violated: {errs:?}");
        }
    }

    #[test]
    fn q2_q3_q8_zero_blocks() {
        let zeros = vec![0.0f32; 256];
        for (packed, ty) in [
            (quantize_q2_k(&zeros).unwrap(), GgmlType::Q2K),
            (quantize_q3_k(&zeros).unwrap(), GgmlType::Q3K),
            (quantize_q8_k(&zeros).unwrap(), GgmlType::Q8K),
        ] {
            let back = dequantize(ty, &packed, 256).unwrap();
            assert!(back.iter().all(|&v| v.abs() < 1e-6), "{ty:?}");
        }
    }

    #[test]
    fn q6_k_roundtrip_error_small() {
        let values = pseudo_random(512, 3);
        let packed = quantize_q6_k(&values).unwrap();
        assert_eq!(packed.len(), 2 * 210);
        let back = dequantize(GgmlType::Q6K, &packed, 512).unwrap();
        let e = rel_rmse(&values, &back);
        assert!(e < 0.02, "Q6_K rel RMSE {e}");
    }

    #[test]
    fn q4_k_roundtrip_error_small() {
        let values = pseudo_random(512, 5);
        let packed = quantize_q4_k(&values).unwrap();
        assert_eq!(packed.len(), 2 * 144);
        let back = dequantize(GgmlType::Q4K, &packed, 512).unwrap();
        let e = rel_rmse(&values, &back);
        assert!(e < 0.05, "Q4_K rel RMSE {e}");
    }

    #[test]
    fn q5_k_roundtrip_error_small() {
        let values = pseudo_random(512, 7);
        let packed = quantize_q5_k(&values).unwrap();
        assert_eq!(packed.len(), 2 * 176);
        let back = dequantize(GgmlType::Q5K, &packed, 512).unwrap();
        let e = rel_rmse(&values, &back);
        assert!(e < 0.03, "Q5_K rel RMSE {e}");
    }

    #[test]
    fn quality_ordering_q6_better_than_q4() {
        let values = pseudo_random(1024, 11);
        let q6 = dequantize(GgmlType::Q6K, &quantize_q6_k(&values).unwrap(), 1024).unwrap();
        let q4 = dequantize(GgmlType::Q4K, &quantize_q4_k(&values).unwrap(), 1024).unwrap();
        assert!(rel_rmse(&values, &q6) < rel_rmse(&values, &q4));
    }

    #[test]
    fn constant_and_zero_blocks() {
        let zeros = vec![0.0f32; 256];
        for (quant, ty) in [
            (quantize_q4_k(&zeros).unwrap(), GgmlType::Q4K),
            (quantize_q5_k(&zeros).unwrap(), GgmlType::Q5K),
            (quantize_q6_k(&zeros).unwrap(), GgmlType::Q6K),
        ] {
            let back = dequantize(ty, &quant, 256).unwrap();
            assert!(back.iter().all(|&v| v.abs() < 1e-6), "{ty:?}");
        }
        let ones = vec![1.0f32; 256];
        let back = dequantize(GgmlType::Q4K, &quantize_q4_k(&ones).unwrap(), 256).unwrap();
        assert!(back.iter().all(|&v| (v - 1.0).abs() < 0.01));
    }

    #[test]
    fn misaligned_length_errors() {
        assert!(quantize_q4_k(&[0.0; 100]).is_err());
        assert!(quantize_q5_k(&[0.0; 100]).is_err());
        assert!(quantize_q6_k(&[0.0; 100]).is_err());
    }
}
