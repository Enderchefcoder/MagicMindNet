//! From-scratch TFLite reader — flatbuffer wire format + the TFLite Model
//! schema subset needed to extract weights.
//!
//! Parses the root `Model` table (subgraphs, buffers), each subgraph's
//! tensors (shape, type, name, buffer index, quantization), and buffer
//! contents (inline `data` vectors, or the TF ≥2.13 `offset`/`size`
//! out-of-band placement for big models). INT8/UINT8/INT32 tensors with
//! quantization parameters dequantize to `scale * (q - zero_point)`,
//! per-tensor or per-channel. No flatbuffers library is linked.

use super::NamedArray;
use half::{bf16, f16};
use mmn_core::MmnError;
use std::fs;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// True when the buffer carries the `TFL3` flatbuffer identifier.
pub fn is_tflite_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 8 && &bytes[4..8] == b"TFL3"
}

/// Bounds-checked little-endian primitive reads at absolute positions.
struct Fb<'a> {
    bytes: &'a [u8],
}

impl<'a> Fb<'a> {
    fn get(&self, pos: usize, n: usize) -> Result<&'a [u8], MmnError> {
        self.bytes
            .get(pos..pos.checked_add(n).ok_or_else(|| err("tflite offset overflow"))?)
            .ok_or_else(|| err("tflite flatbuffer truncated"))
    }

    fn u16(&self, pos: usize) -> Result<u16, MmnError> {
        let b = self.get(pos, 2)?;
        Ok(u16::from_le_bytes([b[0], b[1]]))
    }

    fn u32(&self, pos: usize) -> Result<u32, MmnError> {
        let b = self.get(pos, 4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn i32(&self, pos: usize) -> Result<i32, MmnError> {
        Ok(self.u32(pos)? as i32)
    }

    fn u64(&self, pos: usize) -> Result<u64, MmnError> {
        let b = self.get(pos, 8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    /// Follow a uoffset stored at `pos` (target = pos + u32 value).
    fn indirect(&self, pos: usize) -> Result<usize, MmnError> {
        let off = self.u32(pos)? as usize;
        pos.checked_add(off).ok_or_else(|| err("tflite offset overflow"))
    }

    /// Absolute position of table field `id`'s value, or `None` if absent.
    fn field(&self, table_pos: usize, id: usize) -> Result<Option<usize>, MmnError> {
        let soffset = self.i32(table_pos)?;
        let vtable_pos = (table_pos as i64 - soffset as i64) as usize;
        let vtable_len = self.u16(vtable_pos)? as usize;
        let slot = 4 + 2 * id;
        if slot + 2 > vtable_len {
            return Ok(None);
        }
        let field_off = self.u16(vtable_pos + slot)? as usize;
        if field_off == 0 {
            return Ok(None);
        }
        Ok(Some(table_pos + field_off))
    }

    /// Vector stored at field position: returns (element start, length).
    fn vector(&self, field_pos: usize) -> Result<(usize, usize), MmnError> {
        let target = self.indirect(field_pos)?;
        let len = self.u32(target)? as usize;
        Ok((target + 4, len))
    }

    fn string(&self, field_pos: usize) -> Result<String, MmnError> {
        let (start, len) = self.vector(field_pos)?;
        String::from_utf8(self.get(start, len)?.to_vec())
            .map_err(|e| err(format!("tflite string not UTF-8: {e}")))
    }
}

/// One buffer's payload: inline vector or out-of-band offset/size region.
fn buffer_data<'a>(fb: &Fb<'a>, buffer_table: usize) -> Result<&'a [u8], MmnError> {
    if let Some(field_pos) = fb.field(buffer_table, 0)? {
        let (start, len) = fb.vector(field_pos)?;
        if len > 0 {
            return fb.get(start, len);
        }
    }
    // TF >= 2.13 big-tensor placement: offset (field 1) + size (field 2)
    // relative to the file start; offset <= 1 means "unset".
    let offset = match fb.field(buffer_table, 1)? {
        Some(pos) => fb.u64(pos)? as usize,
        None => 0,
    };
    let size = match fb.field(buffer_table, 2)? {
        Some(pos) => fb.u64(pos)? as usize,
        None => 0,
    };
    if offset > 1 && size > 0 {
        return fb.get(offset, size);
    }
    Ok(&[])
}

struct Quantization {
    scales: Vec<f32>,
    zero_points: Vec<i64>,
    quantized_dimension: usize,
}

fn read_quantization(fb: &Fb<'_>, tensor_table: usize) -> Result<Option<Quantization>, MmnError> {
    let Some(field_pos) = fb.field(tensor_table, 4)? else {
        return Ok(None);
    };
    let q_table = fb.indirect(field_pos)?;
    let scales = match fb.field(q_table, 2)? {
        Some(pos) => {
            let (start, len) = fb.vector(pos)?;
            (0..len)
                .map(|i| Ok(f32::from_bits(fb.u32(start + 4 * i)?)))
                .collect::<Result<Vec<f32>, MmnError>>()?
        }
        None => Vec::new(),
    };
    if scales.is_empty() {
        return Ok(None);
    }
    let zero_points = match fb.field(q_table, 3)? {
        Some(pos) => {
            let (start, len) = fb.vector(pos)?;
            (0..len)
                .map(|i| Ok(fb.u64(start + 8 * i)? as i64))
                .collect::<Result<Vec<i64>, MmnError>>()?
        }
        None => Vec::new(),
    };
    let quantized_dimension = match fb.field(q_table, 6)? {
        Some(pos) => fb.i32(pos)?.max(0) as usize,
        None => 0,
    };
    Ok(Some(Quantization {
        scales,
        zero_points,
        quantized_dimension,
    }))
}

/// Raw integer values -> f32, applying per-tensor/per-channel quantization.
fn apply_quantization(
    raw: Vec<f32>,
    shape: &[usize],
    quant: Option<Quantization>,
) -> Vec<f32> {
    let Some(q) = quant else {
        return raw;
    };
    if q.scales.len() == 1 {
        let scale = q.scales[0];
        let zp = q.zero_points.first().copied().unwrap_or(0) as f32;
        return raw.iter().map(|&v| scale * (v - zp)).collect();
    }
    // Per-channel: channel index = coordinate along quantized_dimension.
    let dim = q.quantized_dimension.min(shape.len().saturating_sub(1));
    let inner: usize = shape.iter().skip(dim + 1).product::<usize>().max(1);
    let channels = shape.get(dim).copied().unwrap_or(1).max(1);
    raw.iter()
        .enumerate()
        .map(|(i, &v)| {
            let channel = (i / inner) % channels;
            let scale = q.scales.get(channel).copied().unwrap_or(1.0);
            let zp = q.zero_points.get(channel).copied().unwrap_or(0) as f32;
            scale * (v - zp)
        })
        .collect()
}

/// Decode one tensor's raw buffer per its TensorType id.
fn decode_elements(type_id: u8, data: &[u8], name: &str) -> Result<Vec<f32>, MmnError> {
    let values = match type_id {
        0 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        1 => data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        2 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        3 => data.iter().map(|&b| b as f32).collect(),
        4 => data
            .chunks_exact(8)
            .map(|c| i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        6 => data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect(),
        7 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        9 => data.iter().map(|&b| b as i8 as f32).collect(),
        10 => data
            .chunks_exact(8)
            .map(|c| f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        12 => data
            .chunks_exact(8)
            .map(|c| u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32)
            .collect(),
        15 => data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        16 => data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        18 => data
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        other => {
            return Err(err(format!(
                "tflite tensor {name}: TensorType {other} not supported \
                 (float/int/uint/bool/bf16 families are)"
            )))
        }
    };
    Ok(values)
}

/// Read every weight tensor (non-empty buffer) in a TFLite byte buffer.
pub fn read_tflite_arrays_bytes(bytes: &[u8]) -> Result<Vec<NamedArray>, MmnError> {
    if !is_tflite_bytes(bytes) {
        return Err(err("not a TFLite file (missing TFL3 identifier)"));
    }
    let fb = Fb { bytes };
    let model = fb.indirect(0)?;
    let buffers: Vec<usize> = match fb.field(model, 4)? {
        Some(pos) => {
            let (start, len) = fb.vector(pos)?;
            (0..len)
                .map(|i| fb.indirect(start + 4 * i))
                .collect::<Result<_, _>>()?
        }
        None => Vec::new(),
    };
    let subgraphs: Vec<usize> = match fb.field(model, 2)? {
        Some(pos) => {
            let (start, len) = fb.vector(pos)?;
            (0..len)
                .map(|i| fb.indirect(start + 4 * i))
                .collect::<Result<_, _>>()?
        }
        None => Vec::new(),
    };
    let mut out = Vec::new();
    for subgraph in subgraphs {
        let Some(tensors_pos) = fb.field(subgraph, 0)? else {
            continue;
        };
        let (start, len) = fb.vector(tensors_pos)?;
        for i in 0..len {
            let tensor = fb.indirect(start + 4 * i)?;
            let buffer_index = match fb.field(tensor, 2)? {
                Some(pos) => fb.u32(pos)? as usize,
                None => 0,
            };
            let Some(&buffer_table) = buffers.get(buffer_index) else {
                continue;
            };
            let data = buffer_data(&fb, buffer_table)?;
            if data.is_empty() {
                continue; // activation tensor, no stored weights
            }
            let name = match fb.field(tensor, 3)? {
                Some(pos) => fb.string(pos)?,
                None => format!("tensor_{i}"),
            };
            let shape: Vec<usize> = match fb.field(tensor, 0)? {
                Some(pos) => {
                    let (s, n) = fb.vector(pos)?;
                    (0..n)
                        .map(|j| Ok(fb.i32(s + 4 * j)?.max(0) as usize))
                        .collect::<Result<_, MmnError>>()?
                }
                None => Vec::new(),
            };
            let type_id = match fb.field(tensor, 1)? {
                Some(pos) => fb.get(pos, 1)?[0],
                None => 0,
            };
            let raw = decode_elements(type_id, data, &name)?;
            let numel: usize = shape.iter().product();
            if !shape.is_empty() && raw.len() != numel {
                return Err(err(format!(
                    "tflite tensor {name}: shape {shape:?} needs {numel} values, got {}",
                    raw.len()
                )));
            }
            let quant = if matches!(type_id, 2 | 3 | 9) {
                read_quantization(&fb, tensor)?
            } else {
                None
            };
            let values = apply_quantization(raw, &shape, quant);
            out.push((name, shape, values));
        }
    }
    Ok(out)
}

/// Read every weight tensor in a `.tflite` file on disk.
pub fn read_tflite_arrays(path: &str) -> Result<Vec<NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read {path}: {e}")))?;
    read_tflite_arrays_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::path::PathBuf;

    fn fixture() -> Vec<u8> {
        let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/fixtures/simple.tflite");
        fs::read(path).expect("simple.tflite fixture missing")
    }

    #[test]
    fn fixture_identifier_detected() {
        assert!(is_tflite_bytes(&fixture()));
        assert!(!is_tflite_bytes(b"TFL3 too short"));
        assert!(!is_tflite_bytes(b"\x00\x00\x00\x00NOPE"));
    }

    #[test]
    fn fixture_weights_extract() {
        let arrays = read_tflite_arrays_bytes(&fixture()).unwrap();
        assert!(!arrays.is_empty(), "no weight tensors found");
        // The fixture's dense kernel is 3x2 filled with 0.1*(i+1) (stored
        // transposed as [2, 3] by the converter).
        let expected: Vec<f32> = (1..=6).map(|i| 0.1 * i as f32).collect();
        let mut sorted_expected = expected.clone();
        sorted_expected.sort_by(f32::total_cmp);
        let found = arrays.iter().any(|(_, shape, values)| {
            let numel: usize = shape.iter().product();
            if numel != 6 {
                return false;
            }
            let mut sorted = values.clone();
            sorted.sort_by(f32::total_cmp);
            sorted
                .iter()
                .zip(&sorted_expected)
                .all(|(a, b)| (a - b).abs() < 1e-6)
        });
        assert!(found, "dense kernel values not found: {arrays:?}");
        // The bias [0.5, -0.5] must be present exactly.
        let bias = arrays
            .iter()
            .find(|(_, shape, _)| shape == &vec![2])
            .expect("bias tensor missing");
        assert_eq!(bias.2, vec![0.5, -0.5]);
    }

    #[test]
    fn junk_rejected() {
        assert!(read_tflite_arrays_bytes(b"not a tflite file at all")
            .err()
            .unwrap()
            .message()
            .contains("TFL3"));
    }

    #[test]
    fn per_tensor_quantization_applies() {
        let raw = vec![10.0, 20.0, 30.0];
        let out = apply_quantization(
            raw,
            &[3],
            Some(Quantization {
                scales: vec![0.5],
                zero_points: vec![10],
                quantized_dimension: 0,
            }),
        );
        assert_eq!(out, vec![0.0, 5.0, 10.0]);
    }

    #[test]
    fn per_channel_quantization_applies() {
        // Shape [2, 2], quantized along dim 0: channel 0 scale 1, ch 1 scale 10.
        let raw = vec![1.0, 2.0, 3.0, 4.0];
        let out = apply_quantization(
            raw,
            &[2, 2],
            Some(Quantization {
                scales: vec![1.0, 10.0],
                zero_points: vec![0, 0],
                quantized_dimension: 0,
            }),
        );
        assert_eq!(out, vec![1.0, 2.0, 30.0, 40.0]);
    }
}
