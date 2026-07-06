//! From-scratch GGUF container reader/writer (versions 2 and 3).
//!
//! Implements the GGUF binary layout directly: header, typed metadata
//! key/values, tensor infos, and the aligned tensor-data section. No
//! llama.cpp or ggml code is used anywhere.

use super::gguf_quant::{
    dequantize, encode_f16, quantize_mxfp4, quantize_q4_0, quantize_q4_1, quantize_q5_0,
    quantize_q5_1, quantize_q8_0, quantize_tq1_0, quantize_tq2_0, GgmlType,
};
use super::gguf_quant_iq::{quantize_iq4_nl, quantize_iq4_xs};
use super::gguf_quant_k_encode::{
    quantize_q2_k, quantize_q3_k, quantize_q4_k, quantize_q5_k, quantize_q6_k, quantize_q8_k,
};
use mmn_core::MmnError;
use std::collections::HashMap;

pub const GGUF_MAGIC: &[u8; 4] = b"GGUF";
pub const GGUF_VERSION: u32 = 3;
pub const DEFAULT_ALIGNMENT: usize = 32;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// A typed GGUF metadata value.
#[derive(Clone, Debug, PartialEq)]
pub enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
    U64(u64),
    I64(i64),
    F64(f64),
}

impl GgufValue {
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_f64(&self) -> Option<f64> {
        match self {
            GgufValue::F32(v) => Some(*v as f64),
            GgufValue::F64(v) => Some(*v),
            other => other.as_u64().map(|v| v as f64),
        }
    }

    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            GgufValue::Bool(b) => Some(*b),
            _ => None,
        }
    }

    fn type_id(&self) -> u32 {
        match self {
            GgufValue::U8(_) => 0,
            GgufValue::I8(_) => 1,
            GgufValue::U16(_) => 2,
            GgufValue::I16(_) => 3,
            GgufValue::U32(_) => 4,
            GgufValue::I32(_) => 5,
            GgufValue::F32(_) => 6,
            GgufValue::Bool(_) => 7,
            GgufValue::String(_) => 8,
            GgufValue::Array(_) => 9,
            GgufValue::U64(_) => 10,
            GgufValue::I64(_) => 11,
            GgufValue::F64(_) => 12,
        }
    }
}

/// Tensor descriptor: dims are stored GGML-style (`ne[0]` fastest-varying).
#[derive(Clone, Debug)]
pub struct GgufTensorInfo {
    pub name: String,
    pub dims: Vec<u64>,
    pub ggml_type: GgmlType,
    pub offset: u64,
}

impl GgufTensorInfo {
    pub fn numel(&self) -> usize {
        self.dims.iter().product::<u64>() as usize
    }

    /// Row-major (C order) shape: GGML dims reversed.
    pub fn row_major_shape(&self) -> Vec<usize> {
        self.dims.iter().rev().map(|&d| d as usize).collect()
    }
}

/// Parsed GGUF file: metadata, tensor infos, and the raw data section.
pub struct GgufFile {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub alignment: usize,
    data: Vec<u8>,
}

impl GgufFile {
    /// Dequantize one tensor into `(row_major_shape, f32 values)`.
    pub fn tensor_f32(&self, info: &GgufTensorInfo) -> Result<(Vec<usize>, Vec<f32>), MmnError> {
        let start = info.offset as usize;
        let numel = info.numel();
        let len = info.ggml_type.data_len(numel)?;
        let slice = self
            .data
            .get(start..start + len)
            .ok_or_else(|| err(format!("GGUF tensor {} data out of bounds", info.name)))?;
        let values = dequantize(info.ggml_type, slice, numel)?;
        Ok((info.row_major_shape(), values))
    }

    pub fn find_tensor(&self, name: &str) -> Option<&GgufTensorInfo> {
        self.tensors.iter().find(|t| t.name == name)
    }
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
            .ok_or_else(|| err("GGUF file truncated"))?;
        let out = &self.bytes[self.pos..end];
        self.pos = end;
        Ok(out)
    }

    fn u32(&mut self) -> Result<u32, MmnError> {
        let b = self.take(4)?;
        Ok(u32::from_le_bytes([b[0], b[1], b[2], b[3]]))
    }

    fn u64(&mut self) -> Result<u64, MmnError> {
        let b = self.take(8)?;
        Ok(u64::from_le_bytes([
            b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7],
        ]))
    }

    fn string(&mut self) -> Result<String, MmnError> {
        let len = self.u64()? as usize;
        if len > self.bytes.len() {
            return Err(err("GGUF file truncated (string length exceeds buffer)"));
        }
        let bytes = self.take(len)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| err(format!("GGUF string not UTF-8: {e}")))
    }

    fn value(&mut self, type_id: u32, depth: usize) -> Result<GgufValue, MmnError> {
        if depth > 4 {
            return Err(err("GGUF metadata array nesting too deep"));
        }
        Ok(match type_id {
            0 => GgufValue::U8(self.take(1)?[0]),
            1 => GgufValue::I8(self.take(1)?[0] as i8),
            2 => {
                let b = self.take(2)?;
                GgufValue::U16(u16::from_le_bytes([b[0], b[1]]))
            }
            3 => {
                let b = self.take(2)?;
                GgufValue::I16(i16::from_le_bytes([b[0], b[1]]))
            }
            4 => GgufValue::U32(self.u32()?),
            5 => GgufValue::I32(self.u32()? as i32),
            6 => GgufValue::F32(f32::from_bits(self.u32()?)),
            7 => GgufValue::Bool(self.take(1)?[0] != 0),
            8 => GgufValue::String(self.string()?),
            9 => {
                let elem_type = self.u32()?;
                let count = self.u64()? as usize;
                if count > self.bytes.len() {
                    return Err(err("GGUF file truncated (array count exceeds buffer)"));
                }
                let mut items = Vec::with_capacity(count.min(1 << 20));
                for _ in 0..count {
                    items.push(self.value(elem_type, depth + 1)?);
                }
                GgufValue::Array(items)
            }
            10 => GgufValue::U64(self.u64()?),
            11 => GgufValue::I64(self.u64()? as i64),
            12 => GgufValue::F64(f64::from_bits(self.u64()?)),
            other => return Err(err(format!("GGUF metadata value type {other} unknown"))),
        })
    }
}

/// Header portion of a GGUF file: everything before the tensor data section.
pub struct GgufHeader {
    pub version: u32,
    pub metadata: HashMap<String, GgufValue>,
    pub tensors: Vec<GgufTensorInfo>,
    pub alignment: usize,
    pub data_start: usize,
}

/// Parse just the GGUF header (magic, metadata KV, tensor infos).
pub fn parse_gguf_header(bytes: &[u8]) -> Result<GgufHeader, MmnError> {
    if bytes.len() < 4 || &bytes[..4] != GGUF_MAGIC {
        return Err(err("not a GGUF file (missing GGUF magic)"));
    }
    let mut r = Reader { bytes, pos: 4 };
    let version = r.u32()?;
    if !(2..=3).contains(&version) {
        return Err(err(format!(
            "GGUF version {version} not supported (versions 2 and 3 are)"
        )));
    }
    let tensor_count = r.u64()? as usize;
    let kv_count = r.u64()? as usize;
    if tensor_count > 1 << 24 || kv_count > 1 << 24 {
        return Err(err("GGUF header counts unreasonably large"));
    }
    let mut metadata = HashMap::with_capacity(kv_count);
    for _ in 0..kv_count {
        let key = r.string()?;
        let type_id = r.u32()?;
        let value = r.value(type_id, 0)?;
        metadata.insert(key, value);
    }
    let alignment = metadata
        .get("general.alignment")
        .and_then(|v| v.as_u64())
        .map(|v| v as usize)
        .unwrap_or(DEFAULT_ALIGNMENT)
        .max(1);
    let mut tensors = Vec::with_capacity(tensor_count);
    for _ in 0..tensor_count {
        let name = r.string()?;
        let n_dims = r.u32()? as usize;
        if n_dims > 4 {
            return Err(err(format!("GGUF tensor {name} has {n_dims} dims (max 4)")));
        }
        let mut dims = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            dims.push(r.u64()?);
        }
        let type_id = r.u32()?;
        let ggml_type = GgmlType::from_id(type_id)?;
        let offset = r.u64()?;
        tensors.push(GgufTensorInfo {
            name,
            dims,
            ggml_type,
            offset,
        });
    }
    let data_start = r.pos.div_ceil(alignment) * alignment;
    Ok(GgufHeader {
        version,
        metadata,
        tensors,
        alignment,
        data_start,
    })
}

/// Read only the GGUF header from disk, loading the file incrementally so
/// multi-gigabyte models are not pulled into memory for inspection.
pub fn read_gguf_header_file(path: &str) -> Result<GgufHeader, MmnError> {
    use std::io::Read;
    let mut file = std::fs::File::open(path)
        .map_err(|e| err(format!("cannot read GGUF {path}: {e}")))?;
    let file_len = file
        .metadata()
        .map(|m| m.len() as usize)
        .unwrap_or(usize::MAX);
    let mut buf: Vec<u8> = Vec::new();
    let mut chunk = 4usize << 20;
    loop {
        let target = buf.len().saturating_add(chunk).min(file_len);
        let old_len = buf.len();
        buf.resize(target, 0);
        let mut filled = old_len;
        while filled < target {
            let n = file
                .read(&mut buf[filled..target])
                .map_err(|e| err(format!("cannot read GGUF {path}: {e}")))?;
            if n == 0 {
                break;
            }
            filled += n;
        }
        buf.truncate(filled);
        match parse_gguf_header(&buf) {
            Ok(header) => return Ok(header),
            Err(e) if filled < file_len && e.message().contains("truncated") => {
                chunk *= 2;
            }
            Err(e) => return Err(e),
        }
    }
}

/// Parse a GGUF byte buffer (header, metadata, tensor infos, data section).
pub fn read_gguf(bytes: &[u8]) -> Result<GgufFile, MmnError> {
    let header = parse_gguf_header(bytes)?;
    if header.data_start > bytes.len() {
        return Err(err("GGUF data section start beyond end of file"));
    }
    Ok(GgufFile {
        version: header.version,
        metadata: header.metadata,
        tensors: header.tensors,
        alignment: header.alignment,
        data: bytes[header.data_start..].to_vec(),
    })
}

fn write_string(out: &mut Vec<u8>, s: &str) {
    out.extend_from_slice(&(s.len() as u64).to_le_bytes());
    out.extend_from_slice(s.as_bytes());
}

fn write_value(out: &mut Vec<u8>, value: &GgufValue) {
    match value {
        GgufValue::U8(v) => out.push(*v),
        GgufValue::I8(v) => out.push(*v as u8),
        GgufValue::U16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I16(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::U32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F32(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::Bool(v) => out.push(*v as u8),
        GgufValue::String(s) => write_string(out, s),
        GgufValue::Array(items) => {
            let elem_type = items.first().map(|v| v.type_id()).unwrap_or(8);
            out.extend_from_slice(&elem_type.to_le_bytes());
            out.extend_from_slice(&(items.len() as u64).to_le_bytes());
            for item in items {
                write_value(out, item);
            }
        }
        GgufValue::U64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::I64(v) => out.extend_from_slice(&v.to_le_bytes()),
        GgufValue::F64(v) => out.extend_from_slice(&v.to_le_bytes()),
    }
}

/// A tensor scheduled for writing: `shape` is row-major (C order).
pub struct GgufWriteTensor<'a> {
    pub name: String,
    pub shape: Vec<usize>,
    pub values: &'a [f32],
    pub ggml_type: GgmlType,
}

/// Serialize a GGUF v3 file. Tensors encode as F32 or Q8_0.
pub fn write_gguf(
    metadata: &[(String, GgufValue)],
    tensors: &[GgufWriteTensor<'_>],
) -> Result<Vec<u8>, MmnError> {
    let alignment = DEFAULT_ALIGNMENT;
    let mut out = Vec::new();
    out.extend_from_slice(GGUF_MAGIC);
    out.extend_from_slice(&GGUF_VERSION.to_le_bytes());
    out.extend_from_slice(&(tensors.len() as u64).to_le_bytes());
    // +1 for general.alignment always written first.
    out.extend_from_slice(&((metadata.len() + 1) as u64).to_le_bytes());
    write_string(&mut out, "general.alignment");
    out.extend_from_slice(&4u32.to_le_bytes()); // type U32
    out.extend_from_slice(&(alignment as u32).to_le_bytes());
    for (key, value) in metadata {
        write_string(&mut out, key);
        out.extend_from_slice(&value.type_id().to_le_bytes());
        write_value(&mut out, value);
    }
    // Encode tensor payloads (block-parallel — quantization searches
    // dominate), then lay out aligned offsets.
    let payloads = encode_payloads(tensors)?;
    let mut offset = 0usize;
    let mut offsets = Vec::with_capacity(tensors.len());
    for payload in &payloads {
        offsets.push(offset);
        offset += payload.len().div_ceil(alignment) * alignment;
    }
    for (t, tensor_offset) in tensors.iter().zip(&offsets) {
        write_string(&mut out, &t.name);
        // Store dims GGML-style: reversed row-major shape.
        let dims: Vec<u64> = t.shape.iter().rev().map(|&d| d as u64).collect();
        out.extend_from_slice(&(dims.len() as u32).to_le_bytes());
        for d in &dims {
            out.extend_from_slice(&d.to_le_bytes());
        }
        out.extend_from_slice(&t.ggml_type.type_id().to_le_bytes());
        out.extend_from_slice(&(*tensor_offset as u64).to_le_bytes());
    }
    // Pad to the aligned data section, then write aligned payloads.
    let data_start = out.len().div_ceil(alignment) * alignment;
    out.resize(data_start, 0);
    for payload in &payloads {
        let padded = payload.len().div_ceil(alignment) * alignment;
        out.extend_from_slice(payload);
        out.resize(out.len() + (padded - payload.len()), 0);
    }
    Ok(out)
}

/// Encode one contiguous run of values for a given GGML type.
///
/// Every supported encoder maps independent fixed-size blocks to fixed-size
/// output, so any slice whose length is a multiple of the block size can be
/// encoded standalone and concatenated.
fn encode_values(ggml_type: GgmlType, values: &[f32]) -> Result<Vec<u8>, MmnError> {
    match ggml_type {
        GgmlType::F32 => Ok(values
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect::<Vec<u8>>()),
        GgmlType::F16 => Ok(encode_f16(values)),
        GgmlType::Q8_0 => quantize_q8_0(values),
        GgmlType::Q4_0 => quantize_q4_0(values),
        GgmlType::Q4_1 => quantize_q4_1(values),
        GgmlType::Q5_0 => quantize_q5_0(values),
        GgmlType::Q5_1 => quantize_q5_1(values),
        GgmlType::Tq1_0 => quantize_tq1_0(values),
        GgmlType::Tq2_0 => quantize_tq2_0(values),
        GgmlType::Mxfp4 => quantize_mxfp4(values),
        GgmlType::Q2K => quantize_q2_k(values),
        GgmlType::Q3K => quantize_q3_k(values),
        GgmlType::Q4K => quantize_q4_k(values),
        GgmlType::Q5K => quantize_q5_k(values),
        GgmlType::Q6K => quantize_q6_k(values),
        GgmlType::Q8K => quantize_q8_k(values),
        GgmlType::Iq4Nl => quantize_iq4_nl(values),
        GgmlType::Iq4Xs => quantize_iq4_xs(values),
        other => Err(err(format!(
            "GGUF writer encodes F32/F16, Q4_0..Q8_0, Q2_K..Q8_K, IQ4_NL/IQ4_XS, TQ1_0/TQ2_0, or MXFP4, not {other:?}"
        ))),
    }
}

/// Independent-block size (elements) for each encodable GGML type.
fn encode_block_elems(ggml_type: GgmlType) -> usize {
    match ggml_type {
        GgmlType::F32 | GgmlType::F16 => 1,
        GgmlType::Q8_0
        | GgmlType::Q4_0
        | GgmlType::Q4_1
        | GgmlType::Q5_0
        | GgmlType::Q5_1
        | GgmlType::Mxfp4
        | GgmlType::Iq4Nl => 32,
        _ => 256,
    }
}

/// Target elements per parallel encoding task (small enough to balance,
/// large enough to amortize per-task overhead).
const ENCODE_SEGMENT_ELEMS: usize = 64 * 1024;

/// Encode all tensor payloads, splitting large tensors into block-aligned
/// segments processed by a work-stealing worker pool (a handful of large
/// matrices dominate real checkpoints, so per-tensor parallelism alone
/// leaves cores idle).
fn encode_payloads(tensors: &[GgufWriteTensor<'_>]) -> Result<Vec<Vec<u8>>, MmnError> {
    for t in tensors {
        let numel: usize = t.shape.iter().product();
        if numel != t.values.len() {
            return Err(err(format!(
                "GGUF tensor {} shape {:?} needs {numel} values, got {}",
                t.name,
                t.shape,
                t.values.len()
            )));
        }
    }
    // (tensor index, segment values); segments of one tensor stay in order.
    let mut tasks: Vec<(usize, &[f32])> = Vec::new();
    let mut segments_per_tensor = vec![0usize; tensors.len()];
    for (i, t) in tensors.iter().enumerate() {
        let block = encode_block_elems(t.ggml_type);
        let seg = ENCODE_SEGMENT_ELEMS.div_ceil(block) * block;
        if t.values.len() <= seg || !t.values.len().is_multiple_of(block) {
            // Small tensors, and misaligned ones so the encoder itself
            // reports its block-multiple error for the full tensor.
            tasks.push((i, t.values));
            segments_per_tensor[i] = 1;
        } else {
            for chunk in t.values.chunks(seg) {
                tasks.push((i, chunk));
                segments_per_tensor[i] += 1;
            }
        }
    }
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(tasks.len().max(1));
    let types: Vec<GgmlType> = tensors.iter().map(|t| t.ggml_type).collect();
    let encoded: Vec<Result<Vec<u8>, MmnError>> = if workers <= 1 || tasks.len() <= 1 {
        tasks
            .iter()
            .map(|(i, values)| encode_values(types[*i], values))
            .collect()
    } else {
        let next = std::sync::atomic::AtomicUsize::new(0);
        let mut collected: Vec<(usize, Result<Vec<u8>, MmnError>)> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = (0..workers)
                    .map(|_| {
                        scope.spawn(|| {
                            let mut done = Vec::new();
                            loop {
                                let idx =
                                    next.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                                let Some((tensor_idx, values)) = tasks.get(idx) else {
                                    break;
                                };
                                done.push((idx, encode_values(types[*tensor_idx], values)));
                            }
                            done
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().expect("gguf encode worker panicked"))
                    .collect()
            });
        collected.sort_by_key(|(idx, _)| *idx);
        collected.into_iter().map(|(_, r)| r).collect()
    };
    let mut payloads: Vec<Vec<u8>> = Vec::with_capacity(tensors.len());
    let mut cursor = 0usize;
    for count in segments_per_tensor {
        let mut payload = Vec::new();
        for encoded_segment in encoded[cursor..cursor + count].iter() {
            match encoded_segment {
                Ok(bytes) => payload.extend_from_slice(bytes),
                Err(e) => {
                    return Err(MmnError::Other {
                        message: e.to_string(),
                    })
                }
            }
        }
        payloads.push(payload);
        cursor += count;
    }
    Ok(payloads)
}

/// True when `bytes` begin with the GGUF magic.
pub fn is_gguf_bytes(bytes: &[u8]) -> bool {
    bytes.len() >= 4 && &bytes[..4] == GGUF_MAGIC
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample_file() -> Vec<u8> {
        let values: Vec<f32> = (0..12).map(|i| i as f32 * 0.5).collect();
        let tensors = vec![GgufWriteTensor {
            name: "token_embd.weight".into(),
            shape: vec![4, 3],
            values: &values,
            ggml_type: GgmlType::F32,
        }];
        let meta = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("mmn".into()),
            ),
            ("mmn.block_count".to_string(), GgufValue::U32(2)),
            ("mmn.rope.freq_base".to_string(), GgufValue::F32(10000.0)),
            ("mmn.use_rope".to_string(), GgufValue::Bool(true)),
            (
                "tokenizer.ggml.tokens".to_string(),
                GgufValue::Array(vec![
                    GgufValue::String("<s>".into()),
                    GgufValue::String("hi".into()),
                ]),
            ),
        ];
        write_gguf(&meta, &tensors).unwrap()
    }

    #[test]
    fn roundtrip_metadata_and_tensor() {
        let bytes = sample_file();
        assert!(is_gguf_bytes(&bytes));
        let file = read_gguf(&bytes).unwrap();
        assert_eq!(file.version, GGUF_VERSION);
        assert_eq!(
            file.metadata.get("general.architecture").unwrap().as_str(),
            Some("mmn")
        );
        assert_eq!(
            file.metadata.get("mmn.block_count").unwrap().as_u64(),
            Some(2)
        );
        assert_eq!(
            file.metadata.get("mmn.use_rope").unwrap().as_bool(),
            Some(true)
        );
        let rope = file.metadata.get("mmn.rope.freq_base").unwrap().as_f64().unwrap();
        assert!((rope - 10000.0).abs() < 1e-3);
        match file.metadata.get("tokenizer.ggml.tokens").unwrap() {
            GgufValue::Array(items) => assert_eq!(items.len(), 2),
            other => panic!("expected array, got {other:?}"),
        }
        let info = file.find_tensor("token_embd.weight").unwrap();
        assert_eq!(info.row_major_shape(), vec![4, 3]);
        let (shape, values) = file.tensor_f32(info).unwrap();
        assert_eq!(shape, vec![4, 3]);
        assert_eq!(values[3], 1.5);
    }

    #[test]
    fn tensor_data_is_aligned() {
        let bytes = sample_file();
        let file = read_gguf(&bytes).unwrap();
        let info = file.find_tensor("token_embd.weight").unwrap();
        assert_eq!(info.offset as usize % DEFAULT_ALIGNMENT, 0);
        assert_eq!(file.alignment, DEFAULT_ALIGNMENT);
    }

    #[test]
    fn q8_0_tensor_roundtrips_in_container() {
        let values: Vec<f32> = (0..64).map(|i| (i as f32 - 30.0) * 0.1).collect();
        let tensors = vec![GgufWriteTensor {
            name: "blk.0.ffn_up.weight".into(),
            shape: vec![2, 32],
            values: &values,
            ggml_type: GgmlType::Q8_0,
        }];
        let bytes = write_gguf(&[], &tensors).unwrap();
        let file = read_gguf(&bytes).unwrap();
        let info = file.find_tensor("blk.0.ffn_up.weight").unwrap();
        assert_eq!(info.ggml_type, GgmlType::Q8_0);
        let (_, back) = file.tensor_f32(info).unwrap();
        for (a, b) in values.iter().zip(&back) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn bad_magic_and_version_error() {
        assert!(read_gguf(b"NOPE").is_err());
        let mut bytes = sample_file();
        bytes[4] = 99; // version
        let e = read_gguf(&bytes).err().unwrap();
        assert!(e.message().contains("version"));
    }

    #[test]
    fn truncated_file_errors() {
        let bytes = sample_file();
        assert!(read_gguf(&bytes[..40]).is_err());
    }

    #[test]
    fn reads_version_2_header() {
        let mut bytes = sample_file();
        bytes[4..8].copy_from_slice(&2u32.to_le_bytes());
        let file = read_gguf(&bytes).unwrap();
        assert_eq!(file.version, 2);
    }

    /// Segmented parallel encoding must be byte-identical to encoding the
    /// whole tensor in one call (blocks are independent by construction).
    #[test]
    fn parallel_segmented_encode_matches_whole_tensor() {
        // > ENCODE_SEGMENT_ELEMS so the writer splits into several segments.
        let n = ENCODE_SEGMENT_ELEMS * 2 + 256 * 3;
        let values: Vec<f32> = (0..n).map(|i| ((i * 37 % 511) as f32 - 255.0) * 0.01).collect();
        for (ggml_type, reference) in [
            (GgmlType::Q6K, quantize_q6_k(&values).unwrap()),
            (GgmlType::Q8_0, quantize_q8_0(&values).unwrap()),
            (GgmlType::F16, encode_f16(&values)),
        ] {
            let tensors = vec![GgufWriteTensor {
                name: "big.weight".into(),
                shape: vec![n / 256, 256],
                values: &values,
                ggml_type,
            }];
            let payloads = encode_payloads(&tensors).unwrap();
            assert_eq!(
                payloads[0], reference,
                "segmented {ggml_type:?} encode diverged from whole-tensor encode"
            );
        }
    }

    /// A large tensor whose length is not a block multiple must still get
    /// the encoder's own block-multiple error (single-task path).
    #[test]
    fn parallel_encode_misaligned_large_tensor_errors() {
        let n = ENCODE_SEGMENT_ELEMS * 2 + 7;
        let values = vec![0.5f32; n];
        let tensors = vec![GgufWriteTensor {
            name: "odd.weight".into(),
            shape: vec![n],
            values: &values,
            ggml_type: GgmlType::Q8_0,
        }];
        let e = encode_payloads(&tensors).err().unwrap();
        assert!(
            e.message().contains("multiple"),
            "expected block-multiple error, got: {}",
            e.message()
        );
    }
}
