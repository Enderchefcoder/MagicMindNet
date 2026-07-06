//! From-scratch safetensors container codec (drop-in for the `safetensors`
//! crate surface MagicMindNet uses).
//!
//! Format: `u64 header_len | header JSON | raw tensor data`. The header maps
//! tensor names to `{dtype, shape, data_offsets}` plus an optional
//! `__metadata__` string map. Offsets are validated for bounds and
//! shape/dtype consistency on read.

use std::collections::HashMap;
use std::fmt;

/// Errors from parsing or serializing safetensors containers.
#[derive(Debug)]
pub struct SafeTensorError {
    message: String,
}

impl fmt::Display for SafeTensorError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for SafeTensorError {}

fn st_err(message: impl Into<String>) -> SafeTensorError {
    SafeTensorError {
        message: message.into(),
    }
}

/// Tensor element types (names match the safetensors JSON encoding).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Dtype {
    BOOL,
    U8,
    I8,
    I16,
    U16,
    F16,
    BF16,
    I32,
    U32,
    F32,
    F64,
    I64,
    U64,
}

impl Dtype {
    pub fn size(&self) -> usize {
        match self {
            Dtype::BOOL | Dtype::U8 | Dtype::I8 => 1,
            Dtype::I16 | Dtype::U16 | Dtype::F16 | Dtype::BF16 => 2,
            Dtype::I32 | Dtype::U32 | Dtype::F32 => 4,
            Dtype::F64 | Dtype::I64 | Dtype::U64 => 8,
        }
    }

    fn name(&self) -> &'static str {
        match self {
            Dtype::BOOL => "BOOL",
            Dtype::U8 => "U8",
            Dtype::I8 => "I8",
            Dtype::I16 => "I16",
            Dtype::U16 => "U16",
            Dtype::F16 => "F16",
            Dtype::BF16 => "BF16",
            Dtype::I32 => "I32",
            Dtype::U32 => "U32",
            Dtype::F32 => "F32",
            Dtype::F64 => "F64",
            Dtype::I64 => "I64",
            Dtype::U64 => "U64",
        }
    }

    fn from_name(name: &str) -> Result<Self, SafeTensorError> {
        Ok(match name {
            "BOOL" => Dtype::BOOL,
            "U8" => Dtype::U8,
            "I8" => Dtype::I8,
            "I16" => Dtype::I16,
            "U16" => Dtype::U16,
            "F16" => Dtype::F16,
            "BF16" => Dtype::BF16,
            "I32" => Dtype::I32,
            "U32" => Dtype::U32,
            "F32" => Dtype::F32,
            "F64" => Dtype::F64,
            "I64" => Dtype::I64,
            "U64" => Dtype::U64,
            other => return Err(st_err(format!("safetensors dtype {other:?} unknown"))),
        })
    }
}

/// Borrowed view over one tensor's bytes.
#[derive(Clone)]
pub struct TensorView<'a> {
    dtype: Dtype,
    shape: Vec<usize>,
    data: &'a [u8],
}

impl<'a> TensorView<'a> {
    pub fn new(dtype: Dtype, shape: Vec<usize>, data: &'a [u8]) -> Result<Self, SafeTensorError> {
        let numel: usize = shape.iter().product();
        if numel * dtype.size() != data.len() {
            return Err(st_err(format!(
                "tensor data length {} does not match shape {:?} of {}",
                data.len(),
                shape,
                dtype.name()
            )));
        }
        Ok(Self { dtype, shape, data })
    }

    pub fn dtype(&self) -> Dtype {
        self.dtype
    }

    pub fn shape(&self) -> &[usize] {
        &self.shape
    }

    pub fn data(&self) -> &'a [u8] {
        self.data
    }
}

/// Parsed header: optional `__metadata__` string map.
pub struct Metadata {
    metadata: Option<HashMap<String, String>>,
}

impl Metadata {
    pub fn metadata(&self) -> &Option<HashMap<String, String>> {
        &self.metadata
    }
}

struct Entry {
    dtype: Dtype,
    shape: Vec<usize>,
    begin: usize,
    end: usize,
}

/// Parsed safetensors file: header entries + the data section.
pub struct SafeTensors<'a> {
    entries: Vec<(String, Entry)>,
    data: &'a [u8],
}

fn parse_header(bytes: &[u8]) -> Result<(usize, serde_json::Value), SafeTensorError> {
    if bytes.len() < 8 {
        return Err(st_err("safetensors file smaller than its header length"));
    }
    let header_len =
        u64::from_le_bytes(bytes[..8].try_into().expect("8 bytes")) as usize;
    if header_len > bytes.len().saturating_sub(8) || header_len > 100_000_000 {
        return Err(st_err("safetensors header length out of bounds"));
    }
    let header: serde_json::Value = serde_json::from_slice(&bytes[8..8 + header_len])
        .map_err(|e| st_err(format!("safetensors header JSON invalid: {e}")))?;
    Ok((header_len, header))
}

impl<'a> SafeTensors<'a> {
    /// Header length and `__metadata__` without touching tensor data.
    pub fn read_metadata(bytes: &[u8]) -> Result<(usize, Metadata), SafeTensorError> {
        let (header_len, header) = parse_header(bytes)?;
        let metadata = header.get("__metadata__").map(|m| {
            m.as_object()
                .map(|obj| {
                    obj.iter()
                        .filter_map(|(k, v)| v.as_str().map(|s| (k.clone(), s.to_string())))
                        .collect()
                })
                .unwrap_or_default()
        });
        Ok((header_len, Metadata { metadata }))
    }

    pub fn deserialize(bytes: &'a [u8]) -> Result<Self, SafeTensorError> {
        let (header_len, header) = parse_header(bytes)?;
        let data = &bytes[8 + header_len..];
        let obj = header
            .as_object()
            .ok_or_else(|| st_err("safetensors header is not a JSON object"))?;
        let mut entries = Vec::with_capacity(obj.len());
        for (name, value) in obj {
            if name == "__metadata__" {
                continue;
            }
            let dtype = Dtype::from_name(
                value["dtype"]
                    .as_str()
                    .ok_or_else(|| st_err(format!("tensor {name} missing dtype")))?,
            )?;
            let shape: Vec<usize> = value["shape"]
                .as_array()
                .ok_or_else(|| st_err(format!("tensor {name} missing shape")))?
                .iter()
                .map(|d| {
                    d.as_u64()
                        .map(|v| v as usize)
                        .ok_or_else(|| st_err(format!("tensor {name} has a bad dim")))
                })
                .collect::<Result<_, _>>()?;
            let offsets = value["data_offsets"]
                .as_array()
                .filter(|a| a.len() == 2)
                .ok_or_else(|| st_err(format!("tensor {name} missing data_offsets")))?;
            let begin = offsets[0].as_u64().unwrap_or(u64::MAX) as usize;
            let end = offsets[1].as_u64().unwrap_or(u64::MAX) as usize;
            let numel: usize = shape.iter().product();
            if end < begin || end > data.len() || end - begin != numel * dtype.size() {
                return Err(st_err(format!(
                    "tensor {name}: data_offsets [{begin}, {end}] inconsistent with shape {shape:?}"
                )));
            }
            entries.push((
                name.clone(),
                Entry {
                    dtype,
                    shape,
                    begin,
                    end,
                },
            ));
        }
        Ok(Self { entries, data })
    }

    pub fn names(&self) -> Vec<&str> {
        self.entries.iter().map(|(n, _)| n.as_str()).collect()
    }

    pub fn tensor(&self, name: &str) -> Result<TensorView<'a>, SafeTensorError> {
        let (_, entry) = self
            .entries
            .iter()
            .find(|(n, _)| n == name)
            .ok_or_else(|| st_err(format!("tensor {name} not found")))?;
        TensorView::new(
            entry.dtype,
            entry.shape.clone(),
            &self.data[entry.begin..entry.end],
        )
    }
}

/// Serialize tensors (sorted by name) + optional metadata into a container.
pub fn serialize(
    views: HashMap<String, TensorView<'_>>,
    metadata: Option<HashMap<String, String>>,
) -> Result<Vec<u8>, SafeTensorError> {
    let mut names: Vec<&String> = views.keys().collect();
    names.sort();
    let mut header = serde_json::Map::new();
    if let Some(meta) = &metadata {
        let mut names: Vec<&String> = meta.keys().collect();
        names.sort();
        let mut obj = serde_json::Map::new();
        for key in names {
            obj.insert(key.clone(), serde_json::json!(meta[key]));
        }
        header.insert("__metadata__".to_string(), serde_json::Value::Object(obj));
    }
    let mut offset = 0usize;
    for name in &names {
        let view = &views[*name];
        let end = offset + view.data.len();
        header.insert(
            (*name).clone(),
            serde_json::json!({
                "dtype": view.dtype.name(),
                "shape": view.shape,
                "data_offsets": [offset, end],
            }),
        );
        offset = end;
    }
    let mut header_bytes = serde_json::to_vec(&serde_json::Value::Object(header))
        .map_err(|e| st_err(format!("safetensors header encode failed: {e}")))?;
    // Pad with spaces so the data section is 8-byte aligned (official layout).
    while !(8 + header_bytes.len()).is_multiple_of(8) {
        header_bytes.push(b' ');
    }
    let mut out = Vec::with_capacity(8 + header_bytes.len() + offset);
    out.extend_from_slice(&(header_bytes.len() as u64).to_le_bytes());
    out.extend_from_slice(&header_bytes);
    for name in &names {
        out.extend_from_slice(views[*name].data);
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_with_metadata() {
        let a: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let b: Vec<u8> = [9.0f32].iter().flat_map(|v| v.to_le_bytes()).collect();
        let mut views = HashMap::new();
        views.insert(
            "alpha".to_string(),
            TensorView::new(Dtype::F32, vec![2, 2], &a).unwrap(),
        );
        views.insert(
            "beta".to_string(),
            TensorView::new(Dtype::F32, vec![1], &b).unwrap(),
        );
        let mut meta = HashMap::new();
        meta.insert("format".to_string(), "test-v1".to_string());
        let bytes = serialize(views, Some(meta)).unwrap();
        let (_, parsed_meta) = SafeTensors::read_metadata(&bytes).unwrap();
        assert_eq!(
            parsed_meta.metadata().as_ref().unwrap().get("format"),
            Some(&"test-v1".to_string())
        );
        let st = SafeTensors::deserialize(&bytes).unwrap();
        let mut names = st.names();
        names.sort();
        assert_eq!(names, vec!["alpha", "beta"]);
        let alpha = st.tensor("alpha").unwrap();
        assert_eq!(alpha.shape(), &[2, 2]);
        assert_eq!(alpha.dtype(), Dtype::F32);
        assert_eq!(alpha.data(), a.as_slice());
        assert!(st.tensor("gamma").is_err());
    }

    #[test]
    fn data_section_is_aligned() {
        let a: Vec<u8> = 5.0f32.to_le_bytes().to_vec();
        let mut views = HashMap::new();
        views.insert(
            "x".to_string(),
            TensorView::new(Dtype::F32, vec![1], &a).unwrap(),
        );
        let bytes = serialize(views, None).unwrap();
        let header_len = u64::from_le_bytes(bytes[..8].try_into().unwrap()) as usize;
        assert_eq!((8 + header_len) % 8, 0);
    }

    #[test]
    fn corrupt_inputs_error() {
        assert!(SafeTensors::deserialize(b"tiny").is_err());
        let mut bogus = 1_000_000u64.to_le_bytes().to_vec();
        bogus.extend_from_slice(b"{}");
        assert!(SafeTensors::deserialize(&bogus).is_err());
        // Offsets inconsistent with shape.
        let bad_header = br#"{"t":{"dtype":"F32","shape":[2],"data_offsets":[0,4]}}"#;
        let mut bytes = (bad_header.len() as u64).to_le_bytes().to_vec();
        bytes.extend_from_slice(bad_header);
        bytes.extend_from_slice(&[0u8; 8]);
        assert!(SafeTensors::deserialize(&bytes).is_err());
    }

    #[test]
    fn shape_mismatch_in_view_errors() {
        let a = [0u8; 8];
        assert!(TensorView::new(Dtype::F32, vec![3], &a).is_err());
        assert!(TensorView::new(Dtype::F16, vec![4], &a).is_ok());
    }
}
