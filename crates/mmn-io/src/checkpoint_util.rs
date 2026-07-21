//! Shared checkpoint JSON/tensor helpers (extracted from `lib.rs` for maintainability).

use mmn_core::{MmnError, Tensor};
use ndarray::ArrayD;
use std::fs;
use std::path::Path;

pub(crate) fn write_file_create_parents(path: &str, contents: impl AsRef<[u8]>) -> Result<(), MmnError> {
    let p = Path::new(path);
    if let Some(parent) = p.parent() {
        if !parent.as_os_str().is_empty() {
            fs::create_dir_all(parent).map_err(|e| MmnError::Other {
                message: e.to_string(),
            })?;
        }
    }
    fs::write(path, contents).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })
}

/// One serialized tensor: the `{"data": [...], "dtype": "F32", "shape": [...]}`
/// JSON entry, deserialized directly into typed fields (no `Value` trees —
/// this is the hot path for the default checkpoint format).
///
/// Field order matches serde_json's alphabetical `Value` object output so
/// files stay byte-identical with earlier releases.
#[derive(serde::Serialize, serde::Deserialize)]
pub(crate) struct TensorEntry {
    pub data: Vec<u8>,
    pub dtype: String,
    pub shape: Vec<usize>,
}

/// Named tensor entries, ordered like serde_json's sorted `Value` objects.
pub(crate) type TensorMap = std::collections::BTreeMap<String, TensorEntry>;

pub(crate) fn tensor_to_entry(t: &Tensor) -> TensorEntry {
    let arr = t.data.as_standard_layout().to_owned();
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<u8> = arr.iter().flat_map(|f| f.to_le_bytes()).collect();
    TensorEntry {
        data,
        dtype: "F32".to_string(),
        shape,
    }
}

pub(crate) fn require_tensor_entry<'a>(
    tensors: &'a TensorMap,
    key: &str,
) -> Result<&'a TensorEntry, MmnError> {
    tensors.get(key).ok_or_else(|| MmnError::Other {
        message: format!("checkpoint missing required tensor: {key}"),
    })
}

/// Optional tensor lookup (GGUF RMS `output_norm` has weight only — no bias).
pub(crate) fn optional_tensor_entry<'a>(
    tensors: &'a TensorMap,
    key: &str,
) -> Option<&'a TensorEntry> {
    tensors.get(key)
}

pub(crate) fn expect_tensor_shape(t: &Tensor, expected: &[usize], name: &str) -> Result<(), MmnError> {
    let shape: Vec<usize> = t.data.shape().to_vec();
    if shape.as_slice() != expected {
        return Err(MmnError::Other {
            message: format!("{name} shape mismatch: expected {expected:?}, got {shape:?}"),
        });
    }
    Ok(())
}

pub(crate) fn tensor_from_entry(entry: &TensorEntry) -> Result<Tensor, MmnError> {
    if !entry.data.len().is_multiple_of(4) {
        return Err(MmnError::Other {
            message: "tensor data truncated".into(),
        });
    }
    let vec: Vec<f32> = entry
        .data
        .chunks_exact(4)
        .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect();
    if vec.len() != entry.shape.iter().product::<usize>() {
        return Err(MmnError::Other {
            message: "tensor data length mismatch".into(),
        });
    }
    Ok(Tensor::from_array(
        ArrayD::from_shape_vec(ndarray::IxDyn(&entry.shape), vec).map_err(|e| MmnError::Other {
            message: e.to_string(),
        })?,
        true,
    ))
}

pub(crate) fn quantize_tensor(t: &mut Tensor, scale: f32) {
    let mut w = t.data.as_ref().clone();
    w.mapv_inplace(|x| (x * scale).round() / scale);
    *t = Tensor::from_array(w, true);
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmn_core::Tensor;

    #[test]
    fn tensor_roundtrip_entry() {
        let t = Tensor::from_array(ndarray::arr2(&[[1.0, 2.0], [3.0, 4.0]]).into_dyn(), false);
        let entry = tensor_to_entry(&t);
        let back = tensor_from_entry(&entry).unwrap();
        assert_eq!(back.shape, t.shape);
        assert!((back.data[[0, 0]] - 1.0).abs() < 1e-6);
    }
}
