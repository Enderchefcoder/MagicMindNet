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

pub(crate) fn tensor_to_entry(t: &Tensor) -> serde_json::Value {
    let arr = t.data.as_standard_layout().to_owned();
    let shape: Vec<usize> = arr.shape().to_vec();
    let data: Vec<u8> = arr.iter().flat_map(|f| f.to_le_bytes()).collect();
    serde_json::json!({
        "dtype": "F32",
        "shape": shape,
        "data": data,
    })
}

pub(crate) fn require_tensor_entry<'a>(
    tensors: &'a serde_json::Value,
    key: &str,
) -> Result<&'a serde_json::Value, MmnError> {
    let entry = &tensors[key];
    if entry.is_object() {
        Ok(entry)
    } else {
        Err(MmnError::Other {
            message: format!("checkpoint missing required tensor: {key}"),
        })
    }
}

pub(crate) fn json_byte(v: &serde_json::Value) -> Result<u8, MmnError> {
    v.as_u64()
        .filter(|&n| n <= 255)
        .map(|n| n as u8)
        .ok_or_else(|| MmnError::Other {
            message: "tensor data byte invalid".into(),
        })
}

pub(crate) fn expect_tensor_shape(t: &Tensor, expected: &[usize], name: &str) -> Result<(), MmnError> {
    let shape: Vec<usize> = t.data.shape().iter().copied().collect();
    if shape.as_slice() != expected {
        return Err(MmnError::Other {
            message: format!("{name} shape mismatch: expected {expected:?}, got {shape:?}"),
        });
    }
    Ok(())
}

pub(crate) fn tensor_from_entry(v: &serde_json::Value) -> Result<Tensor, MmnError> {
    let embed = v.as_object().ok_or_else(|| MmnError::Other {
        message: "tensor entry must be object".into(),
    })?;
    let shape: Vec<usize> = embed["shape"]
        .as_array()
        .ok_or_else(|| MmnError::Other {
            message: "tensor missing shape".into(),
        })?
        .iter()
        .map(|x| x.as_u64().unwrap() as usize)
        .collect();
    let bytes = embed["data"].as_array().ok_or_else(|| MmnError::Other {
        message: "tensor missing data".into(),
    })?;
    let mut vec = Vec::with_capacity(bytes.len() / 4);
    for chunk in bytes.chunks(4) {
        if chunk.len() != 4 {
            return Err(MmnError::Other {
                message: "tensor data truncated".into(),
            });
        }
        vec.push(f32::from_le_bytes([
            json_byte(&chunk[0])?,
            json_byte(&chunk[1])?,
            json_byte(&chunk[2])?,
            json_byte(&chunk[3])?,
        ]));
    }
    if vec.len() != shape.iter().product::<usize>() {
        return Err(MmnError::Other {
            message: "tensor data length mismatch".into(),
        });
    }
    Ok(Tensor::from_array(
        ArrayD::from_shape_vec(ndarray::IxDyn(&shape), vec).map_err(|e| MmnError::Other {
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
