//! Shared F32/F16/BF16 tensor encoding for HF safetensors IO.

use half::{bf16, f16};
use mmn_core::{MmnError, Tensor};
use crate::st_codec::{Dtype, TensorView};
use crate::st_codec::SafeTensorError;

pub const HF_CHATBOT_FORMAT: &str = "mmn-hf-safetensors-v1";
pub const HF_CLASSIFIER_FORMAT: &str = "mmn-hf-classifier-v1";

/// True when the buffer looks like a binary safetensors container:
/// little-endian u64 header length followed by a `{` JSON header.
pub fn looks_like_safetensors(bytes: &[u8]) -> bool {
    if bytes.len() < 9 {
        return false;
    }
    let header_len = u64::from_le_bytes([
        bytes[0], bytes[1], bytes[2], bytes[3], bytes[4], bytes[5], bytes[6], bytes[7],
    ]) as usize;
    header_len > 0
        && header_len.checked_add(8).is_some_and(|end| end <= bytes.len())
        && bytes[8] == b'{'
}

/// Prefer structural safetensors sniffing over "not JSON".
pub fn is_hf_binary_bytes(bytes: &[u8]) -> bool {
    looks_like_safetensors(bytes)
}

pub fn hf_err(e: SafeTensorError) -> MmnError {
    MmnError::Other {
        message: e.to_string(),
    }
}

pub fn tensor_bytes_f32(t: &Tensor) -> (Vec<usize>, Vec<u8>) {
    let arr = t.data.as_standard_layout().into_owned();
    let shape = arr.shape().to_vec();
    let bytes: Vec<u8> = arr.iter().flat_map(|f| f.to_le_bytes()).collect();
    (shape, bytes)
}

pub fn tensor_from_view(name: &str, view: &TensorView<'_>) -> Result<Tensor, MmnError> {
    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data();
    let floats = match view.dtype() {
        Dtype::F32 => decode_f32_tensor(data, &shape, name)?,
        Dtype::F16 => decode_f16_tensor(data, &shape, name)?,
        Dtype::BF16 => decode_bf16_tensor(data, &shape, name)?,
        other => {
            return Err(MmnError::Other {
                message: format!(
                    "tensor {name}: HF safetensors dtype {other:?} not supported (F32/F16/BF16 only)"
                ),
            });
        }
    };
    Ok(Tensor::from_array(floats, true))
}

fn decode_f32_tensor(data: &[u8], shape: &[usize], name: &str) -> Result<ndarray::ArrayD<f32>, MmnError> {
    if !data.len().is_multiple_of(4) {
        return Err(MmnError::Other {
            message: format!("tensor {name}: invalid F32 byte length {}", data.len()),
        });
    }
    let mut floats = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), floats).map_err(|e| MmnError::Other {
        message: format!("tensor {name}: {e}"),
    })
}

fn decode_f16_tensor(data: &[u8], shape: &[usize], name: &str) -> Result<ndarray::ArrayD<f32>, MmnError> {
    if !data.len().is_multiple_of(2) {
        return Err(MmnError::Other {
            message: format!("tensor {name}: invalid F16 byte length {}", data.len()),
        });
    }
    let floats: Vec<f32> = data
        .chunks_exact(2)
        .map(|chunk| f16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), floats).map_err(|e| MmnError::Other {
        message: format!("tensor {name}: {e}"),
    })
}

fn decode_bf16_tensor(data: &[u8], shape: &[usize], name: &str) -> Result<ndarray::ArrayD<f32>, MmnError> {
    if !data.len().is_multiple_of(2) {
        return Err(MmnError::Other {
            message: format!("tensor {name}: invalid BF16 byte length {}", data.len()),
        });
    }
    let floats: Vec<f32> = data
        .chunks_exact(2)
        .map(|chunk| bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32())
        .collect();
    ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(shape), floats).map_err(|e| MmnError::Other {
        message: format!("tensor {name}: {e}"),
    })
}
