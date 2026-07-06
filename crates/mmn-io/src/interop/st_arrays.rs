//! Generic named-array IO over the from-scratch safetensors codec.
//!
//! Reads every dtype the safetensors spec defines (BOOL through F64) into
//! `f32` arrays, and writes `f32` arrays back — the same array-level bridge
//! `read_npz_arrays` / `read_h5_arrays` provide for other ecosystems.

use super::NamedArray;
use crate::st_codec::{serialize, Dtype, SafeTensors, TensorView};
use half::{bf16, f16};
use mmn_core::MmnError;
use std::fs;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// Decode one safetensors view into `f32` values (all spec dtypes).
fn view_to_f32(name: &str, view: &TensorView<'_>) -> Result<Vec<f32>, MmnError> {
    let data = view.data();
    let dtype = view.dtype();
    let elem = dtype.size();
    if !data.len().is_multiple_of(elem) {
        return Err(err(format!(
            "tensor {name}: byte length {} not a multiple of {dtype:?} element size {elem}",
            data.len()
        )));
    }
    let values = match dtype {
        Dtype::BOOL => data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect(),
        Dtype::U8 => data.iter().map(|&b| b as f32).collect(),
        Dtype::I8 => data.iter().map(|&b| b as i8 as f32).collect(),
        Dtype::I16 => data
            .chunks_exact(2)
            .map(|c| i16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        Dtype::U16 => data
            .chunks_exact(2)
            .map(|c| u16::from_le_bytes([c[0], c[1]]) as f32)
            .collect(),
        Dtype::F16 => data
            .chunks_exact(2)
            .map(|c| f16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::BF16 => data
            .chunks_exact(2)
            .map(|c| bf16::from_le_bytes([c[0], c[1]]).to_f32())
            .collect(),
        Dtype::I32 => data
            .chunks_exact(4)
            .map(|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        Dtype::U32 => data
            .chunks_exact(4)
            .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32)
            .collect(),
        Dtype::F32 => data
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
            .collect(),
        Dtype::F64 => data
            .chunks_exact(8)
            .map(|c| {
                f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            })
            .collect(),
        Dtype::I64 => data
            .chunks_exact(8)
            .map(|c| {
                i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            })
            .collect(),
        Dtype::U64 => data
            .chunks_exact(8)
            .map(|c| {
                u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
            })
            .collect(),
    };
    Ok(values)
}

/// Read every tensor in a safetensors buffer as `(name, shape, f32 values)`.
pub fn read_safetensors_arrays_bytes(bytes: &[u8]) -> Result<Vec<NamedArray>, MmnError> {
    let st = SafeTensors::deserialize(bytes).map_err(|e| err(e.to_string()))?;
    let mut names = st.names();
    names.sort_unstable();
    let mut out = Vec::with_capacity(names.len());
    for name in names {
        let view = st.tensor(name).map_err(|e| err(e.to_string()))?;
        let values = view_to_f32(name, &view)?;
        out.push((name.to_string(), view.shape().to_vec(), values));
    }
    Ok(out)
}

/// Read every tensor in a `.safetensors` file on disk.
pub fn read_safetensors_arrays(path: &str) -> Result<Vec<NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read {path}: {e}")))?;
    read_safetensors_arrays_bytes(&bytes)
}

/// Serialize named `f32` arrays as a safetensors buffer.
pub fn write_safetensors_arrays_bytes(arrays: &[NamedArray]) -> Result<Vec<u8>, MmnError> {
    write_safetensors_arrays_bytes_dtype(arrays, "f32")
}

/// Serialize named arrays in F32, F16, or BF16 (the HF half-precision
/// conventions); values convert element-wise from f32.
pub fn write_safetensors_arrays_bytes_dtype(
    arrays: &[NamedArray],
    dtype: &str,
) -> Result<Vec<u8>, MmnError> {
    let st_dtype = match dtype {
        "f32" | "F32" | "float32" => Dtype::F32,
        "f16" | "F16" | "float16" => Dtype::F16,
        "bf16" | "BF16" | "bfloat16" => Dtype::BF16,
        other => {
            return Err(err(format!(
                "safetensors write dtype {other:?} not supported (f32/f16/bf16)"
            )))
        }
    };
    let mut views = Vec::with_capacity(arrays.len());
    for (name, shape, values) in arrays {
        let numel: usize = shape.iter().product();
        if numel != values.len() {
            return Err(err(format!(
                "array {name}: shape {shape:?} needs {numel} values, got {}",
                values.len()
            )));
        }
        let bytes: Vec<u8> = match st_dtype {
            Dtype::F32 => values.iter().flat_map(|v| v.to_le_bytes()).collect(),
            Dtype::F16 => values
                .iter()
                .flat_map(|&v| f16::from_f32(v).to_le_bytes())
                .collect(),
            Dtype::BF16 => values
                .iter()
                .flat_map(|&v| bf16::from_f32(v).to_le_bytes())
                .collect(),
            _ => unreachable!("dtype match above restricts variants"),
        };
        views.push((name.clone(), shape.clone(), bytes));
    }
    let entries: std::collections::HashMap<String, TensorView<'_>> = views
        .iter()
        .map(|(name, shape, bytes)| {
            TensorView::new(st_dtype, shape.clone(), bytes)
                .map(|view| (name.clone(), view))
                .map_err(|e| err(e.to_string()))
        })
        .collect::<Result<_, _>>()?;
    serialize(entries, None).map_err(|e| err(e.to_string()))
}

/// Write named `f32` arrays as a `.safetensors` file.
pub fn write_safetensors_arrays(path: &str, arrays: &[NamedArray]) -> Result<(), MmnError> {
    let bytes = write_safetensors_arrays_bytes(arrays)?;
    crate::checkpoint_util::write_file_create_parents(path, bytes)
}

/// Write named arrays as a `.safetensors` file in F32/F16/BF16.
pub fn write_safetensors_arrays_dtype(
    path: &str,
    arrays: &[NamedArray],
    dtype: &str,
) -> Result<(), MmnError> {
    let bytes = write_safetensors_arrays_bytes_dtype(arrays, dtype)?;
    crate::checkpoint_util::write_file_create_parents(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn f32_arrays_roundtrip() {
        let arrays = vec![
            ("b.weight".to_string(), vec![2, 3], vec![1.0, -2.0, 3.5, 0.0, 4.25, -0.5]),
            ("a.bias".to_string(), vec![2], vec![0.125, -9.0]),
        ];
        let bytes = write_safetensors_arrays_bytes(&arrays).unwrap();
        let back = read_safetensors_arrays_bytes(&bytes).unwrap();
        // Reader returns names sorted.
        assert_eq!(back[0].0, "a.bias");
        assert_eq!(back[1].0, "b.weight");
        assert_eq!(back[1].1, vec![2, 3]);
        assert_eq!(back[1].2, arrays[0].2);
        assert_eq!(back[0].2, arrays[1].2);
    }

    #[test]
    fn every_dtype_decodes_to_f32() {
        // Build one tensor per dtype by hand through the codec.
        let cases: Vec<(&str, Dtype, Vec<u8>, Vec<f32>)> = vec![
            ("bool", Dtype::BOOL, vec![0, 1, 1], vec![0.0, 1.0, 1.0]),
            ("u8", Dtype::U8, vec![0, 200], vec![0.0, 200.0]),
            ("i8", Dtype::I8, vec![0xFF, 0x7F], vec![-1.0, 127.0]),
            (
                "i16",
                Dtype::I16,
                (-300i16).to_le_bytes().iter().chain(40i16.to_le_bytes().iter()).copied().collect(),
                vec![-300.0, 40.0],
            ),
            (
                "u16",
                Dtype::U16,
                60000u16.to_le_bytes().to_vec(),
                vec![60000.0],
            ),
            (
                "f16",
                Dtype::F16,
                f16::from_f32(1.5).to_le_bytes().to_vec(),
                vec![1.5],
            ),
            (
                "bf16",
                Dtype::BF16,
                bf16::from_f32(-2.0).to_le_bytes().to_vec(),
                vec![-2.0],
            ),
            (
                "i32",
                Dtype::I32,
                (-70000i32).to_le_bytes().to_vec(),
                vec![-70000.0],
            ),
            (
                "u32",
                Dtype::U32,
                70000u32.to_le_bytes().to_vec(),
                vec![70000.0],
            ),
            (
                "f64",
                Dtype::F64,
                3.25f64.to_le_bytes().to_vec(),
                vec![3.25],
            ),
            (
                "i64",
                Dtype::I64,
                (-5i64).to_le_bytes().to_vec(),
                vec![-5.0],
            ),
            ("u64", Dtype::U64, 9u64.to_le_bytes().to_vec(), vec![9.0]),
        ];
        for (name, dtype, bytes, expected) in cases {
            let shape = vec![expected.len()];
            let view = TensorView::new(dtype, shape, &bytes).unwrap();
            let got = view_to_f32(name, &view).unwrap();
            assert_eq!(got, expected, "dtype {dtype:?}");
        }
    }

    #[test]
    fn shape_value_mismatch_errors() {
        let arrays = vec![("x".to_string(), vec![3], vec![1.0, 2.0])];
        let e = write_safetensors_arrays_bytes(&arrays).err().unwrap();
        assert!(e.message().contains("needs 3 values"));
    }
}
