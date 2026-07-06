//! From-scratch ONNX model reader: extracts every graph initializer
//! (weights) from a `.onnx` protobuf file. No onnx/protobuf library linked.

use super::proto::{packed_varints, walk_message, ProtoValue};
use half::{bf16, f16};
use mmn_core::MmnError;
use std::fs;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

// TensorProto field numbers.
const F_DIMS: u64 = 1;
const F_DATA_TYPE: u64 = 2;
const F_FLOAT_DATA: u64 = 4;
const F_INT32_DATA: u64 = 5;
const F_INT64_DATA: u64 = 7;
const F_NAME: u64 = 8;
const F_RAW_DATA: u64 = 9;
const F_DOUBLE_DATA: u64 = 10;
const F_UINT64_DATA: u64 = 11;
const F_DATA_LOCATION: u64 = 14;

#[derive(Default)]
struct TensorFields<'a> {
    name: String,
    dims: Vec<usize>,
    data_type: u64,
    raw_data: Option<&'a [u8]>,
    float_data: Vec<f32>,
    int_data: Vec<i64>,
    double_data: Vec<f64>,
    external: bool,
}

fn parse_tensor(buf: &[u8]) -> Result<TensorFields<'_>, MmnError> {
    let mut t = TensorFields::default();
    walk_message(buf, |field, value| {
        match (field, &value) {
            (F_DIMS, ProtoValue::Varint(v)) => t.dims.push(*v as usize),
            (F_DIMS, ProtoValue::Bytes(b)) => {
                t.dims.extend(packed_varints(b)?.into_iter().map(|v| v as usize));
            }
            (F_DATA_TYPE, ProtoValue::Varint(v)) => t.data_type = *v,
            (F_NAME, ProtoValue::Bytes(b)) => {
                t.name = String::from_utf8_lossy(b).into_owned();
            }
            (F_RAW_DATA, ProtoValue::Bytes(b)) => t.raw_data = Some(b),
            (F_FLOAT_DATA, ProtoValue::Bytes(b)) => {
                for chunk in b.chunks_exact(4) {
                    t.float_data
                        .push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
                }
            }
            (F_FLOAT_DATA, ProtoValue::Fixed32(v)) => t.float_data.push(f32::from_bits(*v)),
            (F_DOUBLE_DATA, ProtoValue::Bytes(b)) => {
                for chunk in b.chunks_exact(8) {
                    t.double_data.push(f64::from_le_bytes([
                        chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6],
                        chunk[7],
                    ]));
                }
            }
            (F_DOUBLE_DATA, ProtoValue::Fixed64(v)) => {
                t.double_data.push(f64::from_bits(*v));
            }
            (F_INT32_DATA | F_INT64_DATA | F_UINT64_DATA, ProtoValue::Bytes(b)) => {
                t.int_data
                    .extend(packed_varints(b)?.into_iter().map(|v| v as i64));
            }
            (F_INT32_DATA | F_INT64_DATA | F_UINT64_DATA, ProtoValue::Varint(v)) => {
                t.int_data.push(*v as i64);
            }
            (F_DATA_LOCATION, ProtoValue::Varint(v)) => t.external = *v == 1,
            _ => {}
        }
        Ok(())
    })?;
    Ok(t)
}

/// Decode raw little-endian ONNX tensor bytes per data type.
fn decode_raw(data_type: u64, raw: &[u8], name: &str) -> Result<Vec<f32>, MmnError> {
    let convert = |item: usize, f: &dyn Fn(&[u8]) -> f32| -> Result<Vec<f32>, MmnError> {
        if !raw.len().is_multiple_of(item) {
            return Err(err(format!(
                "onnx tensor {name}: raw length not a multiple of {item}"
            )));
        }
        Ok(raw.chunks_exact(item).map(f).collect())
    };
    match data_type {
        1 => convert(4, &|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]])),
        2 => convert(1, &|c| c[0] as f32),
        3 => convert(1, &|c| (c[0] as i8) as f32),
        4 => convert(2, &|c| u16::from_le_bytes([c[0], c[1]]) as f32),
        5 => convert(2, &|c| i16::from_le_bytes([c[0], c[1]]) as f32),
        6 => convert(4, &|c| i32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32),
        7 => convert(8, &|c| {
            i64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
        }),
        9 => convert(1, &|c| if c[0] != 0 { 1.0 } else { 0.0 }),
        10 => convert(2, &|c| f16::from_le_bytes([c[0], c[1]]).to_f32()),
        11 => convert(8, &|c| {
            f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
        }),
        12 => convert(4, &|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f32),
        13 => convert(8, &|c| {
            u64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32
        }),
        16 => convert(2, &|c| bf16::from_le_bytes([c[0], c[1]]).to_f32()),
        other => Err(err(format!(
            "onnx tensor {name}: data_type {other} not supported (numeric types only)"
        ))),
    }
}

fn tensor_to_array(t: TensorFields<'_>) -> Result<super::NamedArray, MmnError> {
    if t.external {
        return Err(err(format!(
            "onnx tensor {} uses external data files; re-export with save_as_external_data=False",
            t.name
        )));
    }
    let numel: usize = t.dims.iter().product();
    let values = if let Some(raw) = t.raw_data {
        decode_raw(t.data_type, raw, &t.name)?
    } else if !t.float_data.is_empty() {
        t.float_data
    } else if !t.double_data.is_empty() {
        t.double_data.into_iter().map(|v| v as f32).collect()
    } else if !t.int_data.is_empty() {
        t.int_data.into_iter().map(|v| v as f32).collect()
    } else if numel == 0 {
        Vec::new()
    } else {
        return Err(err(format!("onnx tensor {} carries no data", t.name)));
    };
    if values.len() != numel {
        return Err(err(format!(
            "onnx tensor {}: {} values but shape {:?} needs {numel}",
            t.name,
            values.len(),
            t.dims
        )));
    }
    Ok((t.name, t.dims, values))
}

/// Read every graph initializer (weight) in an ONNX model file.
pub fn read_onnx_arrays_bytes(bytes: &[u8]) -> Result<Vec<super::NamedArray>, MmnError> {
    let mut graph: Option<&[u8]> = None;
    walk_message(bytes, |field, value| {
        if field == 7 {
            if let Some(b) = value.as_bytes() {
                graph = Some(b);
            }
        }
        Ok(())
    })
    .map_err(|_| err("not an ONNX file (protobuf parse failed)"))?;
    let graph = graph.ok_or_else(|| err("onnx model has no graph (field 7)"))?;
    let mut out = Vec::new();
    walk_message(graph, |field, value| {
        // GraphProto.initializer = 5
        if field == 5 {
            if let Some(body) = value.as_bytes() {
                out.push(tensor_to_array(parse_tensor(body)?)?);
            }
        }
        Ok(())
    })?;
    if out.is_empty() {
        return Err(err(
            "onnx model has no initializers (weights); only graphs with stored weights are supported",
        ));
    }
    Ok(out)
}

/// Read every weight in an `.onnx` file on disk.
pub fn read_onnx_arrays(path: &str) -> Result<Vec<super::NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read onnx {path}: {e}")))?;
    read_onnx_arrays_bytes(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Handcraft a minimal ModelProto: graph(7) { initializer(5) {...} }.
    fn build_model(tensors: &[(&str, Vec<usize>, Vec<f32>)]) -> Vec<u8> {
        fn varint(mut v: u64, out: &mut Vec<u8>) {
            loop {
                let byte = (v & 0x7F) as u8;
                v >>= 7;
                if v == 0 {
                    out.push(byte);
                    return;
                }
                out.push(byte | 0x80);
            }
        }
        fn field_bytes(field: u64, body: &[u8], out: &mut Vec<u8>) {
            varint(field << 3 | 2, out);
            varint(body.len() as u64, out);
            out.extend_from_slice(body);
        }
        let mut graph = Vec::new();
        for (name, dims, values) in tensors {
            let mut tensor = Vec::new();
            for &d in dims {
                varint(1 << 3, &mut tensor);
                varint(d as u64, &mut tensor);
            }
            varint(2 << 3, &mut tensor);
            varint(1, &mut tensor); // FLOAT
            field_bytes(8, name.as_bytes(), &mut tensor);
            let raw: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
            field_bytes(9, &raw, &mut tensor);
            field_bytes(5, &tensor, &mut graph);
        }
        let mut model = Vec::new();
        varint(1 << 3, &mut model);
        varint(9, &mut model); // ir_version
        field_bytes(7, &graph, &mut model);
        model
    }

    #[test]
    fn reads_initializers_from_handcrafted_model() {
        let model = build_model(&[
            ("w", vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("b", vec![3], vec![-1.0, 0.0, 1.0]),
        ]);
        let arrays = read_onnx_arrays_bytes(&model).unwrap();
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].0, "w");
        assert_eq!(arrays[0].1, vec![2, 3]);
        assert_eq!(arrays[0].2[4], 5.0);
        assert_eq!(arrays[1].2, vec![-1.0, 0.0, 1.0]);
    }

    #[test]
    fn no_graph_and_no_weights_error() {
        assert!(read_onnx_arrays_bytes(&[0x08, 0x09]).is_err());
        let empty_graph = {
            let mut m = vec![0x08, 0x09];
            m.extend_from_slice(&[0x3A, 0x00]); // graph(7) empty
            m
        };
        let e = read_onnx_arrays_bytes(&empty_graph).err().unwrap();
        assert!(e.message().contains("no initializers"));
    }

    #[test]
    fn garbage_bytes_error() {
        assert!(read_onnx_arrays_bytes(b"definitely not protobuf \xff\xff").is_err());
    }
}
