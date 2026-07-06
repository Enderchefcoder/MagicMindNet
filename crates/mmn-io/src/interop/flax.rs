//! Flax/JAX checkpoint bridge over the from-scratch msgpack codec.
//!
//! `flax.serialization.to_bytes` packs the parameter pytree as a msgpack map
//! whose ndarray leaves are ExtType 1 (scalars: ExtType 3) wrapping
//! `packb((shape, dtype.name, tobytes()))`. This module flattens that tree
//! into `(name, shape, f32 values)` triples with `/`-joined paths, and
//! writes trees `flax.serialization.from_bytes` accepts. No msgpack, JAX,
//! or Flax library is linked.

use super::msgpack::{decode, encode, Value};
use super::NamedArray;
use half::{bf16, f16};
use mmn_core::MmnError;
use std::fs;

const EXT_NDARRAY: i8 = 1;
const EXT_NATIVE_COMPLEX: i8 = 2;
const EXT_NPSCALAR: i8 = 3;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// numpy `dtype.name` (or legacy `dtype.str`) → element size in bytes.
fn dtype_size(name: &str) -> Result<usize, MmnError> {
    Ok(match name {
        "bool" | "int8" | "uint8" | "|b1" | "|i1" | "|u1" => 1,
        "float16" | "bfloat16" | "int16" | "uint16" | "<f2" | "<i2" | "<u2" => 2,
        "float32" | "int32" | "uint32" | "<f4" | "<i4" | "<u4" => 4,
        "float64" | "int64" | "uint64" | "<f8" | "<i8" | "<u8" => 8,
        other => return Err(err(format!("flax dtype {other:?} not supported"))),
    })
}

/// Decode one little-endian element of the given dtype to f32.
fn decode_dtype_element(name: &str, b: &[u8]) -> f32 {
    match name {
        "bool" | "|b1" => {
            if b[0] != 0 {
                1.0
            } else {
                0.0
            }
        }
        "int8" | "|i1" => b[0] as i8 as f32,
        "uint8" | "|u1" => b[0] as f32,
        "float16" | "<f2" => f16::from_le_bytes([b[0], b[1]]).to_f32(),
        "bfloat16" => bf16::from_le_bytes([b[0], b[1]]).to_f32(),
        "int16" | "<i2" => i16::from_le_bytes([b[0], b[1]]) as f32,
        "uint16" | "<u2" => u16::from_le_bytes([b[0], b[1]]) as f32,
        "float32" | "<f4" => f32::from_le_bytes([b[0], b[1], b[2], b[3]]),
        "int32" | "<i4" => i32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
        "uint32" | "<u4" => u32::from_le_bytes([b[0], b[1], b[2], b[3]]) as f32,
        "float64" | "<f8" => {
            f64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }
        "int64" | "<i8" => {
            i64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }
        "uint64" | "<u8" => {
            u64::from_le_bytes([b[0], b[1], b[2], b[3], b[4], b[5], b[6], b[7]]) as f32
        }
        _ => unreachable!("dtype_size validates names"),
    }
}

/// Decode one ExtType-1/3 ndarray payload: `packb((shape, dtype, bytes))`.
fn ext_to_array(name: &str, payload: &[u8]) -> Result<(Vec<usize>, Vec<f32>), MmnError> {
    let Value::Array(parts) = decode(payload)? else {
        return Err(err(format!("flax leaf {name}: ext payload is not a tuple")));
    };
    let [shape_value, dtype_value, data_value] = parts.as_slice() else {
        return Err(err(format!(
            "flax leaf {name}: expected (shape, dtype, data), got {} items",
            parts.len()
        )));
    };
    let Value::Array(dims) = shape_value else {
        return Err(err(format!("flax leaf {name}: shape is not a tuple")));
    };
    let shape: Vec<usize> = dims
        .iter()
        .map(|d| match d {
            Value::UInt(n) => Ok(*n as usize),
            Value::Int(n) if *n >= 0 => Ok(*n as usize),
            _ => Err(err(format!("flax leaf {name}: non-integer dim"))),
        })
        .collect::<Result<_, _>>()?;
    let Value::Str(dtype) = dtype_value else {
        return Err(err(format!("flax leaf {name}: dtype is not a string")));
    };
    let Value::Bin(data) = data_value else {
        return Err(err(format!("flax leaf {name}: data is not bytes")));
    };
    let elem = dtype_size(dtype)?;
    let numel: usize = shape.iter().product();
    if data.len() != numel * elem {
        return Err(err(format!(
            "flax leaf {name}: shape {shape:?} needs {} bytes of {dtype}, got {}",
            numel * elem,
            data.len()
        )));
    }
    let values: Vec<f32> = data
        .chunks_exact(elem)
        .map(|chunk| decode_dtype_element(dtype, chunk))
        .collect();
    Ok((shape, values))
}

fn walk(value: &Value, prefix: &str, out: &mut Vec<NamedArray>) -> Result<(), MmnError> {
    match value {
        Value::Map(pairs) => {
            for (key, child) in pairs {
                let Value::Str(key) = key else {
                    return Err(err("flax map keys must be strings"));
                };
                let path = if prefix.is_empty() {
                    key.clone()
                } else {
                    format!("{prefix}/{key}")
                };
                walk(child, &path, out)?;
            }
            Ok(())
        }
        Value::Ext(ext_type, payload)
            if *ext_type == EXT_NDARRAY || *ext_type == EXT_NPSCALAR =>
        {
            let (shape, values) = ext_to_array(prefix, payload)?;
            out.push((prefix.to_string(), shape, values));
            Ok(())
        }
        Value::Ext(ext_type, _) if *ext_type == EXT_NATIVE_COMPLEX => Err(err(format!(
            "flax leaf {prefix}: complex values not supported"
        ))),
        Value::UInt(n) => {
            out.push((prefix.to_string(), vec![], vec![*n as f32]));
            Ok(())
        }
        Value::Int(n) => {
            out.push((prefix.to_string(), vec![], vec![*n as f32]));
            Ok(())
        }
        Value::F32(v) => {
            out.push((prefix.to_string(), vec![], vec![*v]));
            Ok(())
        }
        Value::F64(v) => {
            out.push((prefix.to_string(), vec![], vec![*v as f32]));
            Ok(())
        }
        Value::Bool(b) => {
            out.push((prefix.to_string(), vec![], vec![if *b { 1.0 } else { 0.0 }]));
            Ok(())
        }
        other => Err(err(format!(
            "flax leaf {prefix}: unsupported msgpack value {other:?}"
        ))),
    }
}

/// Read a Flax msgpack checkpoint into sorted `(path, shape, f32)` triples.
pub fn read_flax_arrays_bytes(bytes: &[u8]) -> Result<Vec<NamedArray>, MmnError> {
    let root = decode(bytes)?;
    if !matches!(root, Value::Map(_)) {
        return Err(err("flax checkpoint root must be a msgpack map"));
    }
    let mut out = Vec::new();
    walk(&root, "", &mut out)?;
    out.sort_by(|a, b| a.0.cmp(&b.0));
    Ok(out)
}

/// Read a Flax `.msgpack` checkpoint file.
pub fn read_flax_arrays(path: &str) -> Result<Vec<NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read {path}: {e}")))?;
    read_flax_arrays_bytes(&bytes)
}

/// Node under construction while nesting `a/b/c` paths back into maps.
enum Node {
    Tree(Vec<(String, Node)>),
    Leaf(Value),
}

fn insert(node: &mut Node, path: &str, leaf: Value, full: &str) -> Result<(), MmnError> {
    let Node::Tree(children) = node else {
        return Err(err(format!(
            "flax path {full}: prefix already used by a tensor"
        )));
    };
    let (head, rest) = match path.split_once('/') {
        Some((head, rest)) => (head, Some(rest)),
        None => (path, None),
    };
    let child = match children.iter_mut().find(|(k, _)| k == head) {
        Some((_, child)) => child,
        None => {
            children.push((
                head.to_string(),
                if rest.is_some() {
                    Node::Tree(Vec::new())
                } else {
                    Node::Leaf(Value::Nil)
                },
            ));
            &mut children.last_mut().unwrap().1
        }
    };
    match rest {
        Some(rest) => insert(child, rest, leaf, full),
        None => match child {
            Node::Leaf(slot) if matches!(slot, Value::Nil) => {
                *slot = leaf;
                Ok(())
            }
            _ => Err(err(format!("flax path {full}: duplicate or conflicting"))),
        },
    }
}

fn node_to_value(node: Node) -> Value {
    match node {
        Node::Leaf(value) => value,
        Node::Tree(children) => Value::Map(
            children
                .into_iter()
                .map(|(k, v)| (Value::Str(k), node_to_value(v)))
                .collect(),
        ),
    }
}

/// Serialize `(path, shape, f32)` triples as a Flax msgpack pytree
/// (`flax.serialization.from_bytes`-compatible; F32 ndarray leaves).
pub fn write_flax_arrays_bytes(arrays: &[NamedArray]) -> Result<Vec<u8>, MmnError> {
    let mut root = Node::Tree(Vec::new());
    for (name, shape, values) in arrays {
        let numel: usize = shape.iter().product();
        if numel != values.len() {
            return Err(err(format!(
                "array {name}: shape {shape:?} needs {numel} values, got {}",
                values.len()
            )));
        }
        let payload = encode(&Value::Array(vec![
            Value::Array(shape.iter().map(|&d| Value::UInt(d as u64)).collect()),
            Value::Str("float32".to_string()),
            Value::Bin(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
        ]));
        insert(
            &mut root,
            name,
            Value::Ext(EXT_NDARRAY, payload),
            name,
        )?;
    }
    Ok(encode(&node_to_value(root)))
}

/// Write a Flax `.msgpack` checkpoint file.
pub fn write_flax_arrays(path: &str, arrays: &[NamedArray]) -> Result<(), MmnError> {
    let bytes = write_flax_arrays_bytes(arrays)?;
    crate::checkpoint_util::write_file_create_parents(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn roundtrip_nested_tree() {
        let arrays = vec![
            (
                "params/dense/kernel".to_string(),
                vec![2, 3],
                vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            ),
            ("params/dense/bias".to_string(), vec![3], vec![0.5, -0.5, 1.5]),
            ("step".to_string(), vec![], vec![7.0]),
        ];
        let bytes = write_flax_arrays_bytes(&arrays).unwrap();
        let back = read_flax_arrays_bytes(&bytes).unwrap();
        assert_eq!(back.len(), 3);
        assert_eq!(back[0].0, "params/dense/bias");
        assert_eq!(back[0].2, vec![0.5, -0.5, 1.5]);
        assert_eq!(back[1].1, vec![2, 3]);
        assert_eq!(back[2], ("step".to_string(), vec![], vec![7.0]));
    }

    /// Byte-layout check against the documented flax encoding:
    /// map -> ExtType 1 -> packb((shape, dtype.name, tobytes())).
    #[test]
    fn reads_hand_built_flax_bytes() {
        // {"w": ext1(packb(((2,), "float32", 8 bytes)))}
        let inner = encode(&Value::Array(vec![
            Value::Array(vec![Value::UInt(2)]),
            Value::Str("float32".into()),
            Value::Bin(vec![0, 0, 128, 63, 0, 0, 0, 64]), // 1.0, 2.0 LE
        ]));
        let doc = encode(&Value::Map(vec![(
            Value::Str("w".into()),
            Value::Ext(1, inner),
        )]));
        let arrays = read_flax_arrays_bytes(&doc).unwrap();
        assert_eq!(arrays, vec![("w".to_string(), vec![2], vec![1.0, 2.0])]);
    }

    #[test]
    fn every_dtype_name_decodes() {
        let cases: Vec<(&str, Vec<u8>, Vec<f32>)> = vec![
            ("bool", vec![1, 0], vec![1.0, 0.0]),
            ("uint8", vec![200], vec![200.0]),
            ("int8", vec![0xFF], vec![-1.0]),
            ("float16", f16::from_f32(1.5).to_le_bytes().to_vec(), vec![1.5]),
            ("bfloat16", bf16::from_f32(-2.0).to_le_bytes().to_vec(), vec![-2.0]),
            ("int16", (-300i16).to_le_bytes().to_vec(), vec![-300.0]),
            ("uint16", 60000u16.to_le_bytes().to_vec(), vec![60000.0]),
            ("int32", (-70000i32).to_le_bytes().to_vec(), vec![-70000.0]),
            ("uint32", 70000u32.to_le_bytes().to_vec(), vec![70000.0]),
            ("float32", 0.25f32.to_le_bytes().to_vec(), vec![0.25]),
            ("float64", 3.25f64.to_le_bytes().to_vec(), vec![3.25]),
            ("int64", (-5i64).to_le_bytes().to_vec(), vec![-5.0]),
            ("uint64", 9u64.to_le_bytes().to_vec(), vec![9.0]),
        ];
        for (dtype, data, expected) in cases {
            let inner = encode(&Value::Array(vec![
                Value::Array(vec![Value::UInt(expected.len() as u64)]),
                Value::Str(dtype.into()),
                Value::Bin(data),
            ]));
            let doc = encode(&Value::Map(vec![(
                Value::Str("x".into()),
                Value::Ext(EXT_NPSCALAR, inner),
            )]));
            let arrays = read_flax_arrays_bytes(&doc).unwrap();
            assert_eq!(arrays[0].2, expected, "dtype {dtype}");
        }
    }

    #[test]
    fn corrupt_leaves_error() {
        // Data length mismatch.
        let inner = encode(&Value::Array(vec![
            Value::Array(vec![Value::UInt(3)]),
            Value::Str("float32".into()),
            Value::Bin(vec![0; 4]),
        ]));
        let doc = encode(&Value::Map(vec![(
            Value::Str("w".into()),
            Value::Ext(1, inner),
        )]));
        let e = read_flax_arrays_bytes(&doc).err().unwrap();
        assert!(e.message().contains("needs 12 bytes"));
        // Complex ext type rejected.
        let doc = encode(&Value::Map(vec![(
            Value::Str("c".into()),
            Value::Ext(EXT_NATIVE_COMPLEX, vec![0; 16]),
        )]));
        assert!(read_flax_arrays_bytes(&doc).is_err());
        // Root must be a map.
        assert!(read_flax_arrays_bytes(&encode(&Value::UInt(3))).is_err());
    }

    #[test]
    fn conflicting_paths_error() {
        let arrays = vec![
            ("a".to_string(), vec![1], vec![1.0]),
            ("a/b".to_string(), vec![1], vec![2.0]),
        ];
        let e = write_flax_arrays_bytes(&arrays).err().unwrap();
        assert!(e.message().contains("prefix already used"));
        let dup = vec![
            ("a".to_string(), vec![1], vec![1.0]),
            ("a".to_string(), vec![1], vec![2.0]),
        ];
        assert!(write_flax_arrays_bytes(&dup).is_err());
    }
}
