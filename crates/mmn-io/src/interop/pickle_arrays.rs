//! Numpy arrays inside arbitrary pickle files.
//!
//! CPython pickles numpy arrays as
//! `_reconstruct(ndarray, (0,), b'b')` + `BUILD((version, shape, dtype,
//! fortran_order, data))`, with the dtype itself a
//! `dtype(kind, 0, 1)` + `BUILD((3, byteorder, ...))` pair, and numpy
//! scalars as `scalar(dtype, bytes)`. This module walks any pickle value
//! tree (dicts/lists/tuples) collecting those leaves under dotted path
//! names — which covers PaddlePaddle `.pdparams`, scikit-learn model
//! pickles, and plain pickled ndarray dicts — and writes pickles numpy
//! deserializes, all without numpy installed.

use super::npy::{decode_element, descr_item_size};
use super::pickle::{parse_pickle, PickleValue, PickleWriter};
use super::NamedArray;
use mmn_core::MmnError;
use std::fs;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// Decoded array payload: `(shape, f32 values)`.
type ShapedValues = (Vec<usize>, Vec<f32>);

fn is_numpy_module(module: &str) -> bool {
    module == "numpy"
        || module.starts_with("numpy.core")
        || module.starts_with("numpy._core")
}

/// Decode a pickled `numpy.dtype` into an npy descr string like `"<f4"`.
fn dtype_descr(value: &PickleValue) -> Option<String> {
    let (reduce, state) = match value {
        PickleValue::Build(reduce, state) => (reduce.as_ref(), Some(state.as_ref())),
        other => (other, None),
    };
    let PickleValue::Reduce(callable, args) = reduce else {
        return None;
    };
    let PickleValue::Global(module, name) = callable.as_ref() else {
        return None;
    };
    if !is_numpy_module(module) || name != "dtype" {
        return None;
    }
    let PickleValue::Tuple(args) = args.as_ref() else {
        return None;
    };
    let kind = args.first()?.as_str()?;
    // dtype.__setstate__ state: (version, byteorder, subdtype, names, ...).
    let byteorder = match state {
        Some(PickleValue::Tuple(items)) => items
            .get(1)
            .and_then(|v| v.as_str())
            .unwrap_or("<")
            .to_string(),
        _ => "<".to_string(),
    };
    // "=" (native) and "|" (not applicable) both read as little-endian —
    // every producer this crate targets writes LE data.
    let prefix = match byteorder.as_str() {
        ">" => ">",
        _ if kind.len() == 2 && kind.starts_with('b') => "|",
        "|" => "|",
        _ => "<",
    };
    Some(format!("{prefix}{kind}"))
}

/// Raw byte payload of a pickled value: direct bytes (protocol 3+), or the
/// protocol-2 encoding `_codecs.encode(latin1_string, 'latin1')`.
fn raw_bytes(value: &PickleValue) -> Option<Vec<u8>> {
    match value {
        PickleValue::Bytes(data) => Some(data.clone()),
        PickleValue::Reduce(callable, args) => {
            let PickleValue::Global(module, name) = callable.as_ref() else {
                return None;
            };
            if module != "_codecs" || name != "encode" {
                return None;
            }
            let PickleValue::Tuple(args) = args.as_ref() else {
                return None;
            };
            let text = args.first()?.as_str()?;
            match args.get(1).and_then(|v| v.as_str()) {
                Some("latin1") | Some("latin-1") | None => {}
                _ => return None,
            }
            // latin-1: each char is one byte (code points 0-255).
            let mut bytes = Vec::with_capacity(text.len());
            for ch in text.chars() {
                let code = ch as u32;
                if code > 0xFF {
                    return None;
                }
                bytes.push(code as u8);
            }
            Some(bytes)
        }
        _ => None,
    }
}

/// Convert Fortran-ordered values to C order for `shape`.
fn fortran_to_c(values: &[f32], shape: &[usize]) -> Vec<f32> {
    if shape.len() < 2 {
        return values.to_vec();
    }
    // F strides: stride[i] = product of shape[..i].
    let mut f_strides = vec![1usize; shape.len()];
    for i in 1..shape.len() {
        f_strides[i] = f_strides[i - 1] * shape[i - 1];
    }
    let numel: usize = shape.iter().product();
    let mut out = Vec::with_capacity(numel);
    let mut coords = vec![0usize; shape.len()];
    for _ in 0..numel {
        let f_index: usize = coords
            .iter()
            .zip(&f_strides)
            .map(|(&c, &s)| c * s)
            .sum();
        out.push(values[f_index]);
        // Increment C-order coordinates (last dim fastest).
        for d in (0..shape.len()).rev() {
            coords[d] += 1;
            if coords[d] < shape[d] {
                break;
            }
            coords[d] = 0;
        }
    }
    out
}

/// Decode one `_reconstruct(ndarray, ...)` + BUILD state leaf.
fn ndarray_from_build(
    reduce: &PickleValue,
    state: &PickleValue,
) -> Result<Option<ShapedValues>, MmnError> {
    let PickleValue::Reduce(callable, _) = reduce else {
        return Ok(None);
    };
    let PickleValue::Global(module, name) = callable.as_ref() else {
        return Ok(None);
    };
    if !is_numpy_module(module) || name != "_reconstruct" {
        return Ok(None);
    }
    let PickleValue::Tuple(state) = state else {
        return Err(err("numpy ndarray BUILD state is not a tuple"));
    };
    // (version, shape, dtype, fortran_order, data)
    let [_, shape_value, dtype_value, fortran_value, data_value] = state.as_slice() else {
        return Err(err(format!(
            "numpy ndarray state has {} items, expected 5",
            state.len()
        )));
    };
    let PickleValue::Tuple(dims) = shape_value else {
        return Err(err("numpy ndarray state shape is not a tuple"));
    };
    let shape: Vec<usize> = dims
        .iter()
        .map(|d| match d {
            PickleValue::Int(n) if *n >= 0 => Ok(*n as usize),
            _ => Err(err("numpy ndarray shape dim is not a non-negative int")),
        })
        .collect::<Result<_, _>>()?;
    let descr = dtype_descr(dtype_value)
        .ok_or_else(|| err("numpy ndarray state carries an unsupported dtype"))?;
    // Object arrays pickle their data as a list — not numeric tensors.
    let data = raw_bytes(data_value).ok_or_else(|| {
        err("numpy ndarray data is not raw bytes (object arrays are not supported)")
    })?;
    let item = descr_item_size(&descr)?;
    let numel: usize = shape.iter().product();
    if data.len() != numel * item {
        return Err(err(format!(
            "numpy ndarray shape {shape:?} needs {} bytes of {descr}, got {}",
            numel * item,
            data.len()
        )));
    }
    let values: Vec<f32> = data
        .chunks_exact(item)
        .map(|chunk| decode_element(&descr, chunk))
        .collect::<Result<_, _>>()?;
    let fortran = matches!(fortran_value, PickleValue::Bool(true));
    let values = if fortran {
        fortran_to_c(&values, &shape)
    } else {
        values
    };
    Ok(Some((shape, values)))
}

/// Decode a protocol-5 `_frombuffer(data, dtype, shape, order)` reduce leaf.
fn ndarray_from_frombuffer(
    value: &PickleValue,
) -> Result<Option<ShapedValues>, MmnError> {
    let PickleValue::Reduce(callable, args) = value else {
        return Ok(None);
    };
    let PickleValue::Global(module, name) = callable.as_ref() else {
        return Ok(None);
    };
    if !is_numpy_module(module) || name != "_frombuffer" {
        return Ok(None);
    }
    let PickleValue::Tuple(args) = args.as_ref() else {
        return Ok(None);
    };
    let [data_value, dtype_value, shape_value, order_value] = args.as_slice() else {
        return Err(err(format!(
            "numpy _frombuffer has {} args, expected 4",
            args.len()
        )));
    };
    let data = raw_bytes(data_value)
        .ok_or_else(|| err("numpy _frombuffer data is not raw bytes"))?;
    let descr = dtype_descr(dtype_value)
        .ok_or_else(|| err("numpy _frombuffer carries an unsupported dtype"))?;
    let PickleValue::Tuple(dims) = shape_value else {
        return Err(err("numpy _frombuffer shape is not a tuple"));
    };
    let shape: Vec<usize> = dims
        .iter()
        .map(|d| match d {
            PickleValue::Int(n) if *n >= 0 => Ok(*n as usize),
            _ => Err(err("numpy _frombuffer shape dim invalid")),
        })
        .collect::<Result<_, _>>()?;
    let item = descr_item_size(&descr)?;
    let numel: usize = shape.iter().product();
    if data.len() != numel * item {
        return Err(err(format!(
            "numpy _frombuffer shape {shape:?} needs {} bytes of {descr}, got {}",
            numel * item,
            data.len()
        )));
    }
    let values: Vec<f32> = data
        .chunks_exact(item)
        .map(|chunk| decode_element(&descr, chunk))
        .collect::<Result<_, _>>()?;
    let fortran = order_value.as_str() == Some("F");
    let values = if fortran {
        fortran_to_c(&values, &shape)
    } else {
        values
    };
    Ok(Some((shape, values)))
}

/// Decode a `numpy scalar(dtype, bytes)` reduce leaf.
fn scalar_from_reduce(value: &PickleValue) -> Result<Option<f32>, MmnError> {
    let PickleValue::Reduce(callable, args) = value else {
        return Ok(None);
    };
    let PickleValue::Global(module, name) = callable.as_ref() else {
        return Ok(None);
    };
    if !is_numpy_module(module) || name != "scalar" {
        return Ok(None);
    }
    let PickleValue::Tuple(args) = args.as_ref() else {
        return Ok(None);
    };
    let [dtype_value, data_value] = args.as_slice() else {
        return Ok(None);
    };
    let Some(data) = raw_bytes(data_value) else {
        return Ok(None);
    };
    let descr =
        dtype_descr(dtype_value).ok_or_else(|| err("numpy scalar has unsupported dtype"))?;
    if data.len() != descr_item_size(&descr)? {
        return Err(err("numpy scalar byte length mismatch"));
    }
    Ok(Some(decode_element(&descr, &data)?))
}

fn walk(value: &PickleValue, prefix: &str, out: &mut Vec<NamedArray>) -> Result<(), MmnError> {
    let child_name = |key: &str| {
        if prefix.is_empty() {
            key.to_string()
        } else {
            format!("{prefix}.{key}")
        }
    };
    match value {
        PickleValue::Dict(pairs) => {
            for (key, child) in pairs {
                let key = match key {
                    PickleValue::Str(s) => s.clone(),
                    PickleValue::Int(n) => n.to_string(),
                    _ => continue,
                };
                walk(child, &child_name(&key), out)?;
            }
        }
        PickleValue::List(items) | PickleValue::Tuple(items) => {
            for (i, child) in items.iter().enumerate() {
                walk(child, &child_name(&i.to_string()), out)?;
            }
        }
        PickleValue::Build(reduce, state) => {
            if let Some((shape, values)) = ndarray_from_build(reduce, state)? {
                out.push((prefix.to_string(), shape, values));
            } else {
                // sklearn-style objects: state is usually a dict of fields.
                walk(state, prefix, out)?;
            }
        }
        reduce @ PickleValue::Reduce(..) => {
            if let Some((shape, values)) = ndarray_from_frombuffer(reduce)? {
                out.push((prefix.to_string(), shape, values));
            } else if let Some(v) = scalar_from_reduce(reduce)? {
                out.push((prefix.to_string(), vec![], vec![v]));
            }
        }
        _ => {}
    }
    Ok(())
}

/// Collect every numpy array/scalar in a pickle byte stream, named by their
/// dotted path in the pickled object graph.
pub fn read_pickle_arrays_bytes(bytes: &[u8]) -> Result<Vec<NamedArray>, MmnError> {
    let root = parse_pickle(bytes)?;
    let mut out = Vec::new();
    walk(&root, "", &mut out)?;
    if out.is_empty() {
        return Err(err(
            "pickle contains no numpy arrays (expected a pickled dict/list of ndarrays)",
        ));
    }
    Ok(out)
}

/// Collect every numpy array in a pickle file on disk.
pub fn read_pickle_arrays(path: &str) -> Result<Vec<NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read {path}: {e}")))?;
    read_pickle_arrays_bytes(&bytes)
}

/// Emit one numpy F32 ndarray in the `_reconstruct` + BUILD encoding.
fn write_ndarray(w: &mut PickleWriter, shape: &[usize], values: &[f32]) {
    w.global("numpy.core.multiarray", "_reconstruct");
    w.mark();
    w.global("numpy", "ndarray");
    w.mark();
    w.int(0);
    w.tuple_from_mark();
    w.bytes(b"b");
    w.tuple_from_mark();
    w.reduce();
    // BUILD state: (1, shape, dtype, False, data).
    w.mark();
    w.int(1);
    w.mark();
    for &d in shape {
        w.int(d as i64);
    }
    w.tuple_from_mark();
    // dtype('f4') + __setstate__((3, '<', None, None, None, -1, -1, 0)).
    w.global("numpy", "dtype");
    w.mark();
    w.string("f4");
    w.bool(false);
    w.bool(true);
    w.tuple_from_mark();
    w.reduce();
    w.mark();
    w.int(3);
    w.string("<");
    w.none();
    w.none();
    w.none();
    w.int(-1);
    w.int(-1);
    w.int(0);
    w.tuple_from_mark();
    w.build();
    w.bool(false);
    let data: Vec<u8> = values.iter().flat_map(|v| v.to_le_bytes()).collect();
    w.bytes(&data);
    w.tuple_from_mark();
    w.build();
}

/// Serialize named arrays as a pickled `{name: ndarray}` dict that
/// `pickle.load` (with numpy installed) deserializes.
pub fn write_pickle_arrays_bytes(arrays: &[NamedArray]) -> Result<Vec<u8>, MmnError> {
    let mut w = PickleWriter::new();
    w.empty_dict();
    w.mark();
    for (name, shape, values) in arrays {
        let numel: usize = shape.iter().product();
        if numel != values.len() {
            return Err(err(format!(
                "array {name}: shape {shape:?} needs {numel} values, got {}",
                values.len()
            )));
        }
        w.string(name);
        write_ndarray(&mut w, shape, values);
    }
    w.set_items();
    Ok(w.finish())
}

/// Write named arrays as a numpy-deserializable pickle file.
pub fn write_pickle_arrays(path: &str, arrays: &[NamedArray]) -> Result<(), MmnError> {
    let bytes = write_pickle_arrays_bytes(arrays)?;
    crate::checkpoint_util::write_file_create_parents(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn own_writer_roundtrips_through_reader() {
        let arrays = vec![
            ("layer.w".to_string(), vec![2, 3], vec![1.0, -2.0, 3.5, 0.0, 4.25, -0.5]),
            ("layer.b".to_string(), vec![2], vec![0.125, -9.0]),
        ];
        let bytes = write_pickle_arrays_bytes(&arrays).unwrap();
        let back = read_pickle_arrays_bytes(&bytes).unwrap();
        assert_eq!(back, arrays);
    }

    #[test]
    fn nested_containers_get_dotted_names() {
        let inner = vec![("model.0.weight".to_string(), vec![1], vec![7.0])];
        let bytes = write_pickle_arrays_bytes(&inner).unwrap();
        let back = read_pickle_arrays_bytes(&bytes).unwrap();
        assert_eq!(back[0].0, "model.0.weight");
    }

    #[test]
    fn fortran_order_converts_to_c() {
        // 2x3 F-order data [c00,c10,c01,c11,c02,c12] -> C [c00,c01,c02,...].
        let f_values = vec![0.0, 3.0, 1.0, 4.0, 2.0, 5.0];
        let c = fortran_to_c(&f_values, &[2, 3]);
        assert_eq!(c, vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0]);
    }

    #[test]
    fn pickle_without_arrays_errors() {
        let mut w = PickleWriter::new();
        w.empty_dict();
        w.mark();
        w.string("config");
        w.string("just text");
        w.set_items();
        let bytes = w.finish();
        let e = read_pickle_arrays_bytes(&bytes).err().unwrap();
        assert!(e.message().contains("no numpy arrays"));
    }

    #[test]
    fn length_mismatch_rejected() {
        let arrays = vec![("x".to_string(), vec![3], vec![1.0])];
        assert!(write_pickle_arrays_bytes(&arrays).is_err());
    }
}
