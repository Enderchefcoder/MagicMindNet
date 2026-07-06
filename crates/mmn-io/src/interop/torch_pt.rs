//! PyTorch `.pt` / `.pth` checkpoint interchange, from scratch.
//!
//! `torch.save` produces a ZIP archive with a pickled object graph
//! (`data.pkl`) plus one raw storage blob per tensor. This module reads that
//! layout with the from-scratch [`super::zip`] and [`super::pickle`] modules
//! and writes archives `torch.load` (including `weights_only=True`) accepts.

use super::pickle::{parse_pickle, parse_pickle_prefix, PickleValue, PickleWriter};
use super::zip::{is_zip_bytes, read_zip, write_zip_stored, ZipEntry};
use crate::checkpoint_util::write_file_create_parents;
use crate::hf_safetensors::{
    chatbot_from_external_tensors, chatbot_meta_json, collect_named_tensors, hf_name_to_mmn,
};
use half::{bf16, f16};
use mmn_core::{MmnError, Tensor};
use mmn_models::Chatbot;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs;

const META_KEY: &str = "_mmn_meta";
const ARCHIVE_ROOT: &str = "archive";

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum StorageDtype {
    F32,
    F16,
    BF16,
    F64,
    I64,
    I32,
    I16,
    I8,
    U8,
    Bool,
}

impl StorageDtype {
    fn from_class(name: &str) -> Result<Self, MmnError> {
        Ok(match name {
            "FloatStorage" => StorageDtype::F32,
            "HalfStorage" => StorageDtype::F16,
            "BFloat16Storage" => StorageDtype::BF16,
            "DoubleStorage" => StorageDtype::F64,
            "LongStorage" => StorageDtype::I64,
            "IntStorage" => StorageDtype::I32,
            "ShortStorage" => StorageDtype::I16,
            "CharStorage" => StorageDtype::I8,
            "ByteStorage" => StorageDtype::U8,
            "BoolStorage" => StorageDtype::Bool,
            other => {
                return Err(err(format!(
                    "torch storage class {other} not supported"
                )));
            }
        })
    }

    fn item_size(&self) -> usize {
        match self {
            StorageDtype::F64 | StorageDtype::I64 => 8,
            StorageDtype::F32 | StorageDtype::I32 => 4,
            StorageDtype::F16 | StorageDtype::BF16 | StorageDtype::I16 => 2,
            StorageDtype::I8 | StorageDtype::U8 | StorageDtype::Bool => 1,
        }
    }

    fn decode(&self, chunk: &[u8]) -> f32 {
        match self {
            StorageDtype::F32 => f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]),
            StorageDtype::F16 => f16::from_le_bytes([chunk[0], chunk[1]]).to_f32(),
            StorageDtype::BF16 => bf16::from_le_bytes([chunk[0], chunk[1]]).to_f32(),
            StorageDtype::F64 => f64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            StorageDtype::I64 => i64::from_le_bytes([
                chunk[0], chunk[1], chunk[2], chunk[3], chunk[4], chunk[5], chunk[6], chunk[7],
            ]) as f32,
            StorageDtype::I32 => {
                i32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as f32
            }
            StorageDtype::I16 => i16::from_le_bytes([chunk[0], chunk[1]]) as f32,
            StorageDtype::I8 => (chunk[0] as i8) as f32,
            StorageDtype::U8 => chunk[0] as f32,
            StorageDtype::Bool => {
                if chunk[0] != 0 {
                    1.0
                } else {
                    0.0
                }
            }
        }
    }
}

struct TensorStub {
    storage_key: String,
    dtype: StorageDtype,
    storage_offset: usize,
    shape: Vec<usize>,
    stride: Vec<usize>,
}

fn ints_from_tuple(value: &PickleValue, what: &str) -> Result<Vec<usize>, MmnError> {
    let items = value
        .tuple_items()
        .ok_or_else(|| err(format!("torch tensor {what} is not a tuple")))?;
    items
        .iter()
        .map(|v| {
            v.as_int()
                .filter(|&n| n >= 0)
                .map(|n| n as usize)
                .ok_or_else(|| err(format!("torch tensor {what} has a non-int entry")))
        })
        .collect()
}

fn tensor_stub_from_reduce(value: &PickleValue) -> Result<Option<TensorStub>, MmnError> {
    // Torch may BUILD hook/metadata state onto a rebuilt tensor; the reduce
    // call underneath still describes the tensor.
    let value = match value {
        PickleValue::Build(inner, _) => inner.as_ref(),
        other => other,
    };
    let PickleValue::Reduce(callable, args) = value else {
        return Ok(None);
    };
    let PickleValue::Global(module, name) = callable.as_ref() else {
        return Ok(None);
    };
    if module != "torch._utils" || !(name == "_rebuild_tensor_v2" || name == "_rebuild_tensor") {
        return Ok(None);
    }
    let items = args
        .tuple_items()
        .ok_or_else(|| err("_rebuild_tensor args not a tuple"))?;
    if items.len() < 4 {
        return Err(err("_rebuild_tensor args too short"));
    }
    let PickleValue::PersId(pid) = &items[0] else {
        return Err(err("_rebuild_tensor first arg is not a persistent id"));
    };
    let pid_items = pid
        .tuple_items()
        .ok_or_else(|| err("torch persistent id is not a tuple"))?;
    if pid_items.len() < 5 || pid_items[0].as_str() != Some("storage") {
        return Err(err("torch persistent id is not a storage descriptor"));
    }
    let PickleValue::Global(_, storage_class) = &pid_items[1] else {
        return Err(err("torch storage descriptor missing storage class"));
    };
    let storage_key = pid_items[2]
        .as_str()
        .ok_or_else(|| err("torch storage key is not a string"))?
        .to_string();
    let dtype = StorageDtype::from_class(storage_class)?;
    let storage_offset = items[1]
        .as_int()
        .filter(|&n| n >= 0)
        .ok_or_else(|| err("torch tensor offset invalid"))? as usize;
    let shape = ints_from_tuple(&items[2], "size")?;
    let stride = ints_from_tuple(&items[3], "stride")?;
    Ok(Some(TensorStub {
        storage_key,
        dtype,
        storage_offset,
        shape,
        stride,
    }))
}

fn c_contiguous_stride(shape: &[usize]) -> Vec<usize> {
    let mut stride = vec![1usize; shape.len()];
    for i in (0..shape.len().saturating_sub(1)).rev() {
        stride[i] = stride[i + 1] * shape[i + 1];
    }
    stride
}

/// Materialize a strided view over decoded storage values into C order.
fn gather_strided(
    values: &[f32],
    offset: usize,
    shape: &[usize],
    stride: &[usize],
) -> Result<Vec<f32>, MmnError> {
    let numel: usize = shape.iter().product();
    if shape.is_empty() {
        return values
            .get(offset)
            .map(|&v| vec![v])
            .ok_or_else(|| err("torch scalar offset out of bounds"));
    }
    if stride == c_contiguous_stride(shape) {
        return values
            .get(offset..offset + numel)
            .map(|s| s.to_vec())
            .ok_or_else(|| err("torch tensor data out of storage bounds"));
    }
    let mut out = Vec::with_capacity(numel);
    let mut index = vec![0usize; shape.len()];
    for _ in 0..numel {
        let pos = offset
            + index
                .iter()
                .zip(stride)
                .map(|(i, s)| i * s)
                .sum::<usize>();
        out.push(
            *values
                .get(pos)
                .ok_or_else(|| err("torch strided read out of storage bounds"))?,
        );
        for axis in (0..shape.len()).rev() {
            index[axis] += 1;
            if index[axis] < shape[axis] {
                break;
            }
            index[axis] = 0;
        }
    }
    Ok(out)
}

fn find_pickle_entry(entries: &[ZipEntry]) -> Result<(&ZipEntry, String), MmnError> {
    // Eager checkpoints use data.pkl; TorchScript (torch.jit.save) archives
    // store their tensors in constants.pkl with the same data/ storage dir.
    for pickle_name in ["data.pkl", "constants.pkl"] {
        for entry in entries {
            if entry.name == pickle_name {
                return Ok((entry, String::new()));
            }
            if let Some(prefix) = entry.name.strip_suffix(pickle_name) {
                if prefix.ends_with('/') {
                    return Ok((entry, prefix.to_string()));
                }
            }
        }
    }
    Err(err(
        "not a torch checkpoint: archive has no data.pkl or constants.pkl entry",
    ))
}

/// Little-endian bytes of torch's legacy magic number 0x1950a86a20f9469cfc6c.
const LEGACY_MAGIC_LE: [u8; 10] = [0x6c, 0xfc, 0x9c, 0x46, 0xf9, 0x20, 0x6a, 0xa8, 0x50, 0x19];

/// True when `bytes` are a legacy (pre-1.6, non-zip) torch pickle file.
///
/// Matches the exact leading pattern: PROTO opcode, then a `LONG1` holding
/// torch's 10-byte magic number (possibly behind a protocol-4 FRAME).
pub fn is_legacy_torch_bytes(bytes: &[u8]) -> bool {
    if bytes.len() < 16 || bytes[0] != 0x80 || bytes[1] > 5 {
        return false;
    }
    let window = &bytes[..bytes.len().min(32)];
    window.windows(12).any(|w| {
        w[0] == 0x8a && w[1] == 10 && w[2..12] == LEGACY_MAGIC_LE
    })
}

fn check_legacy_magic(value: &PickleValue) -> Result<(), MmnError> {
    match value {
        PickleValue::Bytes(raw) if raw.as_slice() == LEGACY_MAGIC_LE => Ok(()),
        _ => Err(err(
            "not a legacy torch checkpoint: magic number mismatch in first pickle",
        )),
    }
}

type NamedStubs = Vec<(String, TensorStub)>;

fn stubs_from_state_dict(
    pairs: &[(PickleValue, PickleValue)],
) -> Result<(NamedStubs, Option<serde_json::Value>), MmnError> {
    let mut stubs = Vec::new();
    let mut meta = None;
    for (key, value) in pairs {
        let Some(name) = key.as_str() else { continue };
        if name == META_KEY {
            if let Some(json) = value.as_str() {
                meta = Some(serde_json::from_str(json).map_err(|e| {
                    err(format!("torch checkpoint {META_KEY} JSON invalid: {e}"))
                })?);
            }
            continue;
        }
        if let Some(stub) = tensor_stub_from_reduce(value)? {
            stubs.push((name.to_string(), stub));
        }
    }
    Ok((stubs, meta))
}

fn array_from_stub(
    name: &str,
    stub: &TensorStub,
    storages: &HashMap<&str, &[u8]>,
) -> Result<Vec<f32>, MmnError> {
    let raw = storages.get(stub.storage_key.as_str()).ok_or_else(|| {
        err(format!(
            "torch checkpoint missing storage blob {} for tensor {name}",
            stub.storage_key
        ))
    })?;
    let item = stub.dtype.item_size();
    if !raw.len().is_multiple_of(item) {
        return Err(err(format!(
            "torch storage {} has {} bytes, not a multiple of item size {item}",
            stub.storage_key,
            raw.len()
        )));
    }
    let values: Vec<f32> = raw.chunks_exact(item).map(|c| stub.dtype.decode(c)).collect();
    gather_strided(&values, stub.storage_offset, &stub.shape, &stub.stride)
}

/// Decode all tensors, fanning storage decode out across available cores.
fn arrays_from_stubs(
    stubs: NamedStubs,
    storages: &HashMap<&str, &[u8]>,
) -> Result<Vec<super::NamedArray>, MmnError> {
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(stubs.len().max(1));
    if workers <= 1 {
        let mut arrays = Vec::with_capacity(stubs.len());
        for (name, stub) in stubs {
            let data = array_from_stub(&name, &stub, storages)?;
            arrays.push((name, stub.shape, data));
        }
        return Ok(arrays);
    }
    let chunk_size = stubs.len().div_ceil(workers);
    let results: Vec<Result<Vec<super::NamedArray>, MmnError>> = std::thread::scope(|scope| {
        let handles: Vec<_> = stubs
            .chunks(chunk_size)
            .map(|chunk| {
                scope.spawn(move || {
                    chunk
                        .iter()
                        .map(|(name, stub)| {
                            array_from_stub(name, stub, storages)
                                .map(|data| (name.clone(), stub.shape.clone(), data))
                        })
                        .collect()
                })
            })
            .collect();
        handles
            .into_iter()
            .map(|h| h.join().expect("torch decode worker panicked"))
            .collect()
    });
    let mut arrays = Vec::with_capacity(stubs.len());
    for chunk in results {
        arrays.extend(chunk?);
    }
    Ok(arrays)
}

/// Read a legacy (pre-1.6) `torch.save` file: magic + protocol + sys_info
/// pickles, the object pickle, a storage-key list, then raw storage payloads.
pub fn read_legacy_torch_arrays_bytes(
    bytes: &[u8],
) -> Result<(Vec<super::NamedArray>, Option<serde_json::Value>), MmnError> {
    let (magic, used) = parse_pickle_prefix(bytes)?;
    check_legacy_magic(&magic)?;
    let mut pos = used;
    let (_protocol, used) = parse_pickle_prefix(&bytes[pos..])?;
    pos += used;
    let (_sys_info, used) = parse_pickle_prefix(&bytes[pos..])?;
    pos += used;
    let (root, used) = parse_pickle_prefix(&bytes[pos..])?;
    pos += used;
    let (keys_value, used) = parse_pickle_prefix(&bytes[pos..])?;
    pos += used;
    let PickleValue::Dict(pairs) = root else {
        return Err(err(
            "legacy torch checkpoint does not contain a state dict (torch.save(model.state_dict(), path))",
        ));
    };
    let PickleValue::List(key_items) = keys_value else {
        return Err(err("legacy torch checkpoint missing storage key list"));
    };
    let (stubs, meta) = stubs_from_state_dict(&pairs)?;
    let dtype_by_key: HashMap<&str, StorageDtype> = stubs
        .iter()
        .map(|(_, stub)| (stub.storage_key.as_str(), stub.dtype))
        .collect();
    // Storage payloads follow in key-list order: i64 element count + raw data.
    let mut storage_data: HashMap<String, Vec<u8>> = HashMap::new();
    for key_value in &key_items {
        let key = key_value
            .as_str()
            .ok_or_else(|| err("legacy torch storage key is not a string"))?;
        let dtype = dtype_by_key.get(key).copied().ok_or_else(|| {
            err(format!(
                "legacy torch storage key {key} not referenced by any tensor"
            ))
        })?;
        let header = bytes
            .get(pos..pos + 8)
            .ok_or_else(|| err("legacy torch storage header truncated"))?;
        let numel = i64::from_le_bytes([
            header[0], header[1], header[2], header[3], header[4], header[5], header[6],
            header[7],
        ]);
        if numel < 0 {
            return Err(err(format!("legacy torch storage {key} has negative size")));
        }
        pos += 8;
        let len = numel as usize * dtype.item_size();
        let raw = bytes
            .get(pos..pos + len)
            .ok_or_else(|| err(format!("legacy torch storage {key} data truncated")))?;
        pos += len;
        storage_data.insert(key.to_string(), raw.to_vec());
    }
    let storages: HashMap<&str, &[u8]> = storage_data
        .iter()
        .map(|(k, v)| (k.as_str(), v.as_slice()))
        .collect();
    Ok((arrays_from_stubs(stubs, &storages)?, meta))
}

/// Every named array in a `.pt` state dict, decoded to `(shape, f32 data)`.
///
/// Accepts both the zip-based format (torch >= 1.6) and the legacy pickle
/// stream format, auto-detected from the leading bytes.
pub fn read_torch_arrays_bytes(
    bytes: &[u8],
) -> Result<(Vec<super::NamedArray>, Option<serde_json::Value>), MmnError> {
    if !is_zip_bytes(bytes) && is_legacy_torch_bytes(bytes) {
        return read_legacy_torch_arrays_bytes(bytes);
    }
    let entries = read_zip(bytes)?;
    let (pickle_entry, prefix) = find_pickle_entry(&entries)?;
    let root = parse_pickle(&pickle_entry.data)?;
    let (stubs, meta) = match root {
        PickleValue::Dict(pairs) => stubs_from_state_dict(&pairs)?,
        // TorchScript constants.pkl holds a tuple/list of tensors.
        PickleValue::Tuple(items) | PickleValue::List(items) => {
            let mut stubs = Vec::new();
            for (i, item) in items.iter().enumerate() {
                if let Some(stub) = tensor_stub_from_reduce(item)? {
                    stubs.push((format!("constants.{i}"), stub));
                }
            }
            (stubs, None)
        }
        _ => {
            return Err(err(
                "torch checkpoint does not contain a state dict (torch.save(model.state_dict(), path))",
            ));
        }
    };
    let storages: HashMap<&str, &[u8]> = entries
        .iter()
        .filter_map(|e| {
            e.name
                .strip_prefix(&prefix)
                .and_then(|rest| rest.strip_prefix("data/"))
                .map(|key| (key, e.data.as_slice()))
        })
        .collect();
    Ok((arrays_from_stubs(stubs, &storages)?, meta))
}

/// Read every tensor in a `.pt` file (generic, model-agnostic).
pub fn read_torch_arrays(path: &str) -> Result<Vec<super::NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read torch file {path}: {e}")))?;
    Ok(read_torch_arrays_bytes(&bytes)?.0)
}

/// Import a PyTorch state-dict checkpoint into a `Chatbot`.
pub fn import_torch_pt_bytes(bytes: &[u8]) -> Result<Chatbot, MmnError> {
    let (arrays, meta) = read_torch_arrays_bytes(bytes)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for (name, shape, data) in arrays {
        let Some(mmn_key) = hf_name_to_mmn(&name) else {
            continue;
        };
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| err(format!("torch tensor {name}: {e}")))?;
        tensors.insert(mmn_key, Tensor::from_array(arr, true));
    }
    if tensors.is_empty() {
        return Err(err(
            "torch checkpoint contains no recognizable model tensors (expected keys like embed, model.embed_tokens.weight, blocks.0.attn.q)",
        ));
    }
    chatbot_from_external_tensors(tensors, meta.unwrap_or_else(|| serde_json::json!({})))
}

/// Import a PyTorch checkpoint from disk.
pub fn import_torch_pt(path: &str) -> Result<Chatbot, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read torch file {path}: {e}")))?;
    import_torch_pt_bytes(&bytes)
}

fn write_tensor_reduce(w: &mut PickleWriter, storage_key: &str, shape: &[usize]) {
    let numel: usize = shape.iter().product();
    w.global("torch._utils", "_rebuild_tensor_v2");
    w.mark();
    // Persistent storage id: ('storage', FloatStorage, key, 'cpu', numel).
    w.mark();
    w.string("storage");
    w.global("torch", "FloatStorage");
    w.string(storage_key);
    w.string("cpu");
    w.int(numel as i64);
    w.tuple_from_mark();
    w.binpersid();
    w.int(0); // storage offset
    w.mark();
    for &d in shape {
        w.int(d as i64);
    }
    w.tuple_from_mark();
    w.mark();
    for &s in &c_contiguous_stride(shape) {
        w.int(s as i64);
    }
    w.tuple_from_mark();
    w.bool(false); // requires_grad
    w.global("collections", "OrderedDict");
    w.empty_tuple();
    w.reduce(); // backward hooks
    w.tuple_from_mark();
    w.reduce();
}

/// Serialize named arrays (plus optional meta JSON) as a torch-loadable `.pt`.
pub fn write_torch_arrays(
    arrays: &[super::NamedArray],
    meta: Option<&serde_json::Value>,
) -> Result<Vec<u8>, MmnError> {
    let mut w = PickleWriter::new();
    w.empty_dict();
    w.mark();
    let mut storage_entries: Vec<(String, Vec<u8>)> = Vec::new();
    for (i, (name, shape, data)) in arrays.iter().enumerate() {
        let numel: usize = shape.iter().product();
        if numel != data.len() {
            return Err(err(format!(
                "torch export tensor {name}: shape {:?} needs {numel} values, got {}",
                shape,
                data.len()
            )));
        }
        let storage_key = i.to_string();
        w.string(name);
        write_tensor_reduce(&mut w, &storage_key, shape);
        let bytes: Vec<u8> = data.iter().flat_map(|v| v.to_le_bytes()).collect();
        storage_entries.push((format!("{ARCHIVE_ROOT}/data/{storage_key}"), bytes));
    }
    if let Some(meta) = meta {
        w.string(META_KEY);
        w.string(&meta.to_string());
    }
    w.set_items();
    let pickle_bytes = w.finish();
    let mut entries: Vec<(String, Vec<u8>)> =
        vec![(format!("{ARCHIVE_ROOT}/data.pkl"), pickle_bytes)];
    entries.extend(storage_entries);
    entries.push((format!("{ARCHIVE_ROOT}/version"), b"3\n".to_vec()));
    entries.push((format!("{ARCHIVE_ROOT}/byteorder"), b"little".to_vec()));
    write_zip_stored(&entries)
}

/// Export a `Chatbot` as a PyTorch-loadable `.pt` state dict.
pub fn export_torch_pt(model: &Chatbot, path: &str) -> Result<(), MmnError> {
    let named = collect_named_tensors(model);
    let mut keys: Vec<&String> = named.keys().collect();
    keys.sort();
    let mut arrays: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::with_capacity(named.len());
    for key in keys {
        let arr = named[key].data.as_standard_layout().into_owned();
        arrays.push((
            key.clone(),
            arr.shape().to_vec(),
            arr.iter().copied().collect(),
        ));
    }
    let meta = chatbot_meta_json(model, Default::default());
    let bytes = write_torch_arrays(&arrays, Some(&meta))?;
    write_file_create_parents(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn torch_pt_roundtrip_preserves_weights_and_meta() {
        let model = Chatbot::new_with_seed(false, None, 32, Some(2), Some(16), Some(21));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_pt_rt_{}.pt", std::process::id()));
        export_torch_pt(&model, path.to_str().unwrap()).unwrap();
        let loaded = import_torch_pt(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.shape.vocab_size, 32);
        assert_eq!(loaded.shape.n_layer, 2);
        assert_eq!(loaded.init_seed, Some(21));
        let a = model.lm_head.weight.data[[3, 4]];
        let b = loaded.lm_head.weight.data[[3, 4]];
        assert!((a - b).abs() < 1e-6);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn generic_arrays_roundtrip() {
        let arrays = vec![
            ("w".to_string(), vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]),
            ("b".to_string(), vec![3], vec![-1.0, 0.5, 2.0]),
        ];
        let bytes = write_torch_arrays(&arrays, None).unwrap();
        let (back, meta) = read_torch_arrays_bytes(&bytes).unwrap();
        assert!(meta.is_none());
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].0, "w");
        assert_eq!(back[0].1, vec![2, 3]);
        assert_eq!(back[1].2, vec![-1.0, 0.5, 2.0]);
    }

    #[test]
    fn strided_gather_handles_transposed_views() {
        // 2x3 stored transposed: shape (2,3), stride (1,2) over 6 values.
        let values: Vec<f32> = vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
        let out = gather_strided(&values, 0, &[2, 3], &[1, 2]).unwrap();
        assert_eq!(out, vec![0.0, 2.0, 4.0, 1.0, 3.0, 5.0]);
    }

    #[test]
    fn scalar_and_offset_gather() {
        let values = vec![9.0, 8.0, 7.0];
        assert_eq!(gather_strided(&values, 1, &[], &[]).unwrap(), vec![8.0]);
        assert_eq!(
            gather_strided(&values, 1, &[2], &[1]).unwrap(),
            vec![8.0, 7.0]
        );
        assert!(gather_strided(&values, 2, &[2], &[1]).is_err());
    }

    #[test]
    fn missing_data_pkl_errors() {
        let bytes = write_zip_stored(&[("nope.txt".to_string(), vec![1])]).unwrap();
        let e = import_torch_pt_bytes(&bytes).err().unwrap();
        assert!(e.message().contains("data.pkl"));
    }

    /// TorchScript (`torch.jit.save`) archives: constants.pkl holds a tuple
    /// of tensors, storages live under the same `data/` directory.
    #[test]
    fn torchscript_constants_archive_reads() {
        let mut w = PickleWriter::new();
        w.mark();
        write_tensor_reduce(&mut w, "0", &[2, 2]);
        write_tensor_reduce(&mut w, "1", &[3]);
        w.tuple_from_mark();
        let pickle = w.finish();
        let storage0: Vec<u8> = [1.0f32, 2.0, 3.0, 4.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let storage1: Vec<u8> = [9.0f32, 8.0, 7.0]
            .iter()
            .flat_map(|v| v.to_le_bytes())
            .collect();
        let bytes = write_zip_stored(&[
            ("model/constants.pkl".to_string(), pickle),
            ("model/data/0".to_string(), storage0),
            ("model/data/1".to_string(), storage1),
            ("model/code/model.py".to_string(), b"# script".to_vec()),
        ])
        .unwrap();
        let (arrays, meta) = read_torch_arrays_bytes(&bytes).unwrap();
        assert!(meta.is_none());
        assert_eq!(
            arrays,
            vec![
                ("constants.0".to_string(), vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
                ("constants.1".to_string(), vec![3], vec![9.0, 8.0, 7.0]),
            ]
        );
    }

    #[test]
    fn hf_named_state_dict_imports() {
        // A llama-style external state dict without meta.
        let d = 8usize;
        let vocab = 16usize;
        let fill = |n: usize, v: f32| vec![v; n];
        let arrays = vec![
            ("model.embed_tokens.weight".to_string(), vec![vocab, d], fill(vocab * d, 0.5)),
            ("model.layers.0.self_attn.q_proj.weight".to_string(), vec![d, d], fill(d * d, 0.1)),
            ("model.layers.0.self_attn.k_proj.weight".to_string(), vec![d, d], fill(d * d, 0.1)),
            ("model.layers.0.self_attn.v_proj.weight".to_string(), vec![d, d], fill(d * d, 0.1)),
            ("model.layers.0.self_attn.o_proj.weight".to_string(), vec![d, d], fill(d * d, 0.1)),
            ("model.layers.0.mlp.gate_proj.weight".to_string(), vec![d * 4, d], fill(d * 4 * d, 1.0)),
            ("model.layers.0.mlp.up_proj.weight".to_string(), vec![d * 4, d], fill(d * 4 * d, 2.0)),
            ("model.layers.0.mlp.down_proj.weight".to_string(), vec![d, d * 4], fill(d * 4 * d, 0.3)),
        ];
        let bytes = write_torch_arrays(&arrays, None).unwrap();
        let model = import_torch_pt_bytes(&bytes).unwrap();
        assert_eq!(model.shape.vocab_size, vocab);
        assert_eq!(model.shape.n_layer, 1);
        // SwiGLU gate x up fusion: 1.0 * 2.0 = 2.0.
        assert!((model.blocks[0].ffn.weight.data[[0, 0]] - 2.0).abs() < 1e-6);
    }

    fn legacy_pickle_int(w: &mut PickleWriter, v: i64) {
        w.int(v);
    }

    /// Build a genuine legacy torch stream: 3 header pickles, the state dict,
    /// the storage key list, then `i64 numel + raw f32` payloads.
    fn build_legacy_torch(tensors: &[(&str, Vec<usize>, f32)]) -> Vec<u8> {
        let mut out = Vec::new();
        // Pickle 1: magic number (LONG1 with 10 bytes).
        out.extend_from_slice(&[0x80, 0x02, 0x8a, 10]);
        out.extend_from_slice(&LEGACY_MAGIC_LE);
        out.push(b'.');
        // Pickle 2: protocol version.
        let mut w = PickleWriter::new();
        legacy_pickle_int(&mut w, 1001);
        out.extend_from_slice(&w.finish());
        // Pickle 3: sys info dict.
        let mut w = PickleWriter::new();
        w.empty_dict();
        out.extend_from_slice(&w.finish());
        // Pickle 4: the state dict with legacy persistent ids.
        let mut w = PickleWriter::new();
        w.empty_dict();
        w.mark();
        for (i, (name, shape, _fill)) in tensors.iter().enumerate() {
            w.string(name);
            write_tensor_reduce(&mut w, &i.to_string(), shape);
        }
        w.set_items();
        out.extend_from_slice(&w.finish());
        // Pickle 5: storage key list.
        let mut w = PickleWriter::new();
        w.mark();
        for i in 0..tensors.len() {
            w.string(&i.to_string());
        }
        w.list_from_mark();
        out.extend_from_slice(&w.finish());
        // Raw storages in key order.
        for (_, shape, fill) in tensors {
            let numel: usize = shape.iter().product();
            out.extend_from_slice(&(numel as i64).to_le_bytes());
            for _ in 0..numel {
                out.extend_from_slice(&fill.to_le_bytes());
            }
        }
        out
    }

    #[test]
    fn legacy_torch_format_reads_state_dict() {
        let bytes = build_legacy_torch(&[
            ("w", vec![2, 3], 0.25),
            ("b", vec![3], -1.5),
        ]);
        assert!(is_legacy_torch_bytes(&bytes));
        let (arrays, meta) = read_torch_arrays_bytes(&bytes).unwrap();
        assert!(meta.is_none());
        assert_eq!(arrays.len(), 2);
        assert_eq!(arrays[0].0, "w");
        assert_eq!(arrays[0].1, vec![2, 3]);
        assert!(arrays[0].2.iter().all(|&v| v == 0.25));
        assert_eq!(arrays[1].2, vec![-1.5, -1.5, -1.5]);
    }

    #[test]
    fn legacy_torch_chatbot_imports() {
        let d = 8usize;
        let vocab = 16usize;
        let bytes = build_legacy_torch(&[
            ("embed", vec![vocab, d], 0.5),
            ("lm_head", vec![vocab, d], 0.5),
            ("blocks.0.attn.q", vec![d, d], 0.1),
            ("blocks.0.attn.k", vec![d, d], 0.1),
            ("blocks.0.attn.v", vec![d, d], 0.1),
            ("blocks.0.attn.out", vec![d, d], 0.1),
            ("blocks.0.ffn", vec![d * 4, d], 0.2),
            ("blocks.0.ffn2", vec![d, d * 4], 0.3),
        ]);
        let model = import_torch_pt_bytes(&bytes).unwrap();
        assert_eq!(model.shape.vocab_size, vocab);
        assert_eq!(model.shape.n_layer, 1);
    }

    #[test]
    fn legacy_magic_mismatch_errors() {
        let mut bytes = build_legacy_torch(&[("w", vec![1], 1.0)]);
        bytes[4] ^= 0xFF; // corrupt the magic number
        assert!(!is_legacy_torch_bytes(&bytes));
        let e = read_legacy_torch_arrays_bytes(&bytes).err().unwrap();
        assert!(e.message().contains("magic number"));
    }

    #[test]
    fn legacy_truncated_storage_errors() {
        let bytes = build_legacy_torch(&[("w", vec![4], 2.0)]);
        let e = read_torch_arrays_bytes(&bytes[..bytes.len() - 8]).err().unwrap();
        assert!(e.message().contains("truncated"));
    }

    #[test]
    fn non_state_dict_pickle_errors() {
        let mut w = PickleWriter::new();
        w.string("just a string");
        let entries = vec![
            ("archive/data.pkl".to_string(), w.finish()),
            ("archive/version".to_string(), b"3\n".to_vec()),
        ];
        let bytes = write_zip_stored(&entries).unwrap();
        let e = import_torch_pt_bytes(&bytes).err().unwrap();
        assert!(e.message().contains("state dict"));
    }
}
