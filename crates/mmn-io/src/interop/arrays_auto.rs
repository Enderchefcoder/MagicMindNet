//! Universal named-array IO: one loader that detects any supported tensor
//! container by content, and generic GGUF array read/write.
//!
//! Complements the model-level `detect_checkpoint_kind` / `ai.load()` path:
//! this layer returns raw `(name, shape, f32 values)` triples for *any*
//! weights file, whatever ecosystem produced it.

use super::gguf::{is_gguf_bytes, read_gguf, write_gguf, GgufWriteTensor};
use super::ggml_legacy::{is_ggml_legacy_bytes, read_ggml_legacy_bytes};
use super::gguf_quant::GgmlType;
use super::hdf5::{is_hdf5_bytes, read_h5_arrays_bytes};
use super::npy::{decode_npy, is_npy_bytes};
use super::tflite::{is_tflite_bytes, read_tflite_arrays_bytes};
use super::zip::{is_zip_bytes, zip_entry_names};
use super::NamedArray;
use crate::checkpoint_util::write_file_create_parents;
use mmn_core::MmnError;
use std::fs;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// Dequantize every tensor in a GGUF file into named f32 arrays.
pub fn read_gguf_arrays_bytes(bytes: &[u8]) -> Result<Vec<NamedArray>, MmnError> {
    let file = read_gguf(bytes)?;
    let mut out = Vec::with_capacity(file.tensors.len());
    for info in &file.tensors {
        let (shape, values) = file.tensor_f32(info)?;
        out.push((info.name.clone(), shape, values));
    }
    Ok(out)
}

/// Dequantize every tensor in a `.gguf` file on disk.
pub fn read_gguf_arrays(path: &str) -> Result<Vec<NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read {path}: {e}")))?;
    read_gguf_arrays_bytes(&bytes)
}

/// Write named f32 arrays as a GGUF file (F32 tensors, no extra metadata).
pub fn write_gguf_arrays(path: &str, arrays: &[NamedArray]) -> Result<(), MmnError> {
    let tensors: Vec<GgufWriteTensor<'_>> = arrays
        .iter()
        .map(|(name, shape, values)| GgufWriteTensor {
            name: name.clone(),
            shape: shape.clone(),
            values,
            ggml_type: GgmlType::F32,
        })
        .collect();
    let bytes = write_gguf(&[], &tensors)?;
    write_file_create_parents(path, bytes)
}

/// Format detected by [`read_arrays_auto`], reported alongside the arrays.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ArrayFormat {
    Gguf,
    GgmlLegacy,
    TorchZip,
    Npz,
    KerasZip,
    Npy,
    Hdf5,
    Safetensors,
    LegacyTorch,
    FlaxMsgpack,
    Onnx,
    TfCheckpoint,
    Tflite,
    /// Generic pickle holding numpy arrays (PaddlePaddle `.pdparams`,
    /// sklearn model pickles, plain pickled ndarray dicts).
    Pickle,
    /// Sharded checkpoint index (`*.index.json` + shard files).
    ShardedIndex,
}

impl ArrayFormat {
    pub fn as_str(&self) -> &'static str {
        match self {
            ArrayFormat::Gguf => "gguf",
            ArrayFormat::GgmlLegacy => "ggml-legacy",
            ArrayFormat::TorchZip => "pt",
            ArrayFormat::Npz => "npz",
            ArrayFormat::KerasZip => "keras",
            ArrayFormat::Npy => "npy",
            ArrayFormat::Hdf5 => "h5",
            ArrayFormat::Safetensors => "safetensors",
            ArrayFormat::LegacyTorch => "pt-legacy",
            ArrayFormat::FlaxMsgpack => "flax",
            ArrayFormat::Onnx => "onnx",
            ArrayFormat::TfCheckpoint => "tf-checkpoint",
            ArrayFormat::Tflite => "tflite",
            ArrayFormat::Pickle => "pickle",
            ArrayFormat::ShardedIndex => "sharded",
        }
    }
}

/// True when the buffer looks like a binary safetensors container:
/// little-endian u64 header length followed by a `{` JSON header.
fn looks_like_safetensors(bytes: &[u8]) -> bool {
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

/// True when the buffer starts like a msgpack fixmap with a string key —
/// the shape of every Flax pytree (checked after pickle, whose 0x80 PROTO
/// prefix overlaps the empty-fixmap tag).
fn looks_like_flax_msgpack(bytes: &[u8]) -> bool {
    match bytes.first() {
        // Non-empty fixmap followed by a fixstr/str8 key.
        Some(&tag) if (0x81..=0x8F).contains(&tag) => matches!(
            bytes.get(1),
            Some(&key) if (0xA0..=0xBF).contains(&key) || key == 0xD9
        ),
        // map16 / map32 headers.
        Some(0xDE) | Some(0xDF) => true,
        _ => false,
    }
}

/// Detect which array container a buffer holds (path only used for
/// TF-checkpoint prefix resolution and error text).
pub fn detect_array_format(path: &str, bytes: &[u8]) -> Result<ArrayFormat, MmnError> {
    if is_gguf_bytes(bytes) {
        return Ok(ArrayFormat::Gguf);
    }
    if is_ggml_legacy_bytes(bytes) {
        return Ok(ArrayFormat::GgmlLegacy);
    }
    if is_tflite_bytes(bytes) {
        return Ok(ArrayFormat::Tflite);
    }
    if is_zip_bytes(bytes) {
        let names = zip_entry_names(bytes)?;
        if names
            .iter()
            .any(|n| n.ends_with("data.pkl") || n.ends_with("constants.pkl"))
        {
            return Ok(ArrayFormat::TorchZip);
        }
        if names.iter().any(|n| n.ends_with(".h5")) {
            return Ok(ArrayFormat::KerasZip);
        }
        if names.iter().any(|n| n.ends_with(".npy")) {
            return Ok(ArrayFormat::Npz);
        }
        return Err(err(format!(
            "{path}: zip archive holds no recognizable tensors (no data.pkl, .npy, or .h5 entries)"
        )));
    }
    if is_npy_bytes(bytes) {
        return Ok(ArrayFormat::Npy);
    }
    if is_hdf5_bytes(bytes) {
        return Ok(ArrayFormat::Hdf5);
    }
    if looks_like_safetensors(bytes) {
        return Ok(ArrayFormat::Safetensors);
    }
    if super::torch_pt::is_legacy_torch_bytes(bytes) {
        return Ok(ArrayFormat::LegacyTorch);
    }
    // Generic pickle (protocol 2-5 PROTO prefix, checked after the torch
    // legacy magic): PaddlePaddle .pdparams, sklearn pickles, ndarray dicts.
    if bytes.len() >= 2 && bytes[0] == 0x80 && (2..=5).contains(&bytes[1]) {
        return Ok(ArrayFormat::Pickle);
    }
    if looks_like_flax_msgpack(bytes) {
        return Ok(ArrayFormat::FlaxMsgpack);
    }
    if super::sharded::is_shard_index_bytes(bytes) {
        return Ok(ArrayFormat::ShardedIndex);
    }
    if path.ends_with(".index") || path.ends_with(".ckpt") {
        return Ok(ArrayFormat::TfCheckpoint);
    }
    // ONNX last: protobuf has no magic, so require the parser to accept it.
    if super::onnx::read_onnx_arrays_bytes(bytes).is_ok() {
        return Ok(ArrayFormat::Onnx);
    }
    Err(err(format!(
        "{path}: unrecognized tensor container (tried GGUF, legacy ggml/ggjt, TFLite, zip/pt/npz/keras, npy, HDF5, safetensors, legacy torch, Flax msgpack, ONNX)"
    )))
}

/// Load any supported weights file into named f32 arrays plus the detected
/// format tag. Single unnamed containers (`.npy`) use the name `"arr"`.
pub fn read_arrays_auto(path: &str) -> Result<(ArrayFormat, Vec<NamedArray>), MmnError> {
    // TF checkpoints are two files behind a prefix; resolve them by path.
    let as_path = std::path::Path::new(path);
    if as_path.is_dir()
        || path.ends_with(".index")
        || fs::metadata(format!("{path}.index")).is_ok()
    {
        let arrays = super::tf_checkpoint::read_tf_checkpoint_arrays(path)?;
        return Ok((ArrayFormat::TfCheckpoint, arrays));
    }
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read {path}: {e}")))?;
    let format = detect_array_format(path, &bytes)?;
    let arrays = match format {
        ArrayFormat::Gguf => read_gguf_arrays_bytes(&bytes)?,
        ArrayFormat::GgmlLegacy => read_ggml_legacy_bytes(&bytes)?.arrays,
        ArrayFormat::Tflite => read_tflite_arrays_bytes(&bytes)?,
        ArrayFormat::TorchZip | ArrayFormat::LegacyTorch => {
            super::torch_pt::read_torch_arrays_bytes(&bytes)?.0
        }
        ArrayFormat::Npz => super::npz_chatbot::read_npz_arrays(path)?,
        ArrayFormat::KerasZip | ArrayFormat::Hdf5 => read_h5_arrays_bytes(&bytes)?,
        ArrayFormat::Npy => {
            let arr = decode_npy(&bytes)?;
            vec![("arr".to_string(), arr.shape, arr.data)]
        }
        ArrayFormat::Safetensors => super::st_arrays::read_safetensors_arrays_bytes(&bytes)?,
        ArrayFormat::FlaxMsgpack => super::flax::read_flax_arrays_bytes(&bytes)?,
        ArrayFormat::Pickle => super::pickle_arrays::read_pickle_arrays_bytes(&bytes)?,
        ArrayFormat::ShardedIndex => super::sharded::read_sharded_arrays(path)?,
        ArrayFormat::Onnx => super::onnx::read_onnx_arrays_bytes(&bytes)?,
        ArrayFormat::TfCheckpoint => super::tf_checkpoint::read_tf_checkpoint_arrays(path)?,
    };
    Ok((format, arrays))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp(name: &str) -> String {
        std::env::temp_dir()
            .join(format!("mmn_auto_{}_{name}", std::process::id()))
            .to_string_lossy()
            .into_owned()
    }

    fn sample() -> Vec<NamedArray> {
        vec![
            ("a".to_string(), vec![2, 2], vec![1.0, -2.0, 3.5, 0.25]),
            ("b".to_string(), vec![3], vec![0.5, 0.0, -0.5]),
        ]
    }

    #[test]
    fn gguf_array_roundtrip() {
        let path = temp("arrays.gguf");
        write_gguf_arrays(&path, &sample()).unwrap();
        let (format, back) = read_arrays_auto(&path).unwrap();
        assert_eq!(format, ArrayFormat::Gguf);
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].2, vec![1.0, -2.0, 3.5, 0.25]);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn auto_detects_every_own_writer() {
        let arrays = sample();
        let cases: Vec<(String, ArrayFormat)> = vec![
            {
                let p = temp("auto.npz");
                super::super::npz_chatbot::write_npz_arrays(&p, &arrays).unwrap();
                (p, ArrayFormat::Npz)
            },
            {
                let p = temp("auto.pt");
                let bytes = super::super::torch_pt::write_torch_arrays(&arrays, None).unwrap();
                fs::write(&p, bytes).unwrap();
                (p, ArrayFormat::TorchZip)
            },
            {
                let p = temp("auto.h5");
                super::super::hdf5_write::write_h5_arrays(&p, &arrays).unwrap();
                (p, ArrayFormat::Hdf5)
            },
            {
                let p = temp("auto.safetensors");
                super::super::st_arrays::write_safetensors_arrays(&p, &arrays).unwrap();
                (p, ArrayFormat::Safetensors)
            },
            {
                let p = temp("auto.msgpack");
                super::super::flax::write_flax_arrays(&p, &arrays).unwrap();
                (p, ArrayFormat::FlaxMsgpack)
            },
            {
                let p = temp("auto.onnx");
                super::super::onnx::write_onnx_arrays(&p, &arrays).unwrap();
                (p, ArrayFormat::Onnx)
            },
        ];
        for (path, expected) in cases {
            let (format, back) = read_arrays_auto(&path).unwrap();
            assert_eq!(format, expected, "path {path}");
            assert_eq!(back.len(), 2, "path {path}");
            let a = back.iter().find(|(n, _, _)| n == "a").unwrap();
            assert_eq!(a.1, vec![2, 2], "path {path}");
            assert_eq!(a.2, vec![1.0, -2.0, 3.5, 0.25], "path {path}");
            let _ = fs::remove_file(&path);
        }
    }

    #[test]
    fn auto_reads_single_npy_and_tf_checkpoint() {
        let p = temp("auto.npy");
        fs::write(
            &p,
            super::super::npy::encode_npy_f32(&[2], &[7.0, 8.0]).unwrap(),
        )
        .unwrap();
        let (format, arrays) = read_arrays_auto(&p).unwrap();
        assert_eq!(format, ArrayFormat::Npy);
        assert_eq!(arrays, vec![("arr".to_string(), vec![2], vec![7.0, 8.0])]);
        let _ = fs::remove_file(&p);

        let prefix = temp("auto_ckpt");
        super::super::tf_checkpoint::write_tf_checkpoint_arrays(&prefix, &sample()).unwrap();
        let (format, arrays) = read_arrays_auto(&prefix).unwrap();
        assert_eq!(format, ArrayFormat::TfCheckpoint);
        assert_eq!(arrays.len(), 2);
        let _ = fs::remove_file(format!("{prefix}.index"));
        let _ = fs::remove_file(format!("{prefix}.data-00000-of-00001"));
    }

    #[test]
    fn unrecognized_container_errors() {
        let p = temp("auto.junk");
        fs::write(&p, b"this is not a tensor container at all").unwrap();
        let e = read_arrays_auto(&p).err().unwrap();
        assert!(
            e.message().contains("unrecognized tensor container"),
            "got: {}",
            e.message()
        );
        let _ = fs::remove_file(&p);
    }
}
