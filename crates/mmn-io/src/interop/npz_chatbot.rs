//! NumPy `.npz` chatbot checkpoints: one `.npy` entry per tensor plus a
//! `meta.json` entry. Doubles as the TensorFlow/Keras interchange path
//! (`np.savez(path, **{name: w for ...})` on `model.get_weights()` exports).

use super::npy::{decode_npy, encode_npy_f32};
use super::zip::{read_zip, write_zip_stored};
use crate::checkpoint_util::write_file_create_parents;
use crate::hf_safetensors::{
    chatbot_from_external_tensors, chatbot_meta_json, collect_named_tensors, hf_name_to_mmn,
};
use mmn_core::{MmnError, Tensor};
use mmn_models::Chatbot;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs;

pub const NPZ_META_ENTRY: &str = "meta.json";

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// Export a `Chatbot` as an `.npz` archive readable by `numpy.load`.
pub fn export_npz(model: &Chatbot, path: &str) -> Result<(), MmnError> {
    let named = collect_named_tensors(model);
    let mut keys: Vec<&String> = named.keys().collect();
    keys.sort();
    let mut entries: Vec<(String, Vec<u8>)> = Vec::with_capacity(named.len() + 1);
    let meta = chatbot_meta_json(model, Default::default());
    entries.push((NPZ_META_ENTRY.to_string(), meta.to_string().into_bytes()));
    for key in keys {
        let tensor = &named[key];
        let arr = tensor.data.as_standard_layout().into_owned();
        let shape = arr.shape().to_vec();
        let values: Vec<f32> = arr.iter().copied().collect();
        entries.push((format!("{key}.npy"), encode_npy_f32(&shape, &values)?));
    }
    let bytes = write_zip_stored(&entries)?;
    write_file_create_parents(path, bytes)
}

/// Import a `Chatbot` from an `.npz` archive (MMN or HF tensor names).
pub fn import_npz_bytes(bytes: &[u8]) -> Result<Chatbot, MmnError> {
    let entries = read_zip(bytes)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    let mut meta = serde_json::json!({});
    for entry in &entries {
        if entry.name == NPZ_META_ENTRY {
            meta = serde_json::from_slice(&entry.data)
                .map_err(|e| err(format!("npz meta.json parse failed: {e}")))?;
            continue;
        }
        let Some(stem) = entry.name.strip_suffix(".npy") else {
            continue;
        };
        let Some(mmn_key) = hf_name_to_mmn(stem) else {
            continue;
        };
        let arr = decode_npy(&entry.data)
            .map_err(|e| err(format!("npz entry {}: {}", entry.name, e.message())))?;
        let data = ArrayD::from_shape_vec(IxDyn(&arr.shape), arr.data)
            .map_err(|e| err(format!("npz entry {}: {e}", entry.name)))?;
        tensors.insert(mmn_key, Tensor::from_array(data, true));
    }
    if tensors.is_empty() {
        return Err(err(
            "npz archive contains no recognizable model tensors (expected entries like embed.npy or model.embed_tokens.weight.npy)",
        ));
    }
    chatbot_from_external_tensors(tensors, meta)
}

/// Import a `Chatbot` from an `.npz` file on disk.
pub fn import_npz(path: &str) -> Result<Chatbot, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read npz {path}: {e}")))?;
    import_npz_bytes(&bytes)
}

/// Read every array in an `.npz` archive (generic, model-agnostic).
pub fn read_npz_arrays(path: &str) -> Result<Vec<super::NamedArray>, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read npz {path}: {e}")))?;
    let entries = read_zip(&bytes)?;
    let mut out = Vec::new();
    for entry in entries {
        let Some(stem) = entry.name.strip_suffix(".npy") else {
            continue;
        };
        let arr = decode_npy(&entry.data)
            .map_err(|e| err(format!("npz entry {}: {}", entry.name, e.message())))?;
        out.push((stem.to_string(), arr.shape, arr.data));
    }
    Ok(out)
}

/// Write arrays to an `.npz` archive (generic, model-agnostic).
pub fn write_npz_arrays(path: &str, arrays: &[super::NamedArray]) -> Result<(), MmnError> {
    let mut entries = Vec::with_capacity(arrays.len());
    for (name, shape, data) in arrays {
        entries.push((format!("{name}.npy"), encode_npy_f32(shape, data)?));
    }
    let bytes = write_zip_stored(&entries)?;
    write_file_create_parents(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn npz_roundtrip_preserves_weights_and_meta() {
        let model = Chatbot::new_with_seed(false, None, 48, Some(2), Some(16), Some(9));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_npz_rt_{}.npz", std::process::id()));
        export_npz(&model, path.to_str().unwrap()).unwrap();
        let loaded = import_npz(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.shape.vocab_size, 48);
        assert_eq!(loaded.shape.n_layer, 2);
        assert_eq!(loaded.init_seed, Some(9));
        let a = model.blocks[0].ffn.weight.data[[1, 2]];
        let b = loaded.blocks[0].ffn.weight.data[[1, 2]];
        assert!((a - b).abs() < 1e-6);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn npz_import_accepts_hf_names_without_meta() {
        let d_model = 8usize;
        let vocab = 16usize;
        let mut arrays: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::new();
        let mut push = |name: &str, shape: &[usize], fill: f32| {
            let n: usize = shape.iter().product();
            arrays.push((name.to_string(), shape.to_vec(), vec![fill; n]));
        };
        push("model.embed_tokens.weight", &[vocab, d_model], 0.5);
        push("model.layers.0.self_attn.q_proj.weight", &[d_model, d_model], 0.1);
        push("model.layers.0.self_attn.k_proj.weight", &[d_model, d_model], 0.1);
        push("model.layers.0.self_attn.v_proj.weight", &[d_model, d_model], 0.1);
        push("model.layers.0.self_attn.o_proj.weight", &[d_model, d_model], 0.1);
        push("model.layers.0.mlp.up_proj.weight", &[d_model * 4, d_model], 0.2);
        push("model.layers.0.mlp.down_proj.weight", &[d_model, d_model * 4], 0.3);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_npz_hf_{}.npz", std::process::id()));
        write_npz_arrays(path.to_str().unwrap(), &arrays).unwrap();
        let loaded = import_npz(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.shape.vocab_size, vocab);
        assert_eq!(loaded.shape.d_model, d_model);
        assert_eq!(loaded.shape.n_layer, 1);
        assert!((loaded.embed.weight.data[[0, 0]] - 0.5).abs() < 1e-6);
        // up_proj alone becomes the FFN weight.
        assert!((loaded.blocks[0].ffn.weight.data[[0, 0]] - 0.2).abs() < 1e-6);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn generic_array_roundtrip() {
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_npz_arrays_{}.npz", std::process::id()));
        let arrays = vec![
            ("a".to_string(), vec![2, 2], vec![1.0, 2.0, 3.0, 4.0]),
            ("b/c".to_string(), vec![3], vec![-1.0, 0.0, 1.0]),
        ];
        write_npz_arrays(path.to_str().unwrap(), &arrays).unwrap();
        let back = read_npz_arrays(path.to_str().unwrap()).unwrap();
        assert_eq!(back.len(), 2);
        assert_eq!(back[0].0, "a");
        assert_eq!(back[0].1, vec![2, 2]);
        assert_eq!(back[1].2, vec![-1.0, 0.0, 1.0]);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn empty_archive_errors() {
        let bytes = write_zip_stored(&[("readme.txt".to_string(), b"hi".to_vec())]).unwrap();
        let e = import_npz_bytes(&bytes).err().unwrap();
        assert!(e.message().contains("no recognizable model tensors"));
    }
}
