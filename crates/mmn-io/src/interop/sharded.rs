//! Sharded Hugging Face checkpoints: `model.safetensors.index.json` /
//! `pytorch_model.bin.index.json` plus their shard files.

use super::torch_pt::read_torch_arrays_bytes;
use super::zip::is_zip_bytes;
use crate::hf_safetensors::{chatbot_from_external_tensors, hf_name_to_mmn};
use crate::hf_tensor_codec::{hf_err, tensor_from_view};
use mmn_core::{MmnError, Tensor};
use mmn_models::Chatbot;
use ndarray::{ArrayD, IxDyn};
use crate::st_codec::SafeTensors;
use std::collections::{BTreeSet, HashMap};
use std::fs;
use std::path::Path;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// True when the bytes are a shard index JSON (`weight_map` present).
pub fn is_shard_index_bytes(bytes: &[u8]) -> bool {
    if bytes.first() != Some(&b'{') {
        return false;
    }
    let Ok(text) = std::str::from_utf8(bytes) else {
        return false;
    };
    // Structural scan avoids serde-parsing multi-megabyte checkpoints just
    // to conclude they are not shard indexes.
    if let Some(found) = crate::mmn_json::top_level_key_is_object(text, "weight_map") {
        return found;
    }
    serde_json::from_slice::<serde_json::Value>(bytes)
        .map(|v| v.get("weight_map").is_some_and(|m| m.is_object()))
        .unwrap_or(false)
}

/// Shard file names referenced by an index JSON, in sorted order.
fn shard_files(index: &serde_json::Value) -> Result<Vec<String>, MmnError> {
    let map = index["weight_map"].as_object().ok_or_else(|| {
        err("shard index JSON has no weight_map object (not a sharded checkpoint index)")
    })?;
    let mut files = BTreeSet::new();
    for value in map.values() {
        let file = value
            .as_str()
            .ok_or_else(|| err("shard index weight_map values must be file names"))?;
        files.insert(file.to_string());
    }
    if files.is_empty() {
        return Err(err("shard index weight_map is empty"));
    }
    Ok(files.into_iter().collect())
}

fn collect_safetensors_shard(
    bytes: &[u8],
    tensors: &mut HashMap<String, Tensor>,
) -> Result<(), MmnError> {
    let st = SafeTensors::deserialize(bytes).map_err(hf_err)?;
    for name in st.names() {
        if let Some(mmn_key) = hf_name_to_mmn(name) {
            let view = st.tensor(name).map_err(hf_err)?;
            tensors.insert(mmn_key, tensor_from_view(name, &view)?);
        }
    }
    Ok(())
}

fn collect_torch_shard(
    bytes: &[u8],
    tensors: &mut HashMap<String, Tensor>,
) -> Result<(), MmnError> {
    let (arrays, _meta) = read_torch_arrays_bytes(bytes)?;
    for (name, shape, data) in arrays {
        let Some(mmn_key) = hf_name_to_mmn(&name) else {
            continue;
        };
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), data)
            .map_err(|e| err(format!("shard tensor {name}: {e}")))?;
        tensors.insert(mmn_key, Tensor::from_array(arr, true));
    }
    Ok(())
}

/// Import a sharded checkpoint from its `*.index.json` file.
///
/// Shards resolve relative to the index file and may be HF binary
/// safetensors or PyTorch `.bin`/`.pt` archives (zip or legacy).
pub fn import_sharded(index_path: &str) -> Result<Chatbot, MmnError> {
    let index_bytes = fs::read(index_path)
        .map_err(|e| err(format!("cannot read shard index {index_path}: {e}")))?;
    let index: serde_json::Value = serde_json::from_slice(&index_bytes)
        .map_err(|e| err(format!("shard index {index_path} is not JSON: {e}")))?;
    let dir = Path::new(index_path)
        .parent()
        .unwrap_or_else(|| Path::new("."));
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for file in shard_files(&index)? {
        let shard_path = dir.join(&file);
        let bytes = fs::read(&shard_path).map_err(|e| {
            err(format!(
                "cannot read shard {} (referenced by {index_path}): {e}",
                shard_path.display()
            ))
        })?;
        if is_zip_bytes(&bytes) || super::torch_pt::is_legacy_torch_bytes(&bytes) {
            collect_torch_shard(&bytes, &mut tensors)?;
        } else {
            collect_safetensors_shard(&bytes, &mut tensors)?;
        }
    }
    if tensors.is_empty() {
        return Err(err(
            "sharded checkpoint contains no recognizable model tensors",
        ));
    }
    let meta = index
        .get("metadata")
        .cloned()
        .unwrap_or_else(|| serde_json::json!({}));
    chatbot_from_external_tensors(tensors, meta)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interop::torch_pt::write_torch_arrays;
    use crate::st_codec::{Dtype, TensorView};
    use crate::st_codec::serialize;
    use std::path::PathBuf;

    fn tmp_dir(name: &str) -> PathBuf {
        let dir = std::env::temp_dir().join(format!("mmn_shard_{name}_{}", std::process::id()));
        let _ = fs::create_dir_all(&dir);
        dir
    }

    fn write_safetensors_shard(path: &Path, tensors: &[(&str, Vec<usize>, f32)]) {
        let storages: Vec<(String, Vec<usize>, Vec<u8>)> = tensors
            .iter()
            .map(|(name, shape, fill)| {
                let n: usize = shape.iter().product();
                let bytes: Vec<u8> = vec![*fill; n].iter().flat_map(|f| f.to_le_bytes()).collect();
                (name.to_string(), shape.clone(), bytes)
            })
            .collect();
        let mut views = HashMap::new();
        for (name, shape, bytes) in &storages {
            views.insert(
                name.clone(),
                TensorView::new(Dtype::F32, shape.clone(), bytes).unwrap(),
            );
        }
        fs::write(path, serialize(views, None).unwrap()).unwrap();
    }

    fn llama_tensor_set(d: usize, vocab: usize) -> Vec<(&'static str, Vec<usize>, f32)> {
        vec![
            ("model.embed_tokens.weight", vec![vocab, d], 0.5),
            ("model.layers.0.self_attn.q_proj.weight", vec![d, d], 0.1),
            ("model.layers.0.self_attn.k_proj.weight", vec![d, d], 0.1),
            ("model.layers.0.self_attn.v_proj.weight", vec![d, d], 0.1),
            ("model.layers.0.self_attn.o_proj.weight", vec![d, d], 0.1),
            ("model.layers.0.mlp.up_proj.weight", vec![d * 4, d], 0.2),
            ("model.layers.0.mlp.down_proj.weight", vec![d, d * 4], 0.3),
        ]
    }

    #[test]
    fn sharded_safetensors_checkpoint_imports() {
        let d = 8usize;
        let vocab = 16usize;
        let dir = tmp_dir("st");
        let all = llama_tensor_set(d, vocab);
        let (first, second) = all.split_at(3);
        write_safetensors_shard(&dir.join("model-00001-of-00002.safetensors"), first);
        write_safetensors_shard(&dir.join("model-00002-of-00002.safetensors"), second);
        let mut weight_map = serde_json::Map::new();
        for (i, (name, _, _)) in all.iter().enumerate() {
            let file = if i < 3 {
                "model-00001-of-00002.safetensors"
            } else {
                "model-00002-of-00002.safetensors"
            };
            weight_map.insert(name.to_string(), serde_json::json!(file));
        }
        let index = serde_json::json!({
            "metadata": {"total_size": 0},
            "weight_map": weight_map,
        });
        let index_path = dir.join("model.safetensors.index.json");
        fs::write(&index_path, index.to_string()).unwrap();
        assert!(is_shard_index_bytes(&fs::read(&index_path).unwrap()));
        let model = import_sharded(index_path.to_str().unwrap()).unwrap();
        assert_eq!(model.shape.vocab_size, vocab);
        assert_eq!(model.shape.n_layer, 1);
        assert!((model.embed.weight.data[[0, 0]] - 0.5).abs() < 1e-6);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn sharded_torch_bin_checkpoint_imports() {
        let d = 8usize;
        let vocab = 16usize;
        let dir = tmp_dir("pt");
        let all = llama_tensor_set(d, vocab);
        let (first, second) = all.split_at(4);
        for (file, tensors) in [
            ("pytorch_model-00001-of-00002.bin", first),
            ("pytorch_model-00002-of-00002.bin", second),
        ] {
            let arrays: Vec<crate::NamedArray> = tensors
                .iter()
                .map(|(name, shape, fill)| {
                    let n: usize = shape.iter().product();
                    (name.to_string(), shape.clone(), vec![*fill; n])
                })
                .collect();
            let bytes = write_torch_arrays(&arrays, None).unwrap();
            fs::write(dir.join(file), bytes).unwrap();
        }
        let mut weight_map = serde_json::Map::new();
        for (i, (name, _, _)) in all.iter().enumerate() {
            let file = if i < 4 {
                "pytorch_model-00001-of-00002.bin"
            } else {
                "pytorch_model-00002-of-00002.bin"
            };
            weight_map.insert(name.to_string(), serde_json::json!(file));
        }
        let index = serde_json::json!({"weight_map": weight_map});
        let index_path = dir.join("pytorch_model.bin.index.json");
        fs::write(&index_path, index.to_string()).unwrap();
        let model = import_sharded(index_path.to_str().unwrap()).unwrap();
        assert_eq!(model.shape.vocab_size, vocab);
        assert!((model.blocks[0].ffn2.weight.data[[0, 0]] - 0.3).abs() < 1e-6);
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn missing_shard_file_errors_with_both_paths() {
        let dir = tmp_dir("missing");
        let index = serde_json::json!({
            "weight_map": {"w": "not-there.safetensors"}
        });
        let index_path = dir.join("model.safetensors.index.json");
        fs::write(&index_path, index.to_string()).unwrap();
        let e = import_sharded(index_path.to_str().unwrap()).err().unwrap();
        assert!(e.message().contains("not-there.safetensors"));
        assert!(e.message().contains("index.json"));
        let _ = fs::remove_dir_all(&dir);
    }

    #[test]
    fn non_index_json_rejected() {
        assert!(!is_shard_index_bytes(b"{\"format\": \"mmn-bin-v1\"}"));
        assert!(!is_shard_index_bytes(b"GGUF"));
    }
}
