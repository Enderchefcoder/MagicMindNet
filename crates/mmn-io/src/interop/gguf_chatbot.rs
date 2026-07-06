//! GGUF ↔ `Chatbot` bridge: llama.cpp-style tensor names and `{arch}.*`
//! metadata keys, on top of the from-scratch GGUF container.

use super::gguf::{read_gguf, write_gguf, GgufFile, GgufValue, GgufWriteTensor};
use super::gguf_quant::GgmlType;
use crate::checkpoint_util::write_file_create_parents;
use crate::hf_safetensors::{chatbot_from_external_tensors, collect_named_tensors};
use mmn_core::{MmnError, Tensor};
use mmn_models::Chatbot;
use ndarray::{ArrayD, IxDyn};
use std::collections::HashMap;
use std::fs;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// Map a GGUF (llama.cpp-convention) tensor name to an MMN checkpoint key.
pub fn gguf_name_to_mmn(name: &str) -> Option<String> {
    match name {
        "token_embd.weight" => return Some("embed".into()),
        "output.weight" => return Some("lm_head".into()),
        "position_embd.weight" => return Some("pos_embed".into()),
        // Final-norm has no MMN equivalent (blocks carry their own norms).
        "output_norm.weight" | "output_norm.bias" => return None,
        _ => {}
    }
    let rest = name.strip_prefix("blk.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let i: usize = idx.parse().ok()?;
    let mmn_suffix = match suffix {
        "attn_q.weight" => "attn.q",
        "attn_k.weight" => "attn.k",
        "attn_v.weight" => "attn.v",
        "attn_output.weight" => "attn.out",
        "attn_qkv.weight" => "attn.qkv",
        "ffn_gate.weight" => "ffn",
        "ffn_up.weight" => "ffn.up",
        "ffn_down.weight" => "ffn2",
        "attn_norm.weight" => "ln1.gamma",
        "attn_norm.bias" => "ln1.beta",
        "ffn_norm.weight" => "ln2.gamma",
        "ffn_norm.bias" => "ln2.beta",
        _ => return None,
    };
    Some(format!("blocks.{i}.{mmn_suffix}"))
}

/// Map an MMN checkpoint key to the GGUF (llama.cpp-convention) tensor name.
pub fn mmn_name_to_gguf(key: &str) -> Option<String> {
    match key {
        "embed" => return Some("token_embd.weight".into()),
        "lm_head" => return Some("output.weight".into()),
        "pos_embed" => return Some("position_embd.weight".into()),
        _ => {}
    }
    let rest = key.strip_prefix("blocks.")?;
    let (idx, suffix) = rest.split_once('.')?;
    let i: usize = idx.parse().ok()?;
    let gguf_suffix = match suffix {
        "attn.q" => "attn_q.weight",
        "attn.k" => "attn_k.weight",
        "attn.v" => "attn_v.weight",
        "attn.out" => "attn_output.weight",
        "ffn" => "ffn_up.weight",
        "ffn2" => "ffn_down.weight",
        "ln1.gamma" => "attn_norm.weight",
        "ln1.beta" => "attn_norm.bias",
        "ln2.gamma" => "ffn_norm.weight",
        "ln2.beta" => "ffn_norm.bias",
        _ => return None,
    };
    Some(format!("blk.{i}.{gguf_suffix}"))
}

fn meta_u64(file: &GgufFile, key: &str) -> Option<u64> {
    file.metadata.get(key).and_then(|v| v.as_u64())
}

/// Translate GGUF `{arch}.*` metadata into MMN checkpoint meta JSON.
fn gguf_meta_to_mmn(file: &GgufFile, tensors: &HashMap<String, Tensor>) -> serde_json::Value {
    let arch = file
        .metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("llama")
        .to_string();
    let mut meta = serde_json::json!({});
    if let Some(embed) = tensors.get("embed") {
        let shape = embed.data.shape();
        if shape.len() == 2 {
            meta["vocab_size"] = serde_json::json!(shape[0]);
            meta["d_model"] = serde_json::json!(shape[1]);
        }
    }
    if let Some(d_model) = meta_u64(file, &format!("{arch}.embedding_length")) {
        meta["d_model"] = serde_json::json!(d_model);
    }
    if let Some(n_layer) = meta_u64(file, &format!("{arch}.block_count")) {
        meta["n_layer"] = serde_json::json!(n_layer);
    } else {
        let n_layer = tensors
            .keys()
            .filter_map(|k| {
                k.strip_prefix("blocks.")?
                    .split('.')
                    .next()?
                    .parse::<usize>()
                    .ok()
            })
            .max()
            .map(|i| i + 1)
            .unwrap_or(0);
        meta["n_layer"] = serde_json::json!(n_layer);
    }
    if let Some(ffn_dim) = meta_u64(file, &format!("{arch}.feed_forward_length")) {
        meta["ffn_dim"] = serde_json::json!(ffn_dim);
    }
    if let Some(n_heads) = meta_u64(file, &format!("{arch}.attention.head_count")) {
        meta["num_attention_heads"] = serde_json::json!(n_heads);
    }
    if let Some(n_kv) = meta_u64(file, &format!("{arch}.attention.head_count_kv")) {
        meta["num_key_value_heads"] = serde_json::json!(n_kv);
    }
    let rope_base = file
        .metadata
        .get(&format!("{arch}.rope.freq_base"))
        .and_then(|v| v.as_f64());
    let use_rope = file
        .metadata
        .get(&format!("{arch}.use_rope"))
        .and_then(|v| v.as_bool())
        // Llama-family GGUF models always use rotary embeddings.
        .unwrap_or(rope_base.is_some() || arch != "mmn");
    if use_rope {
        meta["use_rope"] = serde_json::json!(true);
        meta["rope_theta"] = serde_json::json!(rope_base.unwrap_or(10_000.0));
    }
    if let Some(pe) = tensors.get("pos_embed") {
        let shape = pe.data.shape();
        if shape.len() == 2 {
            meta["use_learned_pos_embed"] = serde_json::json!(true);
            meta["max_seq_len"] = serde_json::json!(shape[0]);
        }
    }
    if let Some(seed) = meta_u64(file, &format!("{arch}.seed")) {
        meta["seed"] = serde_json::json!(seed);
    }
    meta["vision"] = serde_json::json!(false);
    meta
}

/// Import a GGUF model file into a `Chatbot` (dequantizing as needed).
pub fn import_gguf_bytes(bytes: &[u8]) -> Result<Chatbot, MmnError> {
    let file = read_gguf(bytes)?;
    let mut tensors: HashMap<String, Tensor> = HashMap::new();
    for info in &file.tensors {
        let Some(mmn_key) = gguf_name_to_mmn(&info.name) else {
            continue;
        };
        let (shape, values) = file.tensor_f32(info)?;
        let arr = ArrayD::from_shape_vec(IxDyn(&shape), values).map_err(|e| {
            err(format!("GGUF tensor {}: {e}", info.name))
        })?;
        tensors.insert(mmn_key, Tensor::from_array(arr, true));
    }
    if tensors.is_empty() {
        return Err(err(
            "GGUF file contains no recognizable model tensors (expected llama.cpp names like token_embd.weight, blk.0.attn_q.weight)",
        ));
    }
    let meta = gguf_meta_to_mmn(&file, &tensors);
    chatbot_from_external_tensors(tensors, meta)
}

/// Import a GGUF model from disk.
pub fn import_gguf(path: &str) -> Result<Chatbot, MmnError> {
    let bytes = fs::read(path).map_err(|e| err(format!("cannot read GGUF {path}: {e}")))?;
    import_gguf_bytes(&bytes)
}

fn chatbot_gguf_metadata(model: &Chatbot) -> Vec<(String, GgufValue)> {
    let mut meta = vec![
        (
            "general.architecture".to_string(),
            GgufValue::String("mmn".into()),
        ),
        (
            "general.name".to_string(),
            GgufValue::String("MagicMindNet Chatbot".into()),
        ),
        (
            "mmn.embedding_length".to_string(),
            GgufValue::U32(model.shape.d_model as u32),
        ),
        (
            "mmn.block_count".to_string(),
            GgufValue::U32(model.shape.n_layer as u32),
        ),
        (
            "mmn.feed_forward_length".to_string(),
            GgufValue::U32(model.shape.ffn_dim as u32),
        ),
        (
            "mmn.attention.head_count".to_string(),
            GgufValue::U32(model.shape.n_heads as u32),
        ),
        (
            "mmn.attention.head_count_kv".to_string(),
            GgufValue::U32(model.shape.n_kv_heads as u32),
        ),
        (
            "mmn.vocab_size".to_string(),
            GgufValue::U32(model.shape.vocab_size as u32),
        ),
        (
            "mmn.context_length".to_string(),
            GgufValue::U32(model.max_seq_len as u32),
        ),
        ("mmn.use_rope".to_string(), GgufValue::Bool(model.use_rope)),
    ];
    if model.use_rope {
        meta.push((
            "mmn.rope.freq_base".to_string(),
            GgufValue::F32(model.rope_theta),
        ));
    }
    if let Some(seed) = model.init_seed {
        meta.push(("mmn.seed".to_string(), GgufValue::U64(seed)));
    }
    meta
}

/// Export a `Chatbot` as a GGUF v3 file (`quant`: "f32" or "q8_0").
pub fn export_gguf(model: &Chatbot, path: &str, quant: &str) -> Result<(), MmnError> {
    if model.vision {
        return Err(err(
            "GGUF export does not support vision models yet; use safetensors or npz",
        ));
    }
    let ggml_type = match quant {
        "f32" | "F32" => GgmlType::F32,
        "q8_0" | "Q8_0" => GgmlType::Q8_0,
        other => {
            return Err(err(format!(
                "GGUF export quant {other:?} not supported (use \"f32\" or \"q8_0\")"
            )));
        }
    };
    let named = collect_named_tensors(model);
    // Deterministic tensor order: sorted MMN keys.
    let mut keys: Vec<&String> = named.keys().collect();
    keys.sort();
    let mut standardized: Vec<(String, Vec<usize>, Vec<f32>)> = Vec::with_capacity(keys.len());
    for key in keys {
        let Some(gguf_name) = mmn_name_to_gguf(key) else {
            return Err(err(format!(
                "tensor {key} has no GGUF name mapping; export as safetensors instead"
            )));
        };
        let tensor = &named[key];
        let arr = tensor.data.as_standard_layout().into_owned();
        let shape = arr.shape().to_vec();
        let values: Vec<f32> = arr.iter().copied().collect();
        standardized.push((gguf_name, shape, values));
    }
    let tensors: Vec<GgufWriteTensor<'_>> = standardized
        .iter()
        .map(|(name, shape, values)| {
            // Row (32-elem-aligned) tensors quantize; small vectors stay F32.
            let use_quant = ggml_type == GgmlType::Q8_0
                && values.len().is_multiple_of(super::gguf_quant::QK);
            GgufWriteTensor {
                name: name.clone(),
                shape: shape.clone(),
                values,
                ggml_type: if use_quant { GgmlType::Q8_0 } else { GgmlType::F32 },
            }
        })
        .collect();
    let meta = chatbot_gguf_metadata(model);
    let bytes = write_gguf(&meta, &tensors)?;
    write_file_create_parents(path, bytes)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn name_mapping_roundtrips() {
        for key in [
            "embed",
            "lm_head",
            "blocks.0.attn.q",
            "blocks.3.ffn2",
            "blocks.1.ln1.gamma",
            "blocks.2.ln2.beta",
        ] {
            let gguf = mmn_name_to_gguf(key).unwrap();
            let back = gguf_name_to_mmn(&gguf).unwrap();
            // ffn maps through ffn_up -> ffn.up which adapt folds back to ffn.
            if key != "blocks.0.ffn" {
                assert_eq!(back, key, "via {gguf}");
            }
        }
        assert_eq!(
            gguf_name_to_mmn("blk.0.ffn_up.weight").as_deref(),
            Some("blocks.0.ffn.up")
        );
        assert_eq!(gguf_name_to_mmn("output_norm.weight"), None);
        assert_eq!(gguf_name_to_mmn("unknown.weight"), None);
    }

    #[test]
    fn gguf_roundtrip_preserves_weights() {
        let model = Chatbot::new_with_seed(false, None, 64, Some(2), Some(16), Some(11));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_rt_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "f32").unwrap();
        let loaded = import_gguf(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.shape.vocab_size, 64);
        assert_eq!(loaded.shape.n_layer, 2);
        assert_eq!(loaded.shape.d_model, 16);
        assert_eq!(loaded.use_rope, model.use_rope);
        let a = model.embed.weight.data[[3, 5]];
        let b = loaded.embed.weight.data[[3, 5]];
        assert!((a - b).abs() < 1e-6);
        let aw = model.blocks[1].attn.out_proj.weight.data[[2, 2]];
        let bw = loaded.blocks[1].attn.out_proj.weight.data[[2, 2]];
        assert!((aw - bw).abs() < 1e-6);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn gguf_q8_0_roundtrip_close() {
        let model = Chatbot::new_with_seed(false, None, 64, Some(1), Some(32), Some(5));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_q8_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "q8_0").unwrap();
        let loaded = import_gguf(path.to_str().unwrap()).unwrap();
        let a = model.embed.weight.data[[1, 1]];
        let b = loaded.embed.weight.data[[1, 1]];
        assert!((a - b).abs() < 0.02, "{a} vs {b}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn export_unknown_quant_errors() {
        let model = Chatbot::new(false, None, 32, Some(1), Some(8));
        let e = export_gguf(&model, "/tmp/never.gguf", "q4_k").unwrap_err();
        assert!(e.message().contains("not supported"));
    }

    #[test]
    fn import_no_model_tensors_errors() {
        let bytes = write_gguf(&[], &[]).unwrap();
        let e = import_gguf_bytes(&bytes).err().unwrap();
        assert!(e.message().contains("no recognizable model tensors"));
    }

    #[test]
    fn import_reads_gqa_meta_from_kv() {
        let model = Chatbot::new_with_seed(false, None, 64, Some(1), Some(16), Some(3));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_meta_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "f32").unwrap();
        let loaded = import_gguf(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.shape.n_heads, model.shape.n_heads);
        assert_eq!(loaded.shape.n_kv_heads, model.shape.n_kv_heads);
        assert_eq!(loaded.shape.ffn_dim, model.shape.ffn_dim);
        let _ = fs::remove_file(&path);
    }
}
