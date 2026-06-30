//! Hugging Face binary safetensors interchange (`safetensors` crate on disk).

use crate::block_tensors::import_block_tensors;
use crate::checkpoint_util::{
    expect_tensor_shape, require_tensor_entry, tensor_from_entry, tensor_to_entry,
    write_file_create_parents,
};
use mmn_core::{MmnError, Tensor};
use mmn_models::{Chatbot, DEFAULT_MAX_SEQ_LEN, DEFAULT_ROPE_THETA};
use safetensors::tensor::{Dtype, TensorView};
use safetensors::{serialize, SafeTensorError, SafeTensors};
use std::collections::HashMap;
use std::fs;

pub const HF_FORMAT: &str = "mmn-hf-safetensors-v1";

fn tensor_bytes_f32(t: &Tensor) -> (Vec<usize>, Vec<u8>) {
    let arr = t.data.as_standard_layout().into_owned();
    let shape = arr.shape().to_vec();
    let bytes: Vec<u8> = arr.iter().flat_map(|f| f.to_le_bytes()).collect();
    (shape, bytes)
}

fn tensor_from_view(name: &str, view: &TensorView<'_>) -> Result<Tensor, MmnError> {
    if view.dtype() != Dtype::F32 {
        return Err(MmnError::Other {
            message: format!("tensor {name}: HF safetensors dtype {:?} not supported (F32 only)", view.dtype()),
        });
    }
    let shape: Vec<usize> = view.shape().to_vec();
    let data = view.data();
    if data.len() % 4 != 0 {
        return Err(MmnError::Other {
            message: format!("tensor {name}: invalid F32 byte length {}", data.len()),
        });
    }
    let mut floats = Vec::with_capacity(data.len() / 4);
    for chunk in data.chunks_exact(4) {
        floats.push(f32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]));
    }
    let arr = ndarray::ArrayD::from_shape_vec(ndarray::IxDyn(&shape), floats).map_err(|e| {
        MmnError::Other {
            message: format!("tensor {name}: {e}"),
        }
    })?;
    Ok(Tensor::from_array(arr, true))
}

fn chatbot_meta_json(model: &Chatbot, bpe_checkpoint: Option<&str>) -> serde_json::Value {
    let mut meta = serde_json::json!({
        "vocab_size": model.shape.vocab_size,
        "n_layer": model.shape.n_layer,
        "d_model": model.shape.d_model,
        "vision": model.vision,
    });
    if model.vision {
        meta["vision_patch_dim"] = serde_json::json!(mmn_models::VISION_PATCH_DIM);
        meta["vision_rgb_dim"] = serde_json::json!(mmn_models::VISION_RGB_DIM);
        if model.vision_patch_conv.is_some() {
            meta["vision_rgb_patch"] = serde_json::json!(true);
        }
        if model.vision_cross_attn.is_some() {
            meta["vision_cross_attn"] = serde_json::json!(true);
        }
    }
    if let Some(seed) = model.init_seed {
        meta["seed"] = serde_json::json!(seed);
    }
    if model.use_learned_pos_embed {
        meta["use_learned_pos_embed"] = serde_json::json!(true);
        meta["max_seq_len"] = serde_json::json!(model.max_seq_len);
    }
    if model.use_rope {
        meta["use_rope"] = serde_json::json!(true);
        meta["rope_theta"] = serde_json::json!(model.rope_theta);
    }
    if let Some(bpe_path) = bpe_checkpoint {
        meta["bpe_checkpoint"] = serde_json::json!(bpe_path);
    }
    meta
}

fn collect_named_tensors(model: &Chatbot) -> HashMap<String, Tensor> {
    let mut map = HashMap::new();
    map.insert("embed".to_string(), model.embed.weight.clone());
    map.insert("lm_head".to_string(), model.lm_head.weight.clone());
    for (i, block) in model.blocks.iter().enumerate() {
        let p = format!("blocks.{i}");
        map.insert(format!("{p}.attn.q"), block.attn.q_proj.weight.clone());
        map.insert(format!("{p}.attn.k"), block.attn.k_proj.weight.clone());
        map.insert(format!("{p}.attn.v"), block.attn.v_proj.weight.clone());
        map.insert(format!("{p}.attn.out"), block.attn.out_proj.weight.clone());
        map.insert(format!("{p}.ffn"), block.ffn.weight.clone());
        map.insert(format!("{p}.ffn2"), block.ffn2.weight.clone());
        map.insert(format!("{p}.ln1.gamma"), block.ln1.gamma.clone());
        map.insert(format!("{p}.ln1.beta"), block.ln1.beta.clone());
        map.insert(format!("{p}.ln2.gamma"), block.ln2.gamma.clone());
        map.insert(format!("{p}.ln2.beta"), block.ln2.beta.clone());
    }
    if let Some(proj) = &model.vision_patch_proj {
        map.insert("vision_patch_proj".to_string(), proj.weight.clone());
    }
    if let Some(conv) = &model.vision_patch_conv {
        map.insert("vision_patch_conv".to_string(), conv.weight.clone());
    }
    if let Some(cross) = &model.vision_cross_attn {
        map.insert("vision_cross_attn.out".to_string(), cross.out_proj.weight.clone());
        map.insert("vision_cross_attn.q".to_string(), cross.q_proj.weight.clone());
        map.insert("vision_cross_attn.k".to_string(), cross.k_proj.weight.clone());
        map.insert("vision_cross_attn.v".to_string(), cross.v_proj.weight.clone());
    }
    if let Some(pe) = &model.pos_embed {
        map.insert("pos_embed".to_string(), pe.weight.clone());
    }
    map
}

/// Write a Hugging Face binary safetensors checkpoint (F32 tensors, MMN key names).
pub fn export_hf_safetensors(
    model: &Chatbot,
    path: &str,
    bpe_checkpoint: Option<&str>,
) -> Result<(), MmnError> {
    let named = collect_named_tensors(model);
    let mut storages: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (k, t) in &named {
        let (shape, bytes) = tensor_bytes_f32(t);
        storages.push((k.clone(), shape, bytes));
    }
    let mut views: HashMap<String, TensorView<'_>> = HashMap::new();
    for (k, shape, bytes) in &storages {
        views.insert(
            k.clone(),
            TensorView::new(Dtype::F32, shape.clone(), bytes).map_err(|e| MmnError::Other {
                message: e.to_string(),
            })?,
        );
    }
    let meta = chatbot_meta_json(model, bpe_checkpoint);
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), HF_FORMAT.to_string());
    metadata.insert("meta".to_string(), meta.to_string());
    let bytes = serialize(views, Some(metadata)).map_err(hf_err)?;
    write_file_create_parents(path, bytes)
}

pub fn is_hf_safetensors_bytes(bytes: &[u8]) -> bool {
    !bytes.is_empty() && bytes[0] != b'{'
}

fn hf_err(e: SafeTensorError) -> MmnError {
    MmnError::Other {
        message: e.to_string(),
    }
}

/// Map HF / MMN tensor names to canonical MMN checkpoint keys.
pub fn hf_name_to_mmn(name: &str) -> Option<String> {
    if name.starts_with("blocks.") {
        return Some(name.to_string());
    }
    match name {
        "embed" | "model.embed_tokens.weight" | "transformer.wte.weight" | "gpt_neox.embed_in.weight" => {
            Some("embed".into())
        }
        "lm_head" | "lm_head.weight" | "model.lm_head.weight" => Some("lm_head".into()),
        "model.embed_positions.weight" | "transformer.wpe.weight" | "pos_embed" => {
            Some("pos_embed".into())
        }
        "vision_patch_proj" | "vision_patch_conv" => Some(name.to_string()),
        "vision_cross_attn.out" | "vision_cross_attn.q" | "vision_cross_attn.k" | "vision_cross_attn.v" => {
            Some(name.to_string())
        }
        _ => parse_hf_layer_tensor(name),
    }
}

fn parse_hf_layer_tensor(name: &str) -> Option<String> {
    let rest = name
        .strip_prefix("model.layers.")
        .or_else(|| name.strip_prefix("transformer.h."))
        .or_else(|| name.strip_prefix("gpt_neox.layers."))?;
    let (idx, suffix) = rest.split_once('.')?;
    let i: usize = idx.parse().ok()?;
    let mmn_suffix = match suffix {
        "self_attn.q_proj.weight" => "attn.q",
        "self_attn.k_proj.weight" => "attn.k",
        "self_attn.v_proj.weight" => "attn.v",
        "self_attn.o_proj.weight" | "self_attn.out_proj.weight" => "attn.out",
        "attn.c_proj.weight" => "attn.out",
        "mlp.gate_proj.weight" | "mlp.fc1.weight" | "mlp.c_fc.weight" => "ffn",
        "mlp.down_proj.weight" | "mlp.fc2.weight" | "mlp.c_proj.weight" => "ffn2",
        "input_layernorm.weight" | "ln_1.weight" => "ln1.gamma",
        "input_layernorm.bias" | "ln_1.bias" => "ln1.beta",
        "post_attention_layernorm.weight" | "ln_2.weight" => "ln2.gamma",
        "post_attention_layernorm.bias" | "ln_2.bias" => "ln2.beta",
        _ => return None,
    };
    Some(format!("blocks.{i}.{mmn_suffix}"))
}

fn tensors_to_json_map(tensors: &HashMap<String, Tensor>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, t) in tensors {
        map.insert(k.clone(), tensor_to_entry(t));
    }
    serde_json::Value::Object(map)
}

fn load_chatbot_from_mmn_tensors(
    tensors: HashMap<String, Tensor>,
    meta: &serde_json::Value,
) -> Result<Chatbot, MmnError> {
    let vocab_size = meta["vocab_size"].as_u64().ok_or_else(|| MmnError::Other {
        message: "checkpoint meta missing vocab_size".into(),
    })? as usize;
    let n_layer = meta["n_layer"].as_u64().ok_or_else(|| MmnError::Other {
        message: "checkpoint meta missing n_layer".into(),
    })? as usize;
    let d_model = meta["d_model"].as_u64().ok_or_else(|| MmnError::Other {
        message: "checkpoint meta missing d_model".into(),
    })? as usize;
    let vision = meta["vision"].as_bool().unwrap_or(false);
    let init_seed = meta["seed"].as_u64();
    let use_learned_pos_embed = meta["use_learned_pos_embed"].as_bool().unwrap_or(false);
    let use_rope = meta["use_rope"].as_bool().unwrap_or(false);
    let rope_theta = meta["rope_theta"]
        .as_f64()
        .unwrap_or(DEFAULT_ROPE_THETA as f64) as f32;
    let max_seq_len = meta["max_seq_len"]
        .as_u64()
        .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize;

    let json_tensors = tensors_to_json_map(&tensors);
    let mut model = Chatbot::new_with_position_options(
        vision,
        None,
        vocab_size,
        Some(n_layer),
        Some(d_model),
        init_seed,
        use_learned_pos_embed,
        max_seq_len,
        use_rope,
        rope_theta,
    );
    model.embed.weight = tensor_from_entry(require_tensor_entry(&json_tensors, "embed")?)?;
    model.lm_head.weight = tensor_from_entry(require_tensor_entry(&json_tensors, "lm_head")?)?;
    expect_tensor_shape(&model.embed.weight, &[vocab_size, d_model], "embed")?;
    expect_tensor_shape(&model.lm_head.weight, &[vocab_size, d_model], "lm_head")?;
    if use_learned_pos_embed {
        let pe = model.pos_embed.as_mut().ok_or_else(|| MmnError::Other {
            message: "use_learned_pos_embed meta set but model has no pos_embed".into(),
        })?;
        pe.weight = tensor_from_entry(require_tensor_entry(&json_tensors, "pos_embed")?)?;
        expect_tensor_shape(&pe.weight, &[max_seq_len, d_model], "pos_embed")?;
    }
    if vision {
        if let Some(proj) = model.vision_patch_proj.as_mut() {
            if let Ok(entry) = require_tensor_entry(&json_tensors, "vision_patch_proj") {
                proj.weight = tensor_from_entry(entry)?;
                expect_tensor_shape(
                    &proj.weight,
                    &[d_model, mmn_models::VISION_PATCH_DIM],
                    "vision_patch_proj",
                )?;
            }
        }
        if let Ok(entry) = require_tensor_entry(&json_tensors, "vision_patch_conv") {
            let conv = model.vision_patch_conv.as_mut().ok_or_else(|| MmnError::Other {
                message: "vision_patch_conv tensor present but model has no conv encoder".into(),
            })?;
            conv.weight = tensor_from_entry(entry)?;
            expect_tensor_shape(
                &conv.weight,
                &[1, mmn_models::VISION_RGB_CHANNELS, 3, 3],
                "vision_patch_conv",
            )?;
        } else {
            model.vision_patch_conv = None;
        }
        if let Ok(entry) = require_tensor_entry(&json_tensors, "vision_cross_attn.out") {
            let cross = model.vision_cross_attn.as_mut().ok_or_else(|| MmnError::Other {
                message: "vision_cross_attn tensor present but model has no cross-attn".into(),
            })?;
            cross.out_proj.weight = tensor_from_entry(entry)?;
            cross.q_proj.weight =
                tensor_from_entry(require_tensor_entry(&json_tensors, "vision_cross_attn.q")?)?;
            cross.k_proj.weight =
                tensor_from_entry(require_tensor_entry(&json_tensors, "vision_cross_attn.k")?)?;
            cross.v_proj.weight =
                tensor_from_entry(require_tensor_entry(&json_tensors, "vision_cross_attn.v")?)?;
        } else {
            model.vision_cross_attn = None;
        }
    }
    import_block_tensors(&mut model, &json_tensors)?;
    Ok(model)
}

/// Import a Hugging Face binary safetensors file into a `Chatbot`.
pub fn import_hf_safetensors(path: &str) -> Result<Chatbot, MmnError> {
    let bytes = fs::read(path).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    import_hf_safetensors_bytes(&bytes)
}

pub fn import_hf_safetensors_bytes(bytes: &[u8]) -> Result<Chatbot, MmnError> {
    let (_header_len, header_meta) = SafeTensors::read_metadata(bytes).map_err(hf_err)?;
    let st = SafeTensors::deserialize(bytes).map_err(hf_err)?;
    let file_meta = header_meta.metadata();
    let meta = file_meta
        .as_ref()
        .and_then(|m| m.get("meta"))
        .map(|meta_str| {
            serde_json::from_str(meta_str).map_err(|e| MmnError::Other {
                message: format!("invalid HF safetensors meta JSON: {e}"),
            })
        })
        .transpose()?
        .unwrap_or_else(|| serde_json::json!({}));
    if let Some(m) = file_meta.as_ref() {
        if let Some(fmt) = m.get("format") {
            if fmt != HF_FORMAT {
                return Err(MmnError::Other {
                    message: format!("Expected {HF_FORMAT} metadata format, got {fmt}"),
                });
            }
        }
    }
    let mut mmn_tensors: HashMap<String, Tensor> = HashMap::new();
    for name in st.names() {
        if let Some(mmn_key) = hf_name_to_mmn(name) {
            let view = st.tensor(name).map_err(hf_err)?;
            let t = tensor_from_view(name, &view)?;
            mmn_tensors.insert(mmn_key, t);
        }
    }
    if meta.get("vocab_size").is_none() {
        let embed = mmn_tensors.get("embed").ok_or_else(|| MmnError::Other {
            message: "HF safetensors missing embed / model.embed_tokens.weight".into(),
        })?;
        let shape: Vec<usize> = embed.data.shape().iter().copied().collect();
        if shape.len() != 2 {
            return Err(MmnError::Other {
                message: format!("embed shape {:?} cannot infer vocab_size/d_model", shape),
            });
        }
        let inferred = serde_json::json!({
            "vocab_size": shape[0],
            "d_model": shape[1],
            "n_layer": count_block_layers(&mmn_tensors),
            "vision": mmn_tensors.contains_key("vision_patch_proj"),
        });
        return load_chatbot_from_mmn_tensors(mmn_tensors, &inferred);
    }
    load_chatbot_from_mmn_tensors(mmn_tensors, &meta)
}

fn count_block_layers(tensors: &HashMap<String, Tensor>) -> usize {
    let mut max_idx = 0usize;
    for key in tensors.keys() {
        if let Some(rest) = key.strip_prefix("blocks.") {
            if let Some(idx) = rest.split('.').next().and_then(|s| s.parse::<usize>().ok()) {
                max_idx = max_idx.max(idx + 1);
            }
        }
    }
    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;
    use mmn_models::Chatbot;

    #[test]
    fn hf_name_aliases_map_llama_style() {
        assert_eq!(
            hf_name_to_mmn("model.layers.2.self_attn.q_proj.weight").as_deref(),
            Some("blocks.2.attn.q")
        );
        assert_eq!(
            hf_name_to_mmn("model.layers.0.mlp.down_proj.weight").as_deref(),
            Some("blocks.0.ffn2")
        );
        assert_eq!(hf_name_to_mmn("model.embed_tokens.weight").as_deref(), Some("embed"));
    }

    #[test]
    fn hf_safetensors_roundtrip_preserves_embed() {
        let model = Chatbot::new(false, None, 64, Some(1), Some(16));
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_hf_st_test.safetensors");
        export_hf_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let loaded = import_hf_safetensors(path.to_str().unwrap()).unwrap();
        assert_eq!(
            model.embed.weight.data[[0, 0]],
            loaded.embed.weight.data[[0, 0]]
        );
    }

    #[test]
    fn is_hf_bytes_detects_binary() {
        let model = Chatbot::new(false, None, 32, Some(1), Some(8));
        let dir = std::env::temp_dir();
        let path = dir.join("mmn_hf_detect.safetensors");
        export_hf_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert!(is_hf_safetensors_bytes(&bytes));
        assert!(!is_hf_safetensors_bytes(b"{\"format\":"));
    }
}
