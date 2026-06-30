use crate::block_tensors::{export_block_tensors, import_block_tensors};
use crate::checkpoint_util::{
    expect_tensor_shape, quantize_tensor, require_tensor_entry, tensor_from_entry, tensor_to_entry,
    write_file_create_parents,
};
use crate::tensor_merge::average_tensors;
use mmn_core::MmnError;
use mmn_models::{Chatbot, DEFAULT_MAX_SEQ_LEN};
use std::collections::HashMap;
use std::fs;

pub fn export_safetensors(
    model: &Chatbot,
    path: &str,
    bpe_checkpoint: Option<&str>,
) -> Result<(), MmnError> {
    let mut map = HashMap::new();
    map.insert("embed".to_string(), tensor_to_entry(&model.embed.weight));
    map.insert("lm_head".to_string(), tensor_to_entry(&model.lm_head.weight));
    export_block_tensors(model, &mut map);
    let mut meta = serde_json::json!({
        "vocab_size": model.shape.vocab_size,
        "n_layer": model.shape.n_layer,
        "d_model": model.shape.d_model,
        "vision": model.vision,
    });
    if model.vision {
        meta["vision_patch_dim"] = serde_json::json!(mmn_models::VISION_PATCH_DIM);
        if let Some(proj) = &model.vision_patch_proj {
            map.insert(
                "vision_patch_proj".to_string(),
                tensor_to_entry(&proj.weight),
            );
        }
    }
    if let Some(seed) = model.init_seed {
        meta["seed"] = serde_json::json!(seed);
    }
    if model.use_learned_pos_embed {
        meta["use_learned_pos_embed"] = serde_json::json!(true);
        meta["max_seq_len"] = serde_json::json!(model.max_seq_len);
        map.insert(
            "pos_embed".to_string(),
            tensor_to_entry(&model.pos_embed.as_ref().unwrap().weight),
        );
    }
    if let Some(bpe_path) = bpe_checkpoint {
        meta["bpe_checkpoint"] = serde_json::json!(bpe_path);
    }
    let wrapper = serde_json::json!({
        "tensors": map,
        "format": "mmn-safetensors-v1",
        "meta": meta,
    });
    write_file_create_parents(path, wrapper.to_string())?;
    Ok(())
}

pub fn import_safetensors(path: &str, _vocab_size: usize) -> Result<Chatbot, MmnError> {
    let text = fs::read_to_string(path).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    if v["format"].as_str() != Some("mmn-safetensors-v1") {
        let got = v["format"].as_str().unwrap_or("<missing>");
        return Err(MmnError::Other {
            message: format!("Expected mmn-safetensors-v1 checkpoint, got {got}"),
        });
    }
    let meta = &v["meta"];
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
    let max_seq_len = meta["max_seq_len"]
        .as_u64()
        .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize;
    let mut model = Chatbot::new_with_pe_options(
        vision,
        None,
        vocab_size,
        Some(n_layer),
        Some(d_model),
        init_seed,
        use_learned_pos_embed,
        max_seq_len,
    );
    model.embed.weight = tensor_from_entry(require_tensor_entry(&v["tensors"], "embed")?)?;
    model.lm_head.weight = tensor_from_entry(require_tensor_entry(&v["tensors"], "lm_head")?)?;
    expect_tensor_shape(&model.embed.weight, &[vocab_size, d_model], "embed")?;
    expect_tensor_shape(&model.lm_head.weight, &[vocab_size, d_model], "lm_head")?;
    if use_learned_pos_embed {
        let pe = model.pos_embed.as_mut().ok_or_else(|| MmnError::Other {
            message: "use_learned_pos_embed meta set but model has no pos_embed".into(),
        })?;
        pe.weight = tensor_from_entry(require_tensor_entry(&v["tensors"], "pos_embed")?)?;
        expect_tensor_shape(&pe.weight, &[max_seq_len, d_model], "pos_embed")?;
    }
    if vision {
        if let Some(proj) = model.vision_patch_proj.as_mut() {
            if let Ok(entry) = require_tensor_entry(&v["tensors"], "vision_patch_proj") {
                proj.weight = tensor_from_entry(entry)?;
                expect_tensor_shape(
                    &proj.weight,
                    &[d_model, mmn_models::VISION_PATCH_DIM],
                    "vision_patch_proj",
                )?;
            }
        }
    }
    import_block_tensors(&mut model, &v["tensors"])?;
    Ok(model)
}

pub fn merge_models(a: &Chatbot, b: &Chatbot) -> Result<Chatbot, MmnError> {
    if a.shape.vocab_size != b.shape.vocab_size
        || a.shape.d_model != b.shape.d_model
        || a.shape.n_layer != b.shape.n_layer
    {
        return Err(MmnError::ModelMismatch {
            message: "Cannot merge models of different sizes".into(),
            fix: "Use two models with the same autoset budget and architecture.".into(),
            explanation: "merge() requires matching vocab size, layer counts, and hidden sizes."
                .into(),
        });
    }
    if a.use_learned_pos_embed != b.use_learned_pos_embed || a.max_seq_len != b.max_seq_len {
        return Err(MmnError::ModelMismatch {
            message: "Cannot merge models with different position-embedding settings".into(),
            fix: "Use two models with the same use_learned_pos_embed and max_seq_len.".into(),
            explanation: "merge() requires matching position embedding configuration.".into(),
        });
    }
    let mut out = Chatbot::new_with_pe_options(
        a.vision || b.vision,
        None,
        a.shape.vocab_size,
        Some(a.shape.n_layer),
        Some(a.shape.d_model),
        a.init_seed.or(b.init_seed),
        a.use_learned_pos_embed,
        a.max_seq_len,
    );
    out.embed.weight = average_tensors(&a.embed.weight, &b.embed.weight);
    out.lm_head.weight = average_tensors(&a.lm_head.weight, &b.lm_head.weight);
    if a.use_learned_pos_embed {
        let pe = out.pos_embed.as_mut().unwrap();
        pe.weight = average_tensors(
            &a.pos_embed.as_ref().unwrap().weight,
            &b.pos_embed.as_ref().unwrap().weight,
        );
    }
    if let Some(out_proj) = out.vision_patch_proj.as_mut() {
        match (&a.vision_patch_proj, &b.vision_patch_proj) {
            (Some(aa), Some(bb)) => {
                out_proj.weight = average_tensors(&aa.weight, &bb.weight);
            }
            (Some(aa), None) => out_proj.weight = aa.weight.clone(),
            (None, Some(bb)) => out_proj.weight = bb.weight.clone(),
            (None, None) => {}
        }
    }
    for (i, block) in out.blocks.iter_mut().enumerate() {
        let ab = &a.blocks[i];
        let bb = &b.blocks[i];
        block.attn.q_proj.weight = average_tensors(&ab.attn.q_proj.weight, &bb.attn.q_proj.weight);
        block.attn.k_proj.weight = average_tensors(&ab.attn.k_proj.weight, &bb.attn.k_proj.weight);
        block.attn.v_proj.weight = average_tensors(&ab.attn.v_proj.weight, &bb.attn.v_proj.weight);
        block.attn.out_proj.weight =
            average_tensors(&ab.attn.out_proj.weight, &bb.attn.out_proj.weight);
        block.ffn.weight = average_tensors(&ab.ffn.weight, &bb.ffn.weight);
        block.ffn2.weight = average_tensors(&ab.ffn2.weight, &bb.ffn2.weight);
        block.ln1.gamma = average_tensors(&ab.ln1.gamma, &bb.ln1.gamma);
        block.ln1.beta = average_tensors(&ab.ln1.beta, &bb.ln1.beta);
        block.ln2.gamma = average_tensors(&ab.ln2.gamma, &bb.ln2.gamma);
        block.ln2.beta = average_tensors(&ab.ln2.beta, &bb.ln2.beta);
    }
    out.init_seed = a.init_seed.or(b.init_seed);
    Ok(out)
}

pub fn quantize_model(model: &mut Chatbot, mode: &str) -> Result<(), MmnError> {
    match mode {
        "int8" | "int4" => {
            let scale = if mode == "int8" { 127.0 } else { 15.0 };
            quantize_tensor(&mut model.embed.weight, scale);
            quantize_tensor(&mut model.lm_head.weight, scale);
            if let Some(pe) = &mut model.pos_embed {
                quantize_tensor(&mut pe.weight, scale);
            }
            if let Some(proj) = &mut model.vision_patch_proj {
                quantize_tensor(&mut proj.weight, scale);
            }
            for block in &mut model.blocks {
                quantize_tensor(&mut block.attn.q_proj.weight, scale);
                quantize_tensor(&mut block.attn.k_proj.weight, scale);
                quantize_tensor(&mut block.attn.v_proj.weight, scale);
                quantize_tensor(&mut block.attn.out_proj.weight, scale);
                quantize_tensor(&mut block.ffn.weight, scale);
                quantize_tensor(&mut block.ffn2.weight, scale);
                quantize_tensor(&mut block.ln1.gamma, scale);
                quantize_tensor(&mut block.ln1.beta, scale);
                quantize_tensor(&mut block.ln2.gamma, scale);
                quantize_tensor(&mut block.ln2.beta, scale);
            }
            Ok(())
        }
        _ => Err(MmnError::Other {
            message: format!("Unknown quant mode: {mode}"),
        }),
    }
}

pub fn export_bin(model: &Chatbot, path: &str) -> Result<(), MmnError> {
    let mut json = serde_json::json!({
        "format": "mmn-bin-v1",
        "vocab_size": model.shape.vocab_size,
        "d_model": model.shape.d_model,
        "n_layer": model.shape.n_layer,
        "vision": model.vision,
    });
    if model.vision {
        json["vision_patch_dim"] = serde_json::json!(mmn_models::VISION_PATCH_DIM);
    }
    if model.use_learned_pos_embed {
        json["use_learned_pos_embed"] = serde_json::json!(true);
        json["max_seq_len"] = serde_json::json!(model.max_seq_len);
    }
    write_file_create_parents(path, json.to_string())?;
    Ok(())
}

pub fn import_bin(path: &str) -> Result<Chatbot, MmnError> {
    let text = fs::read_to_string(path).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    if let Some(fmt) = v["format"].as_str() {
        if fmt != "mmn-bin-v1" {
            return Err(MmnError::Other {
                message: format!("Expected mmn-bin-v1 checkpoint, got {fmt}"),
            });
        }
    } else if v["tensors"].is_object() || v["meta"].is_object() {
        return Err(MmnError::Other {
            message: "Expected mmn-bin-v1 architecture stub; full weights use mmn-safetensors-v1"
                .into(),
        });
    }
    let vocab = v["vocab_size"].as_u64().unwrap_or(32000) as usize;
    let d_model = v["d_model"].as_u64().unwrap_or(128) as usize;
    let n_layer = v["n_layer"].as_u64().unwrap_or(4) as usize;
    let vision = v["vision"].as_bool().unwrap_or(false);
    let use_learned_pos_embed = v["use_learned_pos_embed"].as_bool().unwrap_or(false);
    let max_seq_len = v["max_seq_len"]
        .as_u64()
        .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize;
    Ok(Chatbot::new_with_pe_options(
        vision,
        None,
        vocab,
        Some(n_layer),
        Some(d_model),
        None,
        use_learned_pos_embed,
        max_seq_len,
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn safetensors_roundtrip_preserves_embed_weight() {
        let model = Chatbot::new(false, None, 64, Some(1), Some(16));
        let path = std::env::temp_dir().join(format!(
            "chatbot_io_rt_{}_{}",
            "embed",
            std::process::id()
        ));
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 64).unwrap();
        let w0 = model.embed.weight.data[[0, 0]];
        let w1 = loaded.embed.weight.data[[0, 0]];
        assert!((w0 - w1).abs() < 1e-6);
        let _ = std::fs::remove_file(&path);
    }
}
