use crate::block_tensors::{export_block_tensors, import_block_tensors};
use crate::checkpoint_util::{
    expect_tensor_shape, quantize_tensor, require_tensor_entry, tensor_from_entry, tensor_to_entry,
    write_file_create_parents,
};
use crate::tensor_merge::average_tensors;
use mmn_core::MmnError;
use mmn_models::{Chatbot, DEFAULT_MAX_SEQ_LEN, DEFAULT_ROPE_THETA};
use std::collections::HashMap;
use std::fs;

/// Relative paths to tokenizer sidecars written beside a chatbot checkpoint.
#[derive(Clone, Copy, Default)]
pub struct TokenizerSidecarRefs<'a> {
    pub bpe: Option<&'a str>,
    pub unigram: Option<&'a str>,
}

impl<'a> From<Option<&'a str>> for TokenizerSidecarRefs<'a> {
    fn from(bpe: Option<&'a str>) -> Self {
        Self { bpe, unigram: None }
    }
}

pub fn export_safetensors<'a>(
    model: &Chatbot,
    path: &str,
    tokenizer_sidecars: impl Into<TokenizerSidecarRefs<'a>>,
) -> Result<(), MmnError> {
    let tokenizer_sidecars = tokenizer_sidecars.into();
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
        meta["vision_rgb_dim"] = serde_json::json!(mmn_models::VISION_RGB_DIM);
        if let Some(proj) = &model.vision_patch_proj {
            map.insert(
                "vision_patch_proj".to_string(),
                tensor_to_entry(&proj.weight),
            );
        }
        if let Some(conv) = &model.vision_patch_conv {
            meta["vision_rgb_patch"] = serde_json::json!(true);
            map.insert(
                "vision_patch_conv".to_string(),
                tensor_to_entry(&conv.weight),
            );
        }
        if let Some(cross) = &model.vision_cross_attn {
            meta["vision_cross_attn"] = serde_json::json!(true);
            map.insert(
                "vision_cross_attn.out".to_string(),
                tensor_to_entry(&cross.out_proj.weight),
            );
            map.insert(
                "vision_cross_attn.q".to_string(),
                tensor_to_entry(&cross.q_proj.weight),
            );
            map.insert(
                "vision_cross_attn.k".to_string(),
                tensor_to_entry(&cross.k_proj.weight),
            );
            map.insert(
                "vision_cross_attn.v".to_string(),
                tensor_to_entry(&cross.v_proj.weight),
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
    if model.use_rope {
        meta["use_rope"] = serde_json::json!(true);
        meta["rope_theta"] = serde_json::json!(model.rope_theta);
    }
    if model.shape.n_kv_heads != model.shape.n_heads {
        meta["n_kv_heads"] = serde_json::json!(model.shape.n_kv_heads);
        meta["num_attention_heads"] = serde_json::json!(model.shape.n_heads);
        meta["num_key_value_heads"] = serde_json::json!(model.shape.n_kv_heads);
    }
    if let Some(bpe_path) = tokenizer_sidecars.bpe {
        meta["bpe_checkpoint"] = serde_json::json!(bpe_path);
    }
    if let Some(uni_path) = tokenizer_sidecars.unigram {
        meta["unigram_checkpoint"] = serde_json::json!(uni_path);
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
    let bytes = fs::read(path).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    if crate::hf_safetensors::is_hf_safetensors_bytes(&bytes) {
        return crate::hf_safetensors::import_hf_safetensors_bytes(&bytes);
    }
    let text = std::str::from_utf8(&bytes).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    import_mmn_json_safetensors(text)
}

fn import_mmn_json_safetensors(text: &str) -> Result<Chatbot, MmnError> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| MmnError::Other {
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
    let use_rope = meta["use_rope"].as_bool().unwrap_or(false);
    let rope_theta = meta["rope_theta"]
        .as_f64()
        .unwrap_or(DEFAULT_ROPE_THETA as f64) as f32;
    let max_seq_len = meta["max_seq_len"]
        .as_u64()
        .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize;
    let n_heads = meta
        .get("n_heads")
        .or_else(|| meta.get("num_attention_heads"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let n_kv_heads = meta
        .get("n_kv_heads")
        .or_else(|| meta.get("num_key_value_heads"))
        .and_then(|v| v.as_u64())
        .map(|v| v as usize);
    let mut model = Chatbot::new_with_position_and_ffn(
        vision,
        None,
        vocab_size,
        Some(n_layer),
        Some(d_model),
        None,
        n_heads,
        n_kv_heads,
        init_seed,
        use_learned_pos_embed,
        max_seq_len,
        use_rope,
        rope_theta,
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
        if let Ok(entry) = require_tensor_entry(&v["tensors"], "vision_patch_conv") {
            let conv = model.vision_patch_conv.as_mut().ok_or_else(|| MmnError::Other {
                message: "vision_patch_conv tensor present but model has no conv encoder".into(),
            })?;
            conv.weight = tensor_from_entry(entry)?;
            expect_tensor_shape(
                &conv.weight,
                &[
                    1,
                    mmn_models::VISION_RGB_CHANNELS,
                    3,
                    3,
                ],
                "vision_patch_conv",
            )?;
        } else {
            model.vision_patch_conv = None;
        }
        if let Ok(entry) = require_tensor_entry(&v["tensors"], "vision_cross_attn.out") {
            let cross = model.vision_cross_attn.as_mut().ok_or_else(|| MmnError::Other {
                message: "vision_cross_attn tensor present but model has no cross-attn".into(),
            })?;
            cross.out_proj.weight = tensor_from_entry(entry)?;
            expect_tensor_shape(
                &cross.out_proj.weight,
                &[d_model, d_model],
                "vision_cross_attn.out",
            )?;
            cross.q_proj.weight =
                tensor_from_entry(require_tensor_entry(&v["tensors"], "vision_cross_attn.q")?)?;
            expect_tensor_shape(
                &cross.q_proj.weight,
                &[d_model, d_model],
                "vision_cross_attn.q",
            )?;
            cross.k_proj.weight =
                tensor_from_entry(require_tensor_entry(&v["tensors"], "vision_cross_attn.k")?)?;
            expect_tensor_shape(
                &cross.k_proj.weight,
                &[d_model, d_model],
                "vision_cross_attn.k",
            )?;
            cross.v_proj.weight =
                tensor_from_entry(require_tensor_entry(&v["tensors"], "vision_cross_attn.v")?)?;
            expect_tensor_shape(
                &cross.v_proj.weight,
                &[d_model, d_model],
                "vision_cross_attn.v",
            )?;
        } else {
            model.vision_cross_attn = None;
        }
    }
    import_block_tensors(&mut model, &v["tensors"])?;
    Ok(model)
}

pub fn merge_models(a: &Chatbot, b: &Chatbot) -> Result<Chatbot, MmnError> {
    if a.shape.vocab_size != b.shape.vocab_size
        || a.shape.d_model != b.shape.d_model
        || a.shape.n_layer != b.shape.n_layer
        || a.shape.n_heads != b.shape.n_heads
        || a.shape.n_kv_heads != b.shape.n_kv_heads
    {
        return Err(MmnError::ModelMismatch {
            message: "Cannot merge models of different sizes".into(),
            fix: "Use two models with the same autoset budget and architecture.".into(),
            explanation: "merge() requires matching vocab size, layer counts, and hidden sizes."
                .into(),
        });
    }
    if a.use_learned_pos_embed != b.use_learned_pos_embed
        || a.max_seq_len != b.max_seq_len
        || a.use_rope != b.use_rope
        || (a.use_rope && (a.rope_theta - b.rope_theta).abs() > 1e-3)
    {
        return Err(MmnError::ModelMismatch {
            message: "Cannot merge models with different position-embedding settings".into(),
            fix: "Use two models with the same use_learned_pos_embed, use_rope, max_seq_len, and rope_theta.".into(),
            explanation: "merge() requires matching position embedding configuration.".into(),
        });
    }
    let mut out = Chatbot::new_with_position_and_ffn(
        a.vision || b.vision,
        None,
        a.shape.vocab_size,
        Some(a.shape.n_layer),
        Some(a.shape.d_model),
        Some(a.shape.ffn_dim),
        Some(a.shape.n_heads),
        Some(a.shape.n_kv_heads),
        a.init_seed.or(b.init_seed),
        a.use_learned_pos_embed,
        a.max_seq_len,
        a.use_rope,
        a.rope_theta,
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
    if let Some(out_conv) = out.vision_patch_conv.as_mut() {
        match (&a.vision_patch_conv, &b.vision_patch_conv) {
            (Some(aa), Some(bb)) => {
                out_conv.weight = average_tensors(&aa.weight, &bb.weight);
            }
            (Some(aa), None) => out_conv.weight = aa.weight.clone(),
            (None, Some(bb)) => out_conv.weight = bb.weight.clone(),
            (None, None) => {}
        }
    }
    if let Some(out_cross) = out.vision_cross_attn.as_mut() {
        match (&a.vision_cross_attn, &b.vision_cross_attn) {
            (Some(aa), Some(bb)) => {
                out_cross.out_proj.weight =
                    average_tensors(&aa.out_proj.weight, &bb.out_proj.weight);
                out_cross.q_proj.weight =
                    average_tensors(&aa.q_proj.weight, &bb.q_proj.weight);
                out_cross.k_proj.weight =
                    average_tensors(&aa.k_proj.weight, &bb.k_proj.weight);
                out_cross.v_proj.weight =
                    average_tensors(&aa.v_proj.weight, &bb.v_proj.weight);
            }
            (Some(aa), None) => {
                out_cross.out_proj.weight = aa.out_proj.weight.clone();
                out_cross.q_proj.weight = aa.q_proj.weight.clone();
                out_cross.k_proj.weight = aa.k_proj.weight.clone();
                out_cross.v_proj.weight = aa.v_proj.weight.clone();
            }
            (None, Some(bb)) => {
                out_cross.out_proj.weight = bb.out_proj.weight.clone();
                out_cross.q_proj.weight = bb.q_proj.weight.clone();
                out_cross.k_proj.weight = bb.k_proj.weight.clone();
                out_cross.v_proj.weight = bb.v_proj.weight.clone();
            }
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
            if let Some(conv) = &mut model.vision_patch_conv {
                quantize_tensor(&mut conv.weight, scale);
            }
            if let Some(cross) = &mut model.vision_cross_attn {
                quantize_tensor(&mut cross.out_proj.weight, scale);
                quantize_tensor(&mut cross.q_proj.weight, scale);
                quantize_tensor(&mut cross.k_proj.weight, scale);
                quantize_tensor(&mut cross.v_proj.weight, scale);
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
        "n_heads": model.shape.n_heads,
        "n_kv_heads": model.shape.n_kv_heads,
        "ffn_dim": model.shape.ffn_dim,
        "vision": model.vision,
    });
    if model.vision {
        json["vision_patch_dim"] = serde_json::json!(mmn_models::VISION_PATCH_DIM);
        json["vision_rgb_dim"] = serde_json::json!(mmn_models::VISION_RGB_DIM);
        if model.vision_patch_conv.is_some() {
            json["vision_rgb_patch"] = serde_json::json!(true);
        }
        if model.vision_cross_attn.is_some() {
            json["vision_cross_attn"] = serde_json::json!(true);
        }
    }
    if model.use_learned_pos_embed {
        json["use_learned_pos_embed"] = serde_json::json!(true);
        json["max_seq_len"] = serde_json::json!(model.max_seq_len);
    }
    if model.use_rope {
        json["use_rope"] = serde_json::json!(true);
        json["rope_theta"] = serde_json::json!(model.rope_theta);
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
    let n_heads = v
        .get("n_heads")
        .or_else(|| v.get("num_attention_heads"))
        .and_then(|m| m.as_u64())
        .map(|n| n as usize);
    let n_kv_heads = v
        .get("n_kv_heads")
        .or_else(|| v.get("num_key_value_heads"))
        .and_then(|m| m.as_u64())
        .map(|n| n as usize);
    let ffn_dim = v["ffn_dim"].as_u64().map(|n| n as usize);
    let vision = v["vision"].as_bool().unwrap_or(false);
    let use_learned_pos_embed = v["use_learned_pos_embed"].as_bool().unwrap_or(false);
    let use_rope = v["use_rope"].as_bool().unwrap_or(false);
    let rope_theta = v["rope_theta"]
        .as_f64()
        .unwrap_or(DEFAULT_ROPE_THETA as f64) as f32;
    let max_seq_len = v["max_seq_len"]
        .as_u64()
        .unwrap_or(DEFAULT_MAX_SEQ_LEN as u64) as usize;
    Ok(Chatbot::new_with_position_and_ffn(
        vision,
        None,
        vocab,
        Some(n_layer),
        Some(d_model),
        ffn_dim,
        n_heads,
        n_kv_heads,
        None,
        use_learned_pos_embed,
        max_seq_len,
        use_rope,
        rope_theta,
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
