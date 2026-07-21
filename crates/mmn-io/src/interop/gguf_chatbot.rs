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
        "loop_embd.weight" => return Some("loop_embed.weight".into()),
        "output_norm.weight" => return Some("final_norm.gamma".into()),
        "output_norm.bias" => return Some("final_norm.beta".into()),
        // Legacy GGML single final norm.
        "norm.weight" => return Some("final_norm.gamma".into()),
        // MMN vision prefix tensors (mmproj-style `v.` namespace).
        "v.patch_proj.weight" => return Some("vision_patch_proj".into()),
        "v.patch_conv.weight" => return Some("vision_patch_conv".into()),
        "v.cross_attn_q.weight" => return Some("vision_cross_attn.q".into()),
        "v.cross_attn_k.weight" => return Some("vision_cross_attn.k".into()),
        "v.cross_attn_v.weight" => return Some("vision_cross_attn.v".into()),
        "v.cross_attn_out.weight" => return Some("vision_cross_attn.out".into()),
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
        "ffn_gate.weight" => "ffn_gate",
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
        "loop_embed.weight" => return Some("loop_embd.weight".into()),
        "final_norm.gamma" => return Some("output_norm.weight".into()),
        "final_norm.beta" => return Some("output_norm.bias".into()),
        "vision_patch_proj" => return Some("v.patch_proj.weight".into()),
        "vision_patch_conv" => return Some("v.patch_conv.weight".into()),
        "vision_cross_attn.q" => return Some("v.cross_attn_q.weight".into()),
        "vision_cross_attn.k" => return Some("v.cross_attn_k.weight".into()),
        "vision_cross_attn.v" => return Some("v.cross_attn_v.weight".into()),
        "vision_cross_attn.out" => return Some("v.cross_attn_out.weight".into()),
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
        "ffn_gate" => "ffn_gate.weight",
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
    // Qwen / modern GGUF: head_dim may differ from d_model / n_heads.
    if let Some(key_len) = meta_u64(file, &format!("{arch}.attention.key_length")) {
        meta["head_dim"] = serde_json::json!(key_len);
    } else if let Some(val_len) = meta_u64(file, &format!("{arch}.attention.value_length")) {
        meta["head_dim"] = serde_json::json!(val_len);
    } else if let (Some(n_heads), Some(q)) = (
        meta.get("num_attention_heads")
            .and_then(|v| v.as_u64())
            .map(|v| v as usize),
        tensors.get("blocks.0.attn.q"),
    ) {
        let shape = q.data.shape();
        if shape.len() == 2 && n_heads > 0 && shape[0].is_multiple_of(n_heads) {
            let inferred = shape[0] / n_heads;
            let d_model = meta
                .get("d_model")
                .and_then(|v| v.as_u64())
                .map(|v| v as usize)
                .unwrap_or(shape[1]);
            if inferred != d_model / n_heads {
                meta["head_dim"] = serde_json::json!(inferred);
            }
        }
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
    if let Some(n_loops) = meta_u64(file, &format!("{arch}.n_loops")) {
        meta["n_loops"] = serde_json::json!(n_loops);
    }
    if file
        .metadata
        .get(&format!("{arch}.tie_embeddings"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        meta["tie_embeddings"] = serde_json::json!(true);
    }
    if let Some(norm) = file
        .metadata
        .get(&format!("{arch}.norm"))
        .and_then(|v| v.as_str())
    {
        meta["norm"] = serde_json::json!(norm);
    }
    if let Some(ffn) = file
        .metadata
        .get(&format!("{arch}.ffn_kind"))
        .and_then(|v| v.as_str())
    {
        meta["ffn_kind"] = serde_json::json!(ffn);
    }
    if file
        .metadata
        .get(&format!("{arch}.loop_embed"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
    {
        meta["loop_embed"] = serde_json::json!(true);
    }
    if file
        .metadata
        .get(&format!("{arch}.final_norm"))
        .and_then(|v| v.as_bool())
        .unwrap_or(false)
        || tensors.contains_key("final_norm.gamma")
    {
        meta["final_norm"] = serde_json::json!(true);
    }
    if let Some(rank) = meta_u64(file, &format!("{arch}.lora_rank")) {
        if rank > 0 {
            meta["lora_rank"] = serde_json::json!(rank);
        }
    }
    let vision = file
        .metadata
        .get(&format!("{arch}.vision"))
        .and_then(|v| v.as_bool())
        .unwrap_or_else(|| tensors.contains_key("vision_patch_proj"));
    meta["vision"] = serde_json::json!(vision);
    meta
}

/// Dequantize one mapped tensor into an MMN `Tensor`.
fn dequantized_tensor(
    file: &super::gguf::GgufFile,
    info: &super::gguf::GgufTensorInfo,
) -> Result<Tensor, MmnError> {
    let (shape, values) = file.tensor_f32(info)?;
    let arr = ArrayD::from_shape_vec(IxDyn(&shape), values)
        .map_err(|e| err(format!("GGUF tensor {}: {e}", info.name)))?;
    Ok(Tensor::from_array(arr, true))
}

/// Import a GGUF model file into a `Chatbot` (dequantizing as needed).
///
/// Tensors dequantize in parallel across available cores — large quantized
/// checkpoints decode block-by-block, which is embarrassingly parallel.
pub fn import_gguf_bytes(bytes: &[u8]) -> Result<Chatbot, MmnError> {
    let file = read_gguf(bytes)?;
    let mapped: Vec<(String, &super::gguf::GgufTensorInfo)> = file
        .tensors
        .iter()
        .filter_map(|info| gguf_name_to_mmn(&info.name).map(|key| (key, info)))
        .collect();
    if mapped.is_empty() {
        return Err(err(
            "GGUF file contains no recognizable model tensors (expected llama.cpp names like token_embd.weight, blk.0.attn_q.weight)",
        ));
    }
    let workers = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1)
        .min(mapped.len());
    let mut tensors: HashMap<String, Tensor> = HashMap::with_capacity(mapped.len());
    if workers <= 1 {
        for (key, info) in &mapped {
            tensors.insert(key.clone(), dequantized_tensor(&file, info)?);
        }
    } else {
        let chunk_size = mapped.len().div_ceil(workers);
        let results: Vec<Result<Vec<(String, Tensor)>, MmnError>> =
            std::thread::scope(|scope| {
                let handles: Vec<_> = mapped
                    .chunks(chunk_size)
                    .map(|chunk| {
                        let file = &file;
                        scope.spawn(move || {
                            chunk
                                .iter()
                                .map(|(key, info)| {
                                    dequantized_tensor(file, info).map(|t| (key.clone(), t))
                                })
                                .collect()
                        })
                    })
                    .collect();
                handles
                    .into_iter()
                    .map(|h| h.join().expect("gguf dequant worker panicked"))
                    .collect()
            });
        for chunk in results {
            for (key, tensor) in chunk? {
                tensors.insert(key, tensor);
            }
        }
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
            "mmn.attention.key_length".to_string(),
            GgufValue::U32(model.shape.effective_head_dim() as u32),
        ),
        (
            "mmn.attention.value_length".to_string(),
            GgufValue::U32(model.shape.effective_head_dim() as u32),
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
    if model.vision {
        meta.push(("mmn.vision".to_string(), GgufValue::Bool(true)));
    }
    if model.n_loops != 1 {
        meta.push((
            "mmn.n_loops".to_string(),
            GgufValue::U32(model.n_loops as u32),
        ));
    }
    if model.tie_embeddings {
        meta.push(("mmn.tie_embeddings".to_string(), GgufValue::Bool(true)));
    }
    if model.norm_kind != "layer" {
        meta.push((
            "mmn.norm".to_string(),
            GgufValue::String(model.norm_kind.clone()),
        ));
    }
    if model.ffn_kind != "gelu" {
        meta.push((
            "mmn.ffn_kind".to_string(),
            GgufValue::String(model.ffn_kind.clone()),
        ));
    }
    if model.loop_embed.is_some() {
        meta.push(("mmn.loop_embed".to_string(), GgufValue::Bool(true)));
    }
    if model.final_norm.is_some() {
        meta.push(("mmn.final_norm".to_string(), GgufValue::Bool(true)));
    }
    if let Some(lora) = &model.loop_lora {
        meta.push((
            "mmn.lora_rank".to_string(),
            GgufValue::U32(lora.rank as u32),
        ));
    }
    meta
}

/// Export a `Chatbot` as a GGUF v3 file (`quant`: "f32", "f16", "q8_0", "q4_0").
pub fn export_gguf(model: &Chatbot, path: &str, quant: &str) -> Result<(), MmnError> {
    export_gguf_with_tokenizer(model, path, quant, None)
}

/// Export a `Chatbot` as GGUF with an embedded SentencePiece-style vocabulary
/// (`tokenizer.ggml.model = "llama"`), making the file self-contained.
pub fn export_gguf_with_tokenizer(
    model: &Chatbot,
    path: &str,
    quant: &str,
    tokenizer: Option<&mmn_data::UnigramEncoder>,
) -> Result<(), MmnError> {
    let ggml_type = match quant {
        "f32" | "F32" => GgmlType::F32,
        "f16" | "F16" => GgmlType::F16,
        "q8_0" | "Q8_0" => GgmlType::Q8_0,
        "q4_0" | "Q4_0" => GgmlType::Q4_0,
        "q4_1" | "Q4_1" => GgmlType::Q4_1,
        "q5_0" | "Q5_0" => GgmlType::Q5_0,
        "q5_1" | "Q5_1" => GgmlType::Q5_1,
        "q2_k" | "Q2_K" => GgmlType::Q2K,
        "q3_k" | "Q3_K" => GgmlType::Q3K,
        "q4_k" | "Q4_K" => GgmlType::Q4K,
        "q5_k" | "Q5_K" => GgmlType::Q5K,
        "q6_k" | "Q6_K" => GgmlType::Q6K,
        "iq4_nl" | "IQ4_NL" => GgmlType::Iq4Nl,
        "iq4_xs" | "IQ4_XS" => GgmlType::Iq4Xs,
        other => {
            return Err(err(format!(
                "GGUF export quant {other:?} not supported (use \"f32\", \"f16\", \"q8_0\", \"q4_0\", \"q4_k\", \"q5_k\", or \"q6_k\")"
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
            // ggml quantizes per row: the fastest-varying dimension must be a
            // block multiple (blocks never straddle rows). Narrow tensors
            // (layernorm vectors, small d_model) stay F32.
            let (block_elems, _) = ggml_type.block_layout();
            let row = shape.last().copied().unwrap_or(0);
            let use_quant = ggml_type != GgmlType::F32
                && row > 0
                && row.is_multiple_of(block_elems);
            GgufWriteTensor {
                name: name.clone(),
                shape: shape.clone(),
                values,
                ggml_type: if use_quant { ggml_type } else { GgmlType::F32 },
            }
        })
        .collect();
    let mut meta = chatbot_gguf_metadata(model);
    if let Some(encoder) = tokenizer {
        meta.extend(super::gguf_info::unigram_to_gguf_metadata(encoder));
    }
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
        assert_eq!(
            gguf_name_to_mmn("output_norm.weight").as_deref(),
            Some("final_norm.gamma")
        );
        assert_eq!(
            gguf_name_to_mmn("output_norm.bias").as_deref(),
            Some("final_norm.beta")
        );
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
        let e = export_gguf(&model, "/tmp/never.gguf", "iq2_xxs").unwrap_err();
        assert!(e.message().contains("not supported"));
    }

    #[test]
    fn gguf_f16_roundtrip_close() {
        let model = Chatbot::new_with_seed(false, None, 64, Some(1), Some(16), Some(6));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_f16_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "f16").unwrap();
        let loaded = import_gguf(path.to_str().unwrap()).unwrap();
        let a = model.embed.weight.data[[2, 3]];
        let b = loaded.embed.weight.data[[2, 3]];
        assert!((a - b).abs() < 1e-3, "{a} vs {b}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn gguf_q6_k_roundtrip_close() {
        // Rows must be 256-multiples for k-quants: d_model = 256.
        let model = Chatbot::new_with_seed(false, None, 64, Some(1), Some(256), Some(12));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_q6k_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "q6_k").unwrap();
        let loaded = import_gguf(path.to_str().unwrap()).unwrap();
        let a = model.embed.weight.data[[1, 2]];
        let b = loaded.embed.weight.data[[1, 2]];
        assert!((a - b).abs() < 0.05, "{a} vs {b}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn gguf_quant_falls_back_to_f32_for_narrow_rows() {
        // d_model 16 rows are not 256-multiples: k-quant export stays valid
        // by keeping every tensor F32 (per-row quantization rule).
        let model = Chatbot::new_with_seed(false, None, 64, Some(1), Some(16), Some(3));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_narrow_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "q6_k").unwrap();
        let file = super::super::gguf::read_gguf(&fs::read(&path).unwrap()).unwrap();
        assert!(file
            .tensors
            .iter()
            .all(|t| t.ggml_type == GgmlType::F32));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn gguf_q4_0_roundtrip_close() {
        let model = Chatbot::new_with_seed(false, None, 64, Some(1), Some(32), Some(8));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_q40_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "q4_0").unwrap();
        let loaded = import_gguf(path.to_str().unwrap()).unwrap();
        let a = model.embed.weight.data[[1, 2]];
        let b = loaded.embed.weight.data[[1, 2]];
        assert!((a - b).abs() < 0.25, "{a} vs {b}");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn embedded_tokenizer_roundtrips_through_gguf() {
        let model = Chatbot::new_with_seed(false, None, 512, Some(1), Some(16), Some(4));
        let enc = mmn_data::UnigramEncoder::train(&["hello world", "hello there"], 300);
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_embtok_{}.gguf", std::process::id()));
        export_gguf_with_tokenizer(&model, path.to_str().unwrap(), "f32", Some(&enc)).unwrap();
        let back = super::super::gguf_info::import_gguf_tokenizer(path.to_str().unwrap()).unwrap();
        assert_eq!(back.piece_count(), enc.piece_count());
        let text = "hello world";
        assert_eq!(back.decode(&back.encode(text)), text);
        assert_eq!(back.encode(text), enc.encode(text));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn vision_model_gguf_roundtrips() {
        let model = Chatbot::new_with_seed(true, None, 64, Some(1), Some(16), Some(9));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_vis_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "f32").unwrap();
        let loaded = import_gguf(path.to_str().unwrap()).unwrap();
        assert!(loaded.vision);
        assert!(loaded.vision_patch_proj.is_some());
        let a = model.vision_patch_proj.as_ref().unwrap().weight.data[[0, 0]];
        let b = loaded.vision_patch_proj.as_ref().unwrap().weight.data[[0, 0]];
        assert!((a - b).abs() < 1e-6);
        if model.vision_cross_attn.is_some() {
            assert!(loaded.vision_cross_attn.is_some());
        }
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn gguf_glint_arch_meta_roundtrip() {
        use mmn_models::ChatbotArchExtras;
        let extras = ChatbotArchExtras {
            n_loops: 2,
            tie_embeddings: true,
            use_rms_norm: true,
            use_swiglu: true,
            loop_embed: true,
            ..Default::default()
        };
        let model = Chatbot::new_with_arch(
            false,
            None,
            64,
            Some(1),
            Some(16),
            None,
            Some(4),
            None,
            None,
            Some(7),
            false,
            64,
            false,
            10_000.0,
            extras,
        );
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_glint_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "f32").unwrap();
        let loaded = import_gguf(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.n_loops, 2);
        assert!(loaded.tie_embeddings);
        assert_eq!(loaded.norm_kind, "rms");
        assert_eq!(loaded.ffn_kind, "swiglu");
        assert!(loaded.loop_embed.is_some());
        assert!(loaded.blocks[0].ffn_gate.is_some());
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn iq4_exports_roundtrip_close() {
        let model = Chatbot::new_with_seed(false, None, 64, Some(1), Some(256), Some(15));
        let dir = std::env::temp_dir();
        for quant in ["iq4_nl", "iq4_xs"] {
            let path = dir.join(format!("mmn_gguf_{quant}_{}.gguf", std::process::id()));
            export_gguf(&model, path.to_str().unwrap(), quant).unwrap();
            let loaded = import_gguf(path.to_str().unwrap()).unwrap();
            let a = model.embed.weight.data[[1, 2]];
            let b = loaded.embed.weight.data[[1, 2]];
            assert!((a - b).abs() < 0.2, "{quant}: {a} vs {b}");
            let _ = fs::remove_file(&path);
        }
    }

    #[test]
    fn parallel_dequant_matches_model_shape() {
        // Multi-block model exercises the multi-worker path (many tensors).
        let model = Chatbot::new_with_seed(false, None, 64, Some(4), Some(16), Some(2));
        let dir = std::env::temp_dir();
        let path = dir.join(format!("mmn_gguf_par_{}.gguf", std::process::id()));
        export_gguf(&model, path.to_str().unwrap(), "f32").unwrap();
        let loaded = import_gguf(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.shape.n_layer, 4);
        for i in 0..4 {
            let a = model.blocks[i].ffn.weight.data[[0, 0]];
            let b = loaded.blocks[i].ffn.weight.data[[0, 0]];
            assert!((a - b).abs() < 1e-6, "block {i}");
        }
        let _ = fs::remove_file(&path);
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

    /// Synthetic Qwen-style GGUF: d_model=8, n_heads=2, head_dim=8 → q [16, 8].
    #[test]
    fn import_gguf_custom_head_dim_from_key_length() {
        let d_model = 8usize;
        let n_heads = 2usize;
        let n_kv_heads = 1usize;
        let head_dim = 8usize;
        let q_dim = n_heads * head_dim;
        let kv_dim = n_kv_heads * head_dim;
        let vocab = 32usize;
        let ffn_dim = 16usize;

        let storages: Vec<(String, Vec<usize>, Vec<f32>)> = [
            ("token_embd.weight", vec![vocab, d_model], 0.01),
            ("output.weight", vec![vocab, d_model], 0.02),
            ("blk.0.attn_q.weight", vec![q_dim, d_model], 0.03),
            ("blk.0.attn_k.weight", vec![kv_dim, d_model], 0.04),
            ("blk.0.attn_v.weight", vec![kv_dim, d_model], 0.05),
            ("blk.0.attn_output.weight", vec![d_model, q_dim], 0.06),
            ("blk.0.ffn_up.weight", vec![ffn_dim, d_model], 0.07),
            ("blk.0.ffn_down.weight", vec![d_model, ffn_dim], 0.08),
            ("blk.0.attn_norm.weight", vec![d_model], 1.0),
            ("blk.0.ffn_norm.weight", vec![d_model], 1.0),
        ]
        .into_iter()
        .map(|(name, shape, fill)| {
            let n: usize = shape.iter().product();
            (name.to_string(), shape, vec![fill; n])
        })
        .collect();
        let tensors: Vec<GgufWriteTensor<'_>> = storages
            .iter()
            .map(|(name, shape, data)| GgufWriteTensor {
                name: name.clone(),
                shape: shape.clone(),
                values: data.as_slice(),
                ggml_type: GgmlType::F32,
            })
            .collect();

        let meta = vec![
            (
                "general.architecture".to_string(),
                GgufValue::String("qwen3".into()),
            ),
            (
                "qwen3.embedding_length".to_string(),
                GgufValue::U32(d_model as u32),
            ),
            ("qwen3.block_count".to_string(), GgufValue::U32(1)),
            (
                "qwen3.feed_forward_length".to_string(),
                GgufValue::U32(ffn_dim as u32),
            ),
            (
                "qwen3.attention.head_count".to_string(),
                GgufValue::U32(n_heads as u32),
            ),
            (
                "qwen3.attention.head_count_kv".to_string(),
                GgufValue::U32(n_kv_heads as u32),
            ),
            (
                "qwen3.attention.key_length".to_string(),
                GgufValue::U32(head_dim as u32),
            ),
            (
                "qwen3.attention.value_length".to_string(),
                GgufValue::U32(head_dim as u32),
            ),
            ("qwen3.rope.freq_base".to_string(), GgufValue::F32(10_000.0)),
        ];
        let bytes = write_gguf(&meta, &tensors).unwrap();
        let loaded = import_gguf_bytes(&bytes).unwrap();
        assert_eq!(loaded.shape.d_model, d_model);
        assert_eq!(loaded.shape.n_heads, n_heads);
        assert_eq!(loaded.shape.n_kv_heads, n_kv_heads);
        assert_eq!(loaded.shape.head_dim, Some(head_dim));
        assert_eq!(loaded.blocks[0].attn.head_dim, head_dim);
        assert_eq!(
            loaded.blocks[0].attn.q_proj.weight.data.shape(),
            &[q_dim, d_model]
        );
        assert_eq!(
            loaded.blocks[0].attn.out_proj.weight.data.shape(),
            &[d_model, q_dim]
        );
    }
}
