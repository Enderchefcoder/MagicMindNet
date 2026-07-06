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
    if model.vision {
        return Err(err(
            "GGUF export does not support vision models yet; use safetensors or npz",
        ));
    }
    let ggml_type = match quant {
        "f32" | "F32" => GgmlType::F32,
        "f16" | "F16" => GgmlType::F16,
        "q8_0" | "Q8_0" => GgmlType::Q8_0,
        "q4_0" | "Q4_0" => GgmlType::Q4_0,
        "q4_1" | "Q4_1" => GgmlType::Q4_1,
        "q5_0" | "Q5_0" => GgmlType::Q5_0,
        "q5_1" | "Q5_1" => GgmlType::Q5_1,
        "q4_k" | "Q4_K" => GgmlType::Q4K,
        "q5_k" | "Q5_K" => GgmlType::Q5K,
        "q6_k" | "Q6_K" => GgmlType::Q6K,
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
}
