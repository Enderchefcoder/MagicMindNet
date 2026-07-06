//! GGUF inspection and tokenizer extraction: metadata as JSON, tensor
//! summaries, and SentencePiece-style vocabularies as `UnigramEncoder`s.

use super::gguf::{read_gguf_header_file, GgufHeader, GgufValue};
use mmn_core::MmnError;
use mmn_data::UnigramEncoder;

fn err(message: impl Into<String>) -> MmnError {
    MmnError::Other {
        message: message.into(),
    }
}

/// Convert a GGUF metadata value to JSON (arrays convert element-wise).
pub fn gguf_value_to_json(value: &GgufValue) -> serde_json::Value {
    match value {
        GgufValue::U8(v) => serde_json::json!(v),
        GgufValue::I8(v) => serde_json::json!(v),
        GgufValue::U16(v) => serde_json::json!(v),
        GgufValue::I16(v) => serde_json::json!(v),
        GgufValue::U32(v) => serde_json::json!(v),
        GgufValue::I32(v) => serde_json::json!(v),
        GgufValue::F32(v) => serde_json::json!(v),
        GgufValue::Bool(v) => serde_json::json!(v),
        GgufValue::String(s) => serde_json::json!(s),
        GgufValue::Array(items) => {
            serde_json::Value::Array(items.iter().map(gguf_value_to_json).collect())
        }
        GgufValue::U64(v) => serde_json::json!(v),
        GgufValue::I64(v) => serde_json::json!(v),
        GgufValue::F64(v) => serde_json::json!(v),
    }
}

/// Full-file inspection report: metadata, tensor summaries, sizes.
pub fn gguf_info_json(path: &str) -> Result<serde_json::Value, MmnError> {
    let header = read_gguf_header_file(path)?;
    let mut metadata = serde_json::Map::new();
    let mut keys: Vec<&String> = header.metadata.keys().collect();
    keys.sort();
    for key in keys {
        metadata.insert(key.clone(), gguf_value_to_json(&header.metadata[key]));
    }
    let tensors: Vec<serde_json::Value> = header
        .tensors
        .iter()
        .map(|t| {
            serde_json::json!({
                "name": t.name,
                "shape": t.row_major_shape(),
                "type": format!("{:?}", t.ggml_type),
                "offset": t.offset,
            })
        })
        .collect();
    Ok(serde_json::json!({
        "version": header.version,
        "alignment": header.alignment,
        "tensor_count": header.tensors.len(),
        "metadata": metadata,
        "tensors": tensors,
    }))
}

fn string_array(header: &GgufHeader, key: &str) -> Option<Vec<String>> {
    match header.metadata.get(key)? {
        GgufValue::Array(items) => Some(
            items
                .iter()
                .filter_map(|v| v.as_str().map(str::to_string))
                .collect(),
        ),
        _ => None,
    }
}

fn f32_array(header: &GgufHeader, key: &str) -> Option<Vec<f32>> {
    match header.metadata.get(key)? {
        GgufValue::Array(items) => Some(
            items
                .iter()
                .filter_map(|v| v.as_f64().map(|f| f as f32))
                .collect(),
        ),
        _ => None,
    }
}

/// Decode one SentencePiece token into raw bytes.
///
/// `▁` (U+2581) marks a leading space; `<0xNN>` tokens are byte fallbacks.
pub fn sentencepiece_token_bytes(token: &str) -> Vec<u8> {
    if token.len() == 6 && token.starts_with("<0x") && token.ends_with('>') {
        if let Ok(byte) = u8::from_str_radix(&token[3..5], 16) {
            return vec![byte];
        }
    }
    token.replace('\u{2581}', " ").into_bytes()
}

/// Encode raw piece bytes as a SentencePiece token string (inverse of
/// [`sentencepiece_token_bytes`]): spaces become `▁`, non-UTF-8 or control
/// single bytes become `<0xNN>` fallbacks.
pub fn sentencepiece_token_from_bytes(piece: &[u8]) -> String {
    if piece.len() == 1 && (piece[0] < 0x20 || piece[0] >= 0x7F) {
        return format!("<0x{:02X}>", piece[0]);
    }
    match std::str::from_utf8(piece) {
        Ok(s) => s.replace(' ', "\u{2581}"),
        Err(_) if piece.len() == 1 => format!("<0x{:02X}>", piece[0]),
        Err(_) => String::from_utf8_lossy(piece).replace(' ', "\u{2581}"),
    }
}

/// GGUF metadata entries embedding a unigram vocabulary
/// (`tokenizer.ggml.model = "llama"` convention).
pub fn unigram_to_gguf_metadata(encoder: &UnigramEncoder) -> Vec<(String, GgufValue)> {
    let (pieces, scores) = encoder.pieces_and_scores();
    let tokens: Vec<GgufValue> = pieces
        .iter()
        .map(|p| GgufValue::String(sentencepiece_token_from_bytes(p)))
        .collect();
    let scores: Vec<GgufValue> = scores.into_iter().map(GgufValue::F32).collect();
    vec![
        (
            "tokenizer.ggml.model".to_string(),
            GgufValue::String("llama".into()),
        ),
        ("tokenizer.ggml.tokens".to_string(), GgufValue::Array(tokens)),
        ("tokenizer.ggml.scores".to_string(), GgufValue::Array(scores)),
    ]
}

/// Build a `UnigramEncoder` from a GGUF file's embedded SentencePiece
/// vocabulary (`tokenizer.ggml.model == "llama"`), preserving token ids.
pub fn import_gguf_tokenizer(path: &str) -> Result<UnigramEncoder, MmnError> {
    let header = read_gguf_header_file(path)?;
    let model = header
        .metadata
        .get("tokenizer.ggml.model")
        .and_then(|v| v.as_str())
        .ok_or_else(|| {
            err("GGUF file has no tokenizer.ggml.model metadata (no embedded tokenizer)")
        })?;
    if model != "llama" {
        return Err(err(format!(
            "GGUF tokenizer model {model:?} is not SentencePiece-unigram (\"llama\"); use gguf_info to inspect the raw vocabulary"
        )));
    }
    let tokens = string_array(&header, "tokenizer.ggml.tokens")
        .ok_or_else(|| err("GGUF file missing tokenizer.ggml.tokens array"))?;
    if tokens.is_empty() {
        return Err(err("GGUF tokenizer.ggml.tokens array is empty"));
    }
    let scores = f32_array(&header, "tokenizer.ggml.scores")
        .unwrap_or_else(|| vec![0.0; tokens.len()]);
    if scores.len() != tokens.len() {
        return Err(err(format!(
            "GGUF tokenizer has {} tokens but {} scores",
            tokens.len(),
            scores.len()
        )));
    }
    let pieces: Vec<Vec<u8>> = tokens.iter().map(|t| sentencepiece_token_bytes(t)).collect();
    UnigramEncoder::from_pieces(pieces, scores)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::interop::gguf::{write_gguf, GgufWriteTensor};
    use crate::interop::gguf_quant::GgmlType;
    use std::fs;

    fn write_sample(path: &std::path::Path, extra_meta: Vec<(String, GgufValue)>) {
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let tensors = vec![GgufWriteTensor {
            name: "token_embd.weight".into(),
            shape: vec![4, 2],
            values: &values,
            ggml_type: GgmlType::F32,
        }];
        let mut meta = vec![(
            "general.architecture".to_string(),
            GgufValue::String("mmn".into()),
        )];
        meta.extend(extra_meta);
        let bytes = write_gguf(&meta, &tensors).unwrap();
        fs::write(path, bytes).unwrap();
    }

    #[test]
    fn info_reports_metadata_and_tensors() {
        let path = std::env::temp_dir().join(format!("mmn_gguf_info_{}.gguf", std::process::id()));
        write_sample(
            &path,
            vec![
                ("mmn.block_count".to_string(), GgufValue::U32(3)),
                (
                    "tokenizer.chat_template".to_string(),
                    GgufValue::String("{{ messages }}".into()),
                ),
            ],
        );
        let info = gguf_info_json(path.to_str().unwrap()).unwrap();
        assert_eq!(info["version"], 3);
        assert_eq!(info["tensor_count"], 1);
        assert_eq!(info["metadata"]["mmn.block_count"], 3);
        assert_eq!(info["metadata"]["tokenizer.chat_template"], "{{ messages }}");
        assert_eq!(info["tensors"][0]["name"], "token_embd.weight");
        assert_eq!(info["tensors"][0]["shape"][0], 4);
        assert_eq!(info["tensors"][0]["type"], "F32");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn sentencepiece_bytes_decode() {
        assert_eq!(sentencepiece_token_bytes("\u{2581}hello"), b" hello");
        assert_eq!(sentencepiece_token_bytes("<0x0A>"), vec![0x0A]);
        assert_eq!(sentencepiece_token_bytes("<s>"), b"<s>");
        assert_eq!(sentencepiece_token_bytes("plain"), b"plain");
    }

    #[test]
    fn tokenizer_extraction_preserves_ids_and_roundtrips() {
        let path =
            std::env::temp_dir().join(format!("mmn_gguf_tok_{}.gguf", std::process::id()));
        let tokens = vec![
            GgufValue::String("<unk>".into()),
            GgufValue::String("<s>".into()),
            GgufValue::String("\u{2581}hello".into()),
            GgufValue::String("\u{2581}world".into()),
            GgufValue::String("h".into()),
            GgufValue::String("e".into()),
            GgufValue::String("l".into()),
            GgufValue::String("o".into()),
            GgufValue::String("\u{2581}".into()),
            GgufValue::String("w".into()),
            GgufValue::String("r".into()),
            GgufValue::String("d".into()),
        ];
        let scores: Vec<GgufValue> = (0..tokens.len())
            .map(|i| GgufValue::F32(-(i as f32)))
            .collect();
        write_sample(
            &path,
            vec![
                (
                    "tokenizer.ggml.model".to_string(),
                    GgufValue::String("llama".into()),
                ),
                ("tokenizer.ggml.tokens".to_string(), GgufValue::Array(tokens)),
                ("tokenizer.ggml.scores".to_string(), GgufValue::Array(scores)),
            ],
        );
        let enc = import_gguf_tokenizer(path.to_str().unwrap()).unwrap();
        assert_eq!(enc.piece_count(), 12);
        let ids = enc.encode(" hello world");
        assert!(ids.contains(&2), "expected ▁hello id 2, got {ids:?}");
        assert!(ids.contains(&3), "expected ▁world id 3, got {ids:?}");
        assert_eq!(enc.decode(&[2, 3]), " hello world");
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn tokenizer_missing_metadata_errors() {
        let path =
            std::env::temp_dir().join(format!("mmn_gguf_notok_{}.gguf", std::process::id()));
        write_sample(&path, vec![]);
        let e = import_gguf_tokenizer(path.to_str().unwrap()).err().unwrap();
        assert!(e.message().contains("no embedded tokenizer"));
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn tokenizer_non_llama_model_errors() {
        let path =
            std::env::temp_dir().join(format!("mmn_gguf_bpe_{}.gguf", std::process::id()));
        write_sample(
            &path,
            vec![(
                "tokenizer.ggml.model".to_string(),
                GgufValue::String("gpt2".into()),
            )],
        );
        let e = import_gguf_tokenizer(path.to_str().unwrap()).err().unwrap();
        assert!(e.message().contains("gpt2"));
        let _ = fs::remove_file(&path);
    }
}
