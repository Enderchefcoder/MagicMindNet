//! Hugging Face binary safetensors interchange for `Classifier`.

use crate::checkpoint_util::{
    expect_tensor_shape, require_tensor_entry, tensor_from_entry, tensor_to_entry,
    write_file_create_parents,
};
use crate::hf_tensor_codec::{hf_err, tensor_bytes_f32, tensor_from_view};
use mmn_core::{MmnError, Tensor};
use mmn_models::Classifier;
use safetensors::tensor::{Dtype, TensorView};
use safetensors::{serialize, SafeTensors};
use std::collections::HashMap;
use std::fs;

pub const HF_CLASSIFIER_FORMAT: &str = crate::hf_tensor_codec::HF_CLASSIFIER_FORMAT;

const CLASSIFIER_HIDDEN: usize = 128;

fn classifier_meta_json(model: &Classifier) -> serde_json::Value {
    let mut meta = serde_json::json!({
        "input_dim": model.input_dim,
        "labels": model.labels,
    });
    if let Some(seed) = model.init_seed {
        meta["seed"] = serde_json::json!(seed);
    }
    meta
}

/// Map HF / MMN classifier tensor names to canonical keys (`backbone`, `head`).
pub fn hf_classifier_name_to_mmn(name: &str) -> Option<&'static str> {
    match name {
        "backbone" | "backbone.weight" | "encoder.weight" => Some("backbone"),
        "head" | "head.weight" | "classifier.weight" | "score.weight" => Some("head"),
        _ => None,
    }
}

/// Write a Hugging Face binary safetensors classifier checkpoint (F32 tensors).
pub fn export_hf_classifier_safetensors(model: &Classifier, path: &str) -> Result<(), MmnError> {
    let named: HashMap<&str, &Tensor> = HashMap::from([
        ("backbone", &model.backbone.weight),
        ("head", &model.head.weight),
    ]);
    let mut storages: Vec<(String, Vec<usize>, Vec<u8>)> = Vec::new();
    for (k, t) in &named {
        let (shape, bytes) = tensor_bytes_f32(t);
        storages.push((k.to_string(), shape, bytes));
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
    let meta = classifier_meta_json(model);
    let mut metadata = HashMap::new();
    metadata.insert("format".to_string(), HF_CLASSIFIER_FORMAT.to_string());
    metadata.insert("meta".to_string(), meta.to_string());
    let bytes = serialize(views, Some(metadata)).map_err(hf_err)?;
    write_file_create_parents(path, bytes)
}

pub fn import_hf_classifier_safetensors(path: &str) -> Result<Classifier, MmnError> {
    let bytes = fs::read(path).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    import_hf_classifier_safetensors_bytes(&bytes)
}

pub fn import_hf_classifier_safetensors_bytes(bytes: &[u8]) -> Result<Classifier, MmnError> {
    let (_header_len, header_meta) = SafeTensors::read_metadata(bytes).map_err(hf_err)?;
    let st = SafeTensors::deserialize(bytes).map_err(hf_err)?;
    let file_meta = header_meta.metadata();
    if let Some(m) = file_meta.as_ref() {
        if let Some(fmt) = m.get("format") {
            if fmt == crate::hf_tensor_codec::HF_CHATBOT_FORMAT {
                return Err(MmnError::Other {
                    message: format!(
                        "Expected {HF_CLASSIFIER_FORMAT}, got chatbot format {fmt}"
                    ),
                });
            }
            if fmt != HF_CLASSIFIER_FORMAT {
                return Err(MmnError::Other {
                    message: format!("Expected {HF_CLASSIFIER_FORMAT} metadata format, got {fmt}"),
                });
            }
        }
    }
    let meta = file_meta
        .as_ref()
        .and_then(|m| m.get("meta"))
        .map(|meta_str| {
            serde_json::from_str(meta_str).map_err(|e| MmnError::Other {
                message: format!("invalid HF classifier safetensors meta JSON: {e}"),
            })
        })
        .transpose()?
        .unwrap_or_else(|| serde_json::json!({}));
    let mut mmn_tensors: HashMap<String, Tensor> = HashMap::new();
    for name in st.names() {
        if let Some(mmn_key) = hf_classifier_name_to_mmn(name) {
            let view = st.tensor(name).map_err(hf_err)?;
            let t = tensor_from_view(name, &view)?;
            mmn_tensors.insert(mmn_key.to_string(), t);
        }
    }
    load_classifier_from_tensors(mmn_tensors, &meta)
}

fn tensors_to_json_map(tensors: &HashMap<String, Tensor>) -> serde_json::Value {
    let mut map = serde_json::Map::new();
    for (k, t) in tensors {
        map.insert(k.clone(), tensor_to_entry(t));
    }
    serde_json::Value::Object(map)
}

fn load_classifier_from_tensors(
    tensors: HashMap<String, Tensor>,
    meta: &serde_json::Value,
) -> Result<Classifier, MmnError> {
    let labels: Vec<String> = if let Some(arr) = meta["labels"].as_array() {
        arr.iter()
            .filter_map(|x| x.as_str().map(String::from))
            .collect()
    } else {
        Vec::new()
    };
    let input_dim = meta["input_dim"].as_u64().map(|v| v as usize);
    let json_tensors = tensors_to_json_map(&tensors);
    let backbone_entry = require_tensor_entry(&json_tensors, "backbone")?;
    let head_entry = require_tensor_entry(&json_tensors, "head")?;
    let backbone_shape: Vec<usize> = backbone_entry["shape"]
        .as_array()
        .ok_or_else(|| MmnError::Other {
            message: "backbone missing shape".into(),
        })?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();
    let head_shape: Vec<usize> = head_entry["shape"]
        .as_array()
        .ok_or_else(|| MmnError::Other {
            message: "head missing shape".into(),
        })?
        .iter()
        .filter_map(|v| v.as_u64().map(|n| n as usize))
        .collect();
    if backbone_shape.len() != 2 || head_shape.len() != 2 {
        return Err(MmnError::Other {
            message: "classifier backbone/head must be rank-2".into(),
        });
    }
    let resolved_input_dim = input_dim.unwrap_or(backbone_shape[1]);
    let resolved_labels = if labels.is_empty() {
        if head_shape[0] == 0 {
            return Err(MmnError::Other {
                message: "classifier meta missing labels and head shape is empty".into(),
            });
        }
        (0..head_shape[0])
            .map(|i| format!("label_{i}"))
            .collect()
    } else {
        labels
    };
    if resolved_labels.is_empty() {
        return Err(MmnError::Other {
            message: "classifier meta labels empty".into(),
        });
    }
    let n_labels = resolved_labels.len();
    let init_seed = meta["seed"].as_u64();
    let mut model = Classifier::with_labels(resolved_labels, resolved_input_dim);
    model.init_seed = init_seed;
    model.backbone.weight = tensor_from_entry(backbone_entry)?;
    model.head.weight = tensor_from_entry(head_entry)?;
    expect_tensor_shape(
        &model.backbone.weight,
        &[CLASSIFIER_HIDDEN, resolved_input_dim],
        "backbone",
    )?;
    expect_tensor_shape(
        &model.head.weight,
        &[n_labels, CLASSIFIER_HIDDEN],
        "head",
    )?;
    Ok(model)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::hf_tensor_codec::is_hf_binary_bytes;
    use mmn_core::MmnError;

    #[test]
    fn hf_classifier_name_aliases() {
        assert_eq!(hf_classifier_name_to_mmn("classifier.weight"), Some("head"));
        assert_eq!(hf_classifier_name_to_mmn("encoder.weight"), Some("backbone"));
    }

    #[test]
    fn hf_classifier_safetensors_roundtrip() {
        let clf = Classifier::with_labels(vec!["pos".into(), "neg".into()], 32);
        let path = std::env::temp_dir().join("mmn_hf_clf_test.safetensors");
        export_hf_classifier_safetensors(&clf, path.to_str().unwrap()).unwrap();
        let bytes = fs::read(&path).unwrap();
        assert!(is_hf_binary_bytes(&bytes));
        let loaded = import_hf_classifier_safetensors(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.labels, clf.labels);
        assert_eq!(loaded.input_dim, clf.input_dim);
        assert_eq!(
            clf.backbone.weight.data[[0, 0]],
            loaded.backbone.weight.data[[0, 0]]
        );
    }

    #[test]
    fn rejects_chatbot_hf_format() {
        let bot_path = std::env::temp_dir().join("mmn_hf_clf_chatbot_reject.safetensors");
        let bot = mmn_models::Chatbot::new(false, None, 32, Some(1), Some(8));
        crate::hf_safetensors::export_hf_safetensors(&bot, bot_path.to_str().unwrap(), None)
            .unwrap();
        let result = import_hf_classifier_safetensors(bot_path.to_str().unwrap());
        match result {
            Err(MmnError::Other { message, .. }) => assert!(message.contains("chatbot")),
            Ok(_) => panic!("expected classifier import to reject chatbot checkpoint"),
            Err(other) => panic!("unexpected error: {other:?}"),
        }
    }
}
