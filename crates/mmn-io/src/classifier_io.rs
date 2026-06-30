use crate::checkpoint_util::{
    expect_tensor_shape, quantize_tensor, require_tensor_entry, tensor_from_entry, tensor_to_entry,
    write_file_create_parents,
};
use crate::tensor_merge::average_tensors;
use mmn_core::MmnError;
use mmn_models::Classifier;
use std::collections::HashMap;
use std::fs;

/// Hidden width for `Classifier` backbone/head (must match `mmn_models::Classifier`).
const CLASSIFIER_HIDDEN: usize = 128;

pub fn export_classifier(model: &Classifier, path: &str) -> Result<(), MmnError> {
    let mut map = HashMap::new();
    map.insert(
        "backbone".to_string(),
        tensor_to_entry(&model.backbone.weight),
    );
    map.insert("head".to_string(), tensor_to_entry(&model.head.weight));
    let mut meta = serde_json::json!({
        "input_dim": model.input_dim,
        "labels": model.labels,
    });
    if let Some(seed) = model.init_seed {
        meta["seed"] = serde_json::json!(seed);
    }
    let wrapper = serde_json::json!({
        "tensors": map,
        "format": "mmn-classifier-v1",
        "meta": meta,
    });
    write_file_create_parents(path, wrapper.to_string())?;
    Ok(())
}

pub fn import_classifier(path: &str) -> Result<Classifier, MmnError> {
    let text = fs::read_to_string(path).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    let v: serde_json::Value = serde_json::from_str(&text).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    if v["format"].as_str() != Some("mmn-classifier-v1") {
        return Err(MmnError::Other {
            message: "Expected mmn-classifier-v1 checkpoint".into(),
        });
    }
    let meta = &v["meta"];
    let labels: Vec<String> = meta["labels"]
        .as_array()
        .ok_or_else(|| MmnError::Other {
            message: "classifier meta missing labels".into(),
        })?
        .iter()
        .filter_map(|x| x.as_str().map(String::from))
        .collect();
    if labels.is_empty() {
        return Err(MmnError::Other {
            message: "classifier meta labels empty".into(),
        });
    }
    let input_dim = meta["input_dim"].as_u64().ok_or_else(|| MmnError::Other {
        message: "classifier meta missing input_dim".into(),
    })? as usize;
    let init_seed = meta["seed"].as_u64();
    let n_labels = labels.len();
    let mut model = Classifier::with_labels(labels, input_dim);
    model.init_seed = init_seed;
    let tensors = &v["tensors"];
    model.backbone.weight = tensor_from_entry(require_tensor_entry(tensors, "backbone")?)?;
    model.head.weight = tensor_from_entry(require_tensor_entry(tensors, "head")?)?;
    expect_tensor_shape(&model.backbone.weight, &[CLASSIFIER_HIDDEN, input_dim], "backbone")?;
    expect_tensor_shape(&model.head.weight, &[n_labels, CLASSIFIER_HIDDEN], "head")?;
    Ok(model)
}

pub fn merge_classifiers(a: &Classifier, b: &Classifier) -> Result<Classifier, MmnError> {
    if a.input_dim != b.input_dim || a.labels != b.labels {
        return Err(MmnError::ModelMismatch {
            message: "Cannot merge classifiers with different labels or input_dim".into(),
            fix: "Use two Classifier instances built from the same label set and input_dim.".into(),
            explanation: "merge_classifier averages backbone and head weights in place.".into(),
        });
    }
    let mut out = Classifier::with_labels(a.labels.clone(), a.input_dim);
    out.init_seed = a.init_seed.or(b.init_seed);
    out.backbone.weight = average_tensors(&a.backbone.weight, &b.backbone.weight);
    out.head.weight = average_tensors(&a.head.weight, &b.head.weight);
    Ok(out)
}

pub fn quantize_classifier(model: &mut Classifier, mode: &str) -> Result<(), MmnError> {
    match mode {
        "int8" | "int4" => {
            let scale = if mode == "int8" { 127.0 } else { 15.0 };
            quantize_tensor(&mut model.backbone.weight, scale);
            quantize_tensor(&mut model.head.weight, scale);
            Ok(())
        }
        _ => Err(MmnError::Other {
            message: format!("Unknown quant mode: {mode}"),
        }),
    }
}
