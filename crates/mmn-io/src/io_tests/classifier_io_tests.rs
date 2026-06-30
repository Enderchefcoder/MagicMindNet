use super::helpers::temp_file;
use crate::{
    export_classifier, export_safetensors, import_classifier, import_safetensors,
    merge_classifiers, quantize_classifier,
};
use mmn_core::MmnError;
use mmn_models::{Chatbot, Classifier};
use std::fs;

    #[test]
    fn export_classifier_creates_parent_directory() {
        let model = Classifier::with_labels_seed(vec!["A".into(), "B".into()], 16, Some(1));
        let base = temp_file("nested_clf_export");
        let _ = fs::remove_dir_all(&base);
        let path = base.join("nested").join("clf.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        assert!(path.is_file());
        let _ = fs::remove_dir_all(&base);
    }


    #[test]
    fn classifier_export_includes_seed_in_meta() {
        let model = Classifier::with_labels_seed(vec!["A".into(), "B".into()], 16, Some(7));
        let path = temp_file("clf_seed.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(v["meta"]["seed"].as_u64(), Some(7));
        let loaded = import_classifier(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.init_seed, Some(7));
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_safetensors_rejects_classifier_checkpoint() {
        let model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let path = temp_file("clf_only.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("mmn-safetensors-v1"),
            "expected format error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_classifier_rejects_chatbot_checkpoint() {
        let model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let path = temp_file("wrong_fmt.mmn");
        export_safetensors(&model, path.to_str().unwrap()).unwrap();
        let result = import_classifier(path.to_str().unwrap());
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("mmn-classifier-v1"),
            "expected format error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn classifier_roundtrip_preserves_predictions() {
        let model = Classifier::with_labels(vec!["Happy".into(), "Sad".into()], 32);
        let before = model.predict_text("hello world").unwrap();
        let path = temp_file("clf.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let loaded = import_classifier(path.to_str().unwrap()).unwrap();
        assert_eq!(loaded.labels, model.labels);
        assert_eq!(loaded.input_dim, model.input_dim);
        let after = loaded.predict_text("hello world").unwrap();
        for (label, p) in &before {
            let q = after.get(label).copied().unwrap_or(0.0);
            assert!(
                (p - q).abs() < 1e-5,
                "prob drift for {label}: {p} vs {q}"
            );
        }
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn merge_classifiers_averages_weights() {
        let a = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let b = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let w_a = a.backbone.weight.data[[0, 0]];
        let w_b = b.backbone.weight.data[[0, 0]];
        let merged = merge_classifiers(&a, &b).unwrap();
        let w_m = merged.backbone.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_classifiers_rejects_label_mismatch() {
        let a = Classifier::with_labels(vec!["A".into()], 16);
        let b = Classifier::with_labels(vec!["B".into()], 16);
        assert!(matches!(
            merge_classifiers(&a, &b),
            Err(MmnError::ModelMismatch { .. })
        ));
    }


    #[test]
    fn import_classifier_rejects_missing_labels_meta() {
        let model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let path = temp_file("clf_no_labels.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"].as_object_mut().unwrap().remove("labels");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_classifier(path.to_str().unwrap());
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("labels"),
            "expected labels meta error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_classifier_rejects_missing_input_dim() {
        let model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let path = temp_file("clf_no_input_dim.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"].as_object_mut().unwrap().remove("input_dim");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_classifier(path.to_str().unwrap());
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("input_dim"),
            "expected input_dim meta error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_classifier_rejects_empty_labels() {
        let model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let path = temp_file("clf_empty_labels.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"]["labels"] = serde_json::json!([]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_classifier(path.to_str().unwrap());
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("labels"),
            "expected empty labels error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_classifier_rejects_missing_backbone() {
        let model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let path = temp_file("clf_no_backbone.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"].as_object_mut().unwrap().remove("backbone");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_classifier(path.to_str().unwrap());
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("backbone"),
            "expected missing backbone error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn quantize_classifier_int8_changes_weights() {
        let mut model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let before = model.backbone.weight.data.as_ref().clone();
        quantize_classifier(&mut model, "int8").unwrap();
        assert_ne!(before, *model.backbone.weight.data.as_ref());
    }


    #[test]
    fn import_classifier_rejects_missing_head() {
        let model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let path = temp_file("clf_no_head.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"].as_object_mut().unwrap().remove("head");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_classifier(path.to_str().unwrap());
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("head"),
            "expected missing head error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_classifier_rejects_head_shape_mismatch() {
        let model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let path = temp_file("clf_bad_head.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"]["labels"] = serde_json::json!(["A", "B", "C"]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_classifier(path.to_str().unwrap());
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("head") && msg.contains("shape"),
            "expected head shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_classifier_rejects_invalid_json() {
        let path = temp_file("clf_bad_json.mmn");
        fs::write(&path, "{not json").unwrap();
        let result = import_classifier(path.to_str().unwrap());
        assert!(result.is_err(), "expected invalid JSON to fail");
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_classifier_rejects_empty_file() {
        let path = temp_file("clf_empty.mmn");
        fs::write(&path, "").unwrap();
        let result = import_classifier(path.to_str().unwrap());
        assert!(result.is_err(), "expected empty file to fail");
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_classifier_rejects_backbone_shape_mismatch() {
        let model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let path = temp_file("clf_bad_backbone.mmn");
        export_classifier(&model, path.to_str().unwrap()).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"]["input_dim"] = serde_json::json!(32);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_classifier(path.to_str().unwrap());
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("backbone") && msg.contains("shape"),
            "expected backbone shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn quantize_classifier_int4_changes_weights() {
        let mut model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let before_backbone = model.backbone.weight.data.as_ref().clone();
        let before_head = model.head.weight.data.as_ref().clone();
        quantize_classifier(&mut model, "int4").unwrap();
        assert_ne!(before_backbone, *model.backbone.weight.data.as_ref());
        assert_ne!(before_head, *model.head.weight.data.as_ref());
    }


    #[test]
    fn merge_classifiers_averages_head_weight() {
        let a = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let b = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let w_a = a.head.weight.data[[0, 0]];
        let w_b = b.head.weight.data[[0, 0]];
        let merged = merge_classifiers(&a, &b).unwrap();
        let w_m = merged.head.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn quantize_classifier_int8_changes_head() {
        let mut model = Classifier::with_labels(vec!["A".into(), "B".into()], 16);
        let before_head = model.head.weight.data.as_ref().clone();
        quantize_classifier(&mut model, "int8").unwrap();
        assert_ne!(before_head, *model.head.weight.data.as_ref());
    }
