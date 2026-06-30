use super::helpers::temp_file;
use crate::{
    export_bin, export_safetensors, import_bin, import_safetensors, merge_models, quantize_model,
};
use mmn_core::{MmnError, Tensor};
use mmn_models::Chatbot;
use std::fs;

    #[test]
    fn export_bin_creates_parent_directory() {
        let model = Chatbot::new(false, None, 128, Some(1), Some(16));
        let base = temp_file("nested_bin_export");
        let _ = fs::remove_dir_all(&base);
        let path = base.join("nested").join("arch.bin");
        export_bin(&model, path.to_str().unwrap()).unwrap();
        assert!(path.is_file());
        let _ = fs::remove_dir_all(&base);
    }


    #[test]
    fn export_safetensors_creates_parent_directory() {
        let model = Chatbot::new(false, None, 128, Some(1), Some(16));
        let base = temp_file("nested_export");
        let _ = fs::remove_dir_all(&base);
        let path = base.join("nested").join("bot.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        assert!(path.is_file());
        let _ = fs::remove_dir_all(&base);
    }


    #[test]
    fn export_includes_bpe_checkpoint_meta() {
        let model = Chatbot::new(false, None, 128, Some(1), Some(16));
        let path = temp_file("bpe_meta.mmn");
        export_safetensors(&model, path.to_str().unwrap(), Some("bot.bpe.mmn")).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["meta"]["bpe_checkpoint"].as_str(), Some("bot.bpe.mmn"));
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn export_includes_seed_in_meta() {
        let model = Chatbot::new_with_seed(false, None, 256, Some(2), Some(32), Some(42));
        let path = temp_file("seed_meta.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["meta"]["seed"].as_u64(), Some(42));
        let loaded = import_safetensors(path.to_str().unwrap(), 256).unwrap();
        assert_eq!(loaded.init_seed, Some(42));
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn safetensors_roundtrip_preserves_weights() {
        let model = Chatbot::new(false, None, 512, Some(2), Some(64));
        let embed_before: Vec<f32> = model.embed.weight.data.iter().copied().collect();
        let head_before: Vec<f32> = model.lm_head.weight.data.iter().copied().collect();
        let path = temp_file("roundtrip.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 512).unwrap();
        let embed_after: Vec<f32> = loaded.embed.weight.data.iter().copied().collect();
        let head_after: Vec<f32> = loaded.lm_head.weight.data.iter().copied().collect();
        assert_eq!(embed_before, embed_after);
        assert_eq!(head_before, head_after);
        assert_eq!(loaded.shape.n_layer, model.shape.n_layer);
        assert_eq!(loaded.shape.d_model, model.shape.d_model);
        assert_eq!(loaded.shape.vocab_size, model.shape.vocab_size);
        let ffn_before: Vec<f32> = model.blocks[0].ffn.weight.data.iter().copied().collect();
        let ffn_after: Vec<f32> = loaded.blocks[0].ffn.weight.data.iter().copied().collect();
        assert_eq!(ffn_before, ffn_after);
        let ln_gamma: Vec<f32> = model.blocks[0].ln1.gamma.data.iter().copied().collect();
        let ln_gamma_loaded: Vec<f32> = loaded.blocks[0].ln1.gamma.data.iter().copied().collect();
        assert_eq!(ln_gamma, ln_gamma_loaded);
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn learned_pos_embed_roundtrip_preserves_weights() {
        let model = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), Some(3), true, 32,
        );
        let pe_before: Vec<f32> = model
            .pos_embed
            .as_ref()
            .unwrap()
            .weight
            .data
            .iter()
            .copied()
            .collect();
        let path = temp_file("learned_pos.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["meta"]["use_learned_pos_embed"], true);
        assert_eq!(v["meta"]["max_seq_len"].as_u64(), Some(32));
        let loaded = import_safetensors(path.to_str().unwrap(), 128).unwrap();
        assert!(loaded.use_learned_pos_embed);
        assert_eq!(loaded.max_seq_len, 32);
        let pe_after: Vec<f32> = loaded
            .pos_embed
            .as_ref()
            .unwrap()
            .weight
            .data
            .iter()
            .copied()
            .collect();
        assert_eq!(pe_before, pe_after);
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_pos_embed_when_meta_requires() {
        let model = Chatbot::new_with_pe_options(
            false, None, 64, Some(1), Some(16), None, true, 16,
        );
        let path = temp_file("missing_pos_embed.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"].as_object_mut().unwrap().remove("pos_embed");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 64);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(msg.contains("pos_embed"), "expected pos_embed error, got: {msg}");
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_preserves_forward_loss() {
        let model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let tokens: Vec<usize> = (0..8).collect();
        let targets: Vec<usize> = (1..9).collect();
        let loss_before = model.loss_on_batch(&tokens, &targets).unwrap();
        let path = temp_file("loss.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 256).unwrap();
        let loss_after = loaded.loss_on_batch(&tokens, &targets).unwrap();
        assert!(
            (loss_before - loss_after).abs() < 1e-4,
            "loss drift after import: {loss_before} vs {loss_after}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_preserves_forward_loss_learned_pos_embed() {
        let model = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), Some(11), true, 32,
        );
        let tokens: Vec<usize> = (0..6).collect();
        let targets: Vec<usize> = (1..7).collect();
        let loss_before = model.loss_on_batch(&tokens, &targets).unwrap();
        let path = temp_file("learned_pos_loss.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 128).unwrap();
        assert!(loaded.use_learned_pos_embed);
        assert_eq!(loaded.max_seq_len, 32);
        let loss_after = loaded.loss_on_batch(&tokens, &targets).unwrap();
        assert!(
            (loss_before - loss_after).abs() < 1e-4,
            "learned PE loss drift after import: {loss_before} vs {loss_after}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn merge_models_preserves_init_seed_from_first() {
        let a = Chatbot::new_with_seed(false, None, 256, Some(2), Some(32), Some(5));
        let b = Chatbot::new_with_seed(false, None, 256, Some(2), Some(32), Some(9));
        let merged = merge_models(&a, &b).unwrap();
        assert_eq!(merged.init_seed, Some(5));
    }


    #[test]
    fn merge_rejects_vocab_mismatch() {
        let a = Chatbot::new(false, None, 64, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        assert!(matches!(
            merge_models(&a, &b),
            Err(MmnError::ModelMismatch { .. })
        ));
    }


    #[test]
    fn merge_rejects_mismatched_shapes() {
        let a = Chatbot::new(false, None, 512, Some(2), Some(64));
        let b = Chatbot::new(false, None, 512, Some(4), Some(64));
        assert!(matches!(
            merge_models(&a, &b),
            Err(MmnError::ModelMismatch { .. })
        ));
    }


    #[test]
    fn import_bin_rejects_safetensors_checkpoint() {
        let model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let path = temp_file("safetensors_for_bin.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let result = import_bin(path.to_str().unwrap());
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("mmn-bin-v1"),
            "expected format error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn quantize_int8_changes_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let before_embed = model.embed.weight.data.as_ref().clone();
        let before_head = model.lm_head.weight.data.as_ref().clone();
        quantize_model(&mut model, "int8").unwrap();
        assert_ne!(before_embed, *model.embed.weight.data.as_ref());
        assert_ne!(before_head, *model.lm_head.weight.data.as_ref());
    }


    #[test]
    fn quantize_int4_changes_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(2), Some(32));
        let before_embed = model.embed.weight.data.as_ref().clone();
        quantize_model(&mut model, "int4").unwrap();
        assert_ne!(before_embed, *model.embed.weight.data.as_ref());
    }


    #[test]
    fn quantize_int8_changes_pos_embed_weights() {
        let mut model = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), Some(4), true, 32,
        );
        let before = model.pos_embed.as_ref().unwrap().weight.data.as_ref().clone();
        quantize_model(&mut model, "int8").unwrap();
        assert_ne!(before, *model.pos_embed.as_ref().unwrap().weight.data.as_ref());
    }


    #[test]
    fn quantize_int4_changes_pos_embed_weights() {
        let mut model = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), Some(5), true, 32,
        );
        let before = model.pos_embed.as_ref().unwrap().weight.data.as_ref().clone();
        quantize_model(&mut model, "int4").unwrap();
        assert_ne!(before, *model.pos_embed.as_ref().unwrap().weight.data.as_ref());
    }


    #[test]
    fn merge_rejects_d_model_mismatch() {
        let a = Chatbot::new(false, None, 512, Some(2), Some(64));
        let b = Chatbot::new(false, None, 512, Some(2), Some(128));
        assert!(matches!(
            merge_models(&a, &b),
            Err(MmnError::ModelMismatch { .. })
        ));
    }


    #[test]
    fn merge_or_vision_flag() {
        let a = Chatbot::new(false, None, 256, Some(2), Some(32));
        let b = Chatbot::new(true, None, 256, Some(2), Some(32));
        let merged = merge_models(&a, &b).unwrap();
        assert!(merged.vision);
    }


    #[test]
    fn import_rejects_missing_embed_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_embed.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"].as_object_mut().unwrap().remove("embed");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("embed"),
            "expected missing tensor error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_incomplete_meta() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("incomplete_meta.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"].as_object_mut().unwrap().remove("n_layer");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("n_layer"),
            "expected meta error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_tensor_data_length_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_tensor.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["embed"]["data"] = serde_json::json!([0, 0, 0, 0]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("length mismatch"),
            "expected tensor mismatch error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_pos_embed_shape_mismatch() {
        let model = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), None, true, 32,
        );
        let path = temp_file("bad_pos_embed_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"]["max_seq_len"] = serde_json::json!(64);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 128);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("pos_embed") && msg.contains("shape"),
            "expected pos_embed shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn learned_pos_embed_increases_parameter_count() {
        let sinusoidal = Chatbot::new(false, None, 128, Some(1), Some(16));
        let learned = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), None, true, 32,
        );
        assert_eq!(
            learned.parameters(),
            sinusoidal.parameters() + 32 * 16
        );
    }


    #[test]
    fn import_rejects_embed_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_embed_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"]["d_model"] = serde_json::json!(32);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("embed") && msg.contains("shape"),
            "expected embed shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_non_numeric_tensor_bytes() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_bytes.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["embed"]["data"] = serde_json::json!([999, 999, 999, 999]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("byte") || msg.contains("invalid"),
            "expected byte parse error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_bin_rejects_invalid_json() {
        let path = temp_file("bin_bad_json.json");
        fs::write(&path, "{not json").unwrap();
        let result = import_bin(path.to_str().unwrap());
        assert!(result.is_err(), "expected invalid JSON to fail");
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_bin_rejects_empty_file() {
        let path = temp_file("bin_empty.json");
        fs::write(&path, "").unwrap();
        let result = import_bin(path.to_str().unwrap());
        assert!(result.is_err(), "expected empty file to fail");
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_vocab_size_meta() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("no_vocab_meta.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"].as_object_mut().unwrap().remove("vocab_size");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 0);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("vocab_size"),
            "expected vocab_size meta error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_invalid_json() {
        let path = temp_file("invalid_json.mmn");
        fs::write(&path, "{not json").unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 0);
        assert!(result.is_err());
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_empty_file() {
        let path = temp_file("empty.mmn");
        fs::write(&path, "").unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 0);
        assert!(result.is_err());
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_bin_empty_object_uses_documented_defaults() {
        let path = temp_file("empty_bin.json");
        fs::write(&path, "{}").unwrap();
        let model = import_bin(path.to_str().unwrap()).unwrap();
        assert_eq!(model.shape.vocab_size, 32000);
        assert_eq!(model.shape.d_model, 128);
        assert_eq!(model.shape.n_layer, 4);
        assert!(!model.vision);
        assert!(!model.use_learned_pos_embed);
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn bin_learned_pos_embed_roundtrip_preserves_meta() {
        let model = Chatbot::new_with_pe_options(
            false, None, 128, Some(2), Some(32), Some(5), true, 64,
        );
        let path = temp_file("learned_pos_arch.bin");
        export_bin(&model, path.to_str().unwrap()).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["use_learned_pos_embed"], true);
        assert_eq!(v["max_seq_len"].as_u64(), Some(64));
        let loaded = import_bin(path.to_str().unwrap()).unwrap();
        assert!(loaded.use_learned_pos_embed);
        assert_eq!(loaded.max_seq_len, 64);
        assert_eq!(loaded.shape.vocab_size, 128);
        assert_eq!(loaded.shape.n_layer, 2);
        assert_eq!(loaded.shape.d_model, 32);
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn bin_vision_and_learned_pos_embed_roundtrip_preserves_meta() {
        let model = Chatbot::new_with_pe_options(
            true, None, 128, Some(1), Some(16), Some(6), true, 32,
        );
        let path = temp_file("vision_learned_arch.bin");
        export_bin(&model, path.to_str().unwrap()).unwrap();
        let loaded = import_bin(path.to_str().unwrap()).unwrap();
        assert!(loaded.vision);
        assert!(loaded.use_learned_pos_embed);
        assert_eq!(loaded.max_seq_len, 32);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn safetensors_vision_patch_proj_roundtrip() {
        let model = Chatbot::new_with_seed(true, None, 128, Some(1), Some(16), Some(9));
        let w_before = model.vision_patch_proj.as_ref().unwrap().weight.data[[0, 0]];
        let path = temp_file("vision_patch.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 128).unwrap();
        assert!(loaded.has_vision_patch_encoder());
        let w_after = loaded.vision_patch_proj.as_ref().unwrap().weight.data[[0, 0]];
        assert_eq!(w_before, w_after);
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn safetensors_rope_meta_roundtrip() {
        let model = Chatbot::new_with_position_options(
            false, None, 128, Some(1), Some(16), Some(3), false, 512, true, 5000.0,
        );
        assert!(model.uses_rope());
        let path = temp_file("rope_meta.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 128).unwrap();
        assert!(loaded.uses_rope());
        assert!((loaded.rope_theta - 5000.0).abs() < 1e-3);
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_preserves_forward_loss_rope() {
        let model = Chatbot::new_with_position_options(
            false, None, 128, Some(1), Some(16), Some(11), false, 512, true, 8000.0,
        );
        let tokens: Vec<usize> = (0..6).collect();
        let targets: Vec<usize> = (1..7).collect();
        let loss_before = model.loss_on_batch(&tokens, &targets).unwrap();
        let path = temp_file("rope_loss.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 128).unwrap();
        assert!(loaded.uses_rope());
        assert!((loaded.rope_theta - 8000.0).abs() < 1e-3);
        let loss_after = loaded.loss_on_batch(&tokens, &targets).unwrap();
        assert!(
            (loss_before - loss_after).abs() < 1e-4,
            "RoPE loss drift after import: {loss_before} vs {loss_after}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn bin_rope_roundtrip_preserves_meta() {
        let model = Chatbot::new_with_position_options(
            false, None, 128, Some(2), Some(32), Some(5), false, 512, true, 7500.0,
        );
        let path = temp_file("rope_arch.bin");
        export_bin(&model, path.to_str().unwrap()).unwrap();
        let text = fs::read_to_string(&path).unwrap();
        let v: serde_json::Value = serde_json::from_str(&text).unwrap();
        assert_eq!(v["use_rope"], true);
        assert_eq!(v["rope_theta"].as_f64().unwrap(), 7500.0);
        let loaded = import_bin(path.to_str().unwrap()).unwrap();
        assert!(loaded.uses_rope());
        assert!((loaded.rope_theta - 7500.0).abs() < 1e-3);
        assert_eq!(loaded.shape.vocab_size, 128);
        assert_eq!(loaded.shape.n_layer, 2);
        assert_eq!(loaded.shape.d_model, 32);
        let _ = fs::remove_file(&path);
    }

    #[test]
    fn import_rejects_block_tensor_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_block_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.attn.q"]["shape"] = serde_json::json!([8, 32]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.attn.q") && msg.contains("shape"),
            "expected block shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_n_layer_meta_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_n_layer.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"]["n_layer"] = serde_json::json!(2);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.1"),
            "expected missing second block error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_d_model_meta() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("no_d_model_meta.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["meta"].as_object_mut().unwrap().remove("d_model");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("d_model"),
            "expected d_model meta error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_ffn_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_ffn_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.ffn"]["shape"] = serde_json::json!([128, 8]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ffn") && msg.contains("shape"),
            "expected ffn shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_ln_gamma_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_ln_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.ln1.gamma"]["shape"] = serde_json::json!([4, 4]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ln1.gamma") && msg.contains("shape"),
            "expected ln gamma shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block_ffn_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_ffn.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"].as_object_mut().unwrap().remove("blocks.0.ffn");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ffn"),
            "expected missing ffn error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_lm_head_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_lm_head_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["lm_head"]["shape"] = serde_json::json!([128, 32]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("lm_head") && msg.contains("shape"),
            "expected lm_head shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn merge_models_averages_block_attn_q() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].attn.q_proj.weight.data[[0, 0]];
        let w_b = b.blocks[0].attn.q_proj.weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].attn.q_proj.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn import_rejects_ffn2_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_ffn2_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.ffn2"]["shape"] = serde_json::json!([32, 32]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ffn2") && msg.contains("shape"),
            "expected ffn2 shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn quantize_int4_changes_block_ffn_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].ffn.weight.data.as_ref().clone();
        quantize_model(&mut model, "int4").unwrap();
        assert_ne!(before, *model.blocks[0].ffn.weight.data.as_ref());
    }


    #[test]
    fn merge_models_averages_embed() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.embed.weight.data[[0, 0]];
        let w_b = b.embed.weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.embed.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_models_averages_lm_head() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.lm_head.weight.data[[0, 0]];
        let w_b = b.lm_head.weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.lm_head.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_models_averages_pos_embed() {
        let a = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), Some(1), true, 32,
        );
        let b = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), Some(2), true, 32,
        );
        let w_a = a.pos_embed.as_ref().unwrap().weight.data[[0, 0]];
        let w_b = b.pos_embed.as_ref().unwrap().weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.pos_embed.as_ref().unwrap().weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_rejects_pos_embed_settings_mismatch() {
        let learned = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), None, true, 32,
        );
        let sinusoidal = Chatbot::new(false, None, 128, Some(1), Some(16));
        let result = merge_models(&learned, &sinusoidal);
        let msg = result.as_ref().err().expect("merge should fail").message();
        assert!(
            msg.contains("position") || msg.contains("pos_embed"),
            "expected PE mismatch error, got: {msg}"
        );
        let a = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), None, true, 32,
        );
        let b = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), None, true, 64,
        );
        let result = merge_models(&a, &b);
        assert!(result.is_err());
    }


    #[test]
    fn import_rejects_attn_k_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_attn_k_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.attn.k"]["shape"] = serde_json::json!([8, 32]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.attn.k") && msg.contains("shape"),
            "expected attn.k shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_ln2_gamma_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("bad_ln2_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.ln2.gamma"]["shape"] = serde_json::json!([4, 4]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ln2.gamma") && msg.contains("shape"),
            "expected ln2 gamma shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_attn_v_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("attn_v_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.attn.v"]["shape"] = serde_json::json!([8, 32]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.attn.v") && msg.contains("shape"),
            "expected attn.v shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_attn_out_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("attn_out_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.attn.out"]["shape"] = serde_json::json!([8, 32]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.attn.out") && msg.contains("shape"),
            "expected attn.out shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_lm_head_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_lm_head.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"].as_object_mut().unwrap().remove("lm_head");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("lm_head"),
            "expected missing lm_head error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn quantize_int8_changes_block_ffn_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].ffn.weight.data.as_ref().clone();
        quantize_model(&mut model, "int8").unwrap();
        assert_ne!(before, *model.blocks[0].ffn.weight.data.as_ref());
    }


    #[test]
    fn import_rejects_ln1_beta_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("ln1_beta_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.ln1.beta"]["shape"] = serde_json::json!([4, 4]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ln1.beta") && msg.contains("shape"),
            "expected ln1 beta shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_ln2_beta_shape_mismatch() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("ln2_beta_shape.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]["blocks.0.ln2.beta"]["shape"] = serde_json::json!([4, 4]);
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ln2.beta") && msg.contains("shape"),
            "expected ln2 beta shape error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn merge_models_averages_block_attn_k() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].attn.k_proj.weight.data[[0, 0]];
        let w_b = b.blocks[0].attn.k_proj.weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].attn.k_proj.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_models_averages_block_attn_v() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].attn.v_proj.weight.data[[0, 0]];
        let w_b = b.blocks[0].attn.v_proj.weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].attn.v_proj.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_models_averages_block_attn_out() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].attn.out_proj.weight.data[[0, 0]];
        let w_b = b.blocks[0].attn.out_proj.weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].attn.out_proj.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn import_rejects_missing_block_attn_q_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_attn_q.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.0.attn.q");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.attn.q"),
            "expected missing attn.q error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block_ffn2_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_ffn2.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"].as_object_mut().unwrap().remove("blocks.0.ffn2");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ffn2"),
            "expected missing ffn2 error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn merge_models_averages_block_ffn() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].ffn.weight.data[[0, 0]];
        let w_b = b.blocks[0].ffn.weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].ffn.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_models_averages_block_ffn2() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].ffn2.weight.data[[0, 0]];
        let w_b = b.blocks[0].ffn2.weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].ffn2.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn import_rejects_missing_block_attn_k_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_attn_k.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.0.attn.k");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.attn.k"),
            "expected missing attn.k error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block_ln1_gamma_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_ln1_gamma.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.0.ln1.gamma");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ln1.gamma"),
            "expected missing ln1.gamma error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn merge_models_averages_block_ln1_gamma() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].ln1.gamma.data[[0]];
        let w_b = b.blocks[0].ln1.gamma.data[[0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].ln1.gamma.data[[0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_models_averages_block_ln1_beta() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].ln1.beta.data[[0]];
        let w_b = b.blocks[0].ln1.beta.data[[0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].ln1.beta.data[[0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_models_averages_block_ln2_gamma() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].ln2.gamma.data[[0]];
        let w_b = b.blocks[0].ln2.gamma.data[[0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].ln2.gamma.data[[0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn merge_models_averages_block_ln2_beta() {
        let a = Chatbot::new(false, None, 128, Some(1), Some(16));
        let b = Chatbot::new(false, None, 128, Some(1), Some(16));
        let w_a = a.blocks[0].ln2.beta.data[[0]];
        let w_b = b.blocks[0].ln2.beta.data[[0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[0].ln2.beta.data[[0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-6);
    }


    #[test]
    fn quantize_int8_changes_block_attn_v_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].attn.v_proj.weight.data.as_ref().clone();
        quantize_model(&mut model, "int8").unwrap();
        assert_ne!(before, *model.blocks[0].attn.v_proj.weight.data.as_ref());
    }


    #[test]
    fn quantize_int8_changes_block_attn_out_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].attn.out_proj.weight.data.as_ref().clone();
        quantize_model(&mut model, "int8").unwrap();
        assert_ne!(before, *model.blocks[0].attn.out_proj.weight.data.as_ref());
    }


    #[test]
    fn quantize_int4_changes_block_attn_v_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].attn.v_proj.weight.data.as_ref().clone();
        quantize_model(&mut model, "int4").unwrap();
        assert_ne!(before, *model.blocks[0].attn.v_proj.weight.data.as_ref());
    }


    #[test]
    fn quantize_int8_changes_block_attn_k_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].attn.k_proj.weight.data.as_ref().clone();
        quantize_model(&mut model, "int8").unwrap();
        assert_ne!(before, *model.blocks[0].attn.k_proj.weight.data.as_ref());
    }


    #[test]
    fn quantize_int4_changes_block_attn_out_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].attn.out_proj.weight.data.as_ref().clone();
        quantize_model(&mut model, "int4").unwrap();
        assert_ne!(before, *model.blocks[0].attn.out_proj.weight.data.as_ref());
    }


    #[test]
    fn quantize_int4_changes_block_ffn2_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].ffn2.weight.data.as_ref().clone();
        quantize_model(&mut model, "int4").unwrap();
        assert_ne!(before, *model.blocks[0].ffn2.weight.data.as_ref());
    }


    #[test]
    fn quantize_int8_changes_block_attn_q_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].attn.q_proj.weight.data.as_ref().clone();
        quantize_model(&mut model, "int8").unwrap();
        assert_ne!(before, *model.blocks[0].attn.q_proj.weight.data.as_ref());
    }


    #[test]
    fn quantize_int4_changes_block_attn_k_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].attn.k_proj.weight.data.as_ref().clone();
        quantize_model(&mut model, "int4").unwrap();
        assert_ne!(before, *model.blocks[0].attn.k_proj.weight.data.as_ref());
    }


    #[test]
    fn quantize_int4_changes_block_attn_q_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].attn.q_proj.weight.data.as_ref().clone();
        quantize_model(&mut model, "int4").unwrap();
        assert_ne!(before, *model.blocks[0].attn.q_proj.weight.data.as_ref());
    }


    #[test]
    fn quantize_int8_changes_block_ffn2_weights() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let before = model.blocks[0].ffn2.weight.data.as_ref().clone();
        quantize_model(&mut model, "int8").unwrap();
        assert_ne!(before, *model.blocks[0].ffn2.weight.data.as_ref());
    }


    #[test]
    fn quantize_int8_changes_block_ln1_gamma_when_non_default() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let mut g = model.blocks[0].ln1.gamma.data.as_ref().clone();
        g[[0]] = 1.37;
        model.blocks[0].ln1.gamma = Tensor::from_array(g, true);
        let before = model.blocks[0].ln1.gamma.data[[0]];
        quantize_model(&mut model, "int8").unwrap();
        let after = model.blocks[0].ln1.gamma.data[[0]];
        let expected = (before * 127.0).round() / 127.0;
        assert_ne!(before, after);
        assert!((after - expected).abs() < 1e-6);
    }


    #[test]
    fn quantize_int4_changes_block_ln2_beta_when_non_default() {
        let mut model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let mut b = model.blocks[0].ln2.beta.data.as_ref().clone();
        b[[0]] = 0.25;
        model.blocks[0].ln2.beta = Tensor::from_array(b, true);
        let before = model.blocks[0].ln2.beta.data[[0]];
        quantize_model(&mut model, "int4").unwrap();
        let after = model.blocks[0].ln2.beta.data[[0]];
        let expected = (before * 15.0).round() / 15.0;
        assert_ne!(before, after);
        assert!((after - expected).abs() < 1e-6);
    }


    #[test]
    fn quantize_int8_learned_pos_embed_preserves_forward_loss_within_tolerance() {
        let mut model = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), Some(9), true, 32,
        );
        let tokens: Vec<usize> = (0..6).collect();
        let targets: Vec<usize> = (1..7).collect();
        let loss_before = model.loss_on_batch(&tokens, &targets).unwrap();
        quantize_model(&mut model, "int8").unwrap();
        let loss_after = model.loss_on_batch(&tokens, &targets).unwrap();
        assert!(loss_after.is_finite() && loss_after > 0.0);
        let rel = (loss_after - loss_before).abs() / loss_before.max(1e-6);
        assert!(
            rel < 0.5,
            "int8 quantize loss drift too large: {loss_before} -> {loss_after} (rel={rel})"
        );
    }


    #[test]
    fn quantize_int4_learned_pos_embed_preserves_forward_loss_within_tolerance() {
        let mut model = Chatbot::new_with_pe_options(
            false, None, 128, Some(1), Some(16), Some(10), true, 32,
        );
        let tokens: Vec<usize> = (0..6).collect();
        let targets: Vec<usize> = (1..7).collect();
        let loss_before = model.loss_on_batch(&tokens, &targets).unwrap();
        quantize_model(&mut model, "int4").unwrap();
        let loss_after = model.loss_on_batch(&tokens, &targets).unwrap();
        assert!(loss_after.is_finite() && loss_after > 0.0);
        let rel = (loss_after - loss_before).abs() / loss_before.max(1e-6);
        assert!(
            rel < 0.5,
            "int4 quantize loss drift too large: {loss_before} -> {loss_after} (rel={rel})"
        );
    }


    #[test]
    fn quantize_preserves_vision_flag_on_export_roundtrip() {
        let mut model = Chatbot::new(true, None, 128, Some(1), Some(16));
        quantize_model(&mut model, "int8").unwrap();
        assert!(model.vision);
        let path = temp_file("vision_quant.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let loaded = import_safetensors(path.to_str().unwrap(), 128).unwrap();
        assert!(loaded.vision);
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block_attn_v_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_attn_v.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.0.attn.v");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.attn.v"),
            "expected missing attn.v error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block_attn_out_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_attn_out.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.0.attn.out");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.attn.out"),
            "expected missing attn.out error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block_ln1_beta_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_ln1_beta.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.0.ln1.beta");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ln1.beta"),
            "expected missing ln1.beta error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block_ln2_gamma_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_ln2_gamma.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.0.ln2.gamma");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ln2.gamma"),
            "expected missing ln2.gamma error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block_ln2_beta_tensor() {
        let model = Chatbot::new(false, None, 256, Some(1), Some(16));
        let path = temp_file("missing_ln2_beta.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.0.ln2.beta");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.0.ln2.beta"),
            "expected missing ln2.beta error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block1_ffn_tensor() {
        let model = Chatbot::new(false, None, 256, Some(2), Some(16));
        let path = temp_file("missing_b1_ffn.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.1.ffn");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.1.ffn"),
            "expected missing blocks.1.ffn error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn import_rejects_missing_block1_attn_q_tensor() {
        let model = Chatbot::new(false, None, 256, Some(2), Some(16));
        let path = temp_file("missing_b1_attn_q.mmn");
        export_safetensors(&model, path.to_str().unwrap(), None).unwrap();
        let mut v: serde_json::Value =
            serde_json::from_str(&fs::read_to_string(&path).unwrap()).unwrap();
        v["tensors"]
            .as_object_mut()
            .unwrap()
            .remove("blocks.1.attn.q");
        fs::write(&path, v.to_string()).unwrap();
        let result = import_safetensors(path.to_str().unwrap(), 256);
        let msg = result.as_ref().err().expect("import should fail").message();
        assert!(
            msg.contains("blocks.1.attn.q"),
            "expected missing blocks.1.attn.q error, got: {msg}"
        );
        let _ = fs::remove_file(&path);
    }


    #[test]
    fn merge_models_averages_block1_ffn() {
        let a = Chatbot::new(false, None, 128, Some(2), Some(16));
        let b = Chatbot::new(false, None, 128, Some(2), Some(16));
        let w_a = a.blocks[1].ffn.weight.data[[0, 0]];
        let w_b = b.blocks[1].ffn.weight.data[[0, 0]];
        let merged = merge_models(&a, &b).unwrap();
        let w_m = merged.blocks[1].ffn.weight.data[[0, 0]];
        assert!((w_m - (w_a + w_b) / 2.0).abs() < 1e-5);
    }