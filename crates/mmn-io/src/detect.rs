//! Checkpoint file inspection: which model family does a file store?

use crate::hf_tensor_codec::{hf_err, is_hf_binary_bytes, HF_CHATBOT_FORMAT, HF_CLASSIFIER_FORMAT};
use crate::interop::gguf::is_gguf_bytes;
use crate::interop::zip::{is_zip_bytes, zip_entry_names};
use mmn_core::MmnError;
use safetensors::SafeTensors;
use std::fs;

/// Model family stored in a checkpoint file.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CheckpointKind {
    /// `mmn-safetensors-v1` JSON or HF binary safetensors chatbot weights.
    Chatbot,
    /// `mmn-classifier-v1` JSON or `mmn-hf-classifier-v1` binary classifier weights.
    Classifier,
    /// `mmn-diffusion-v1` JSON VAE + UNet weights.
    Diffusion,
    /// `mmn-bin-v1` architecture stub (no weights).
    ChatbotBin,
    /// GGUF container (llama.cpp-convention chatbot weights).
    ChatbotGguf,
    /// NumPy `.npz` archive of chatbot weights.
    ChatbotNpz,
    /// PyTorch `.pt` / `.pth` state-dict archive of chatbot weights.
    ChatbotTorch,
}

impl CheckpointKind {
    /// Human-readable family name for error messages.
    pub fn family(&self) -> &'static str {
        match self {
            CheckpointKind::Chatbot
            | CheckpointKind::ChatbotBin
            | CheckpointKind::ChatbotGguf
            | CheckpointKind::ChatbotNpz
            | CheckpointKind::ChatbotTorch => "Chatbot",
            CheckpointKind::Classifier => "Classifier",
            CheckpointKind::Diffusion => "Diffusion",
        }
    }
}

fn detect_zip_kind(bytes: &[u8]) -> Result<CheckpointKind, MmnError> {
    let names = zip_entry_names(bytes)?;
    if names
        .iter()
        .any(|n| n == "data.pkl" || n.ends_with("/data.pkl"))
    {
        return Ok(CheckpointKind::ChatbotTorch);
    }
    if names.iter().any(|n| n.ends_with(".npy") || n == "meta.json") {
        return Ok(CheckpointKind::ChatbotNpz);
    }
    Err(MmnError::Other {
        message: "zip archive is neither a torch checkpoint (data.pkl) nor an npz archive (.npy entries)"
            .into(),
    })
}

fn detect_binary_kind(bytes: &[u8]) -> Result<CheckpointKind, MmnError> {
    let (_header_len, header_meta) = SafeTensors::read_metadata(bytes).map_err(hf_err)?;
    let format = header_meta
        .metadata()
        .as_ref()
        .and_then(|m| m.get("format"))
        .cloned();
    match format.as_deref() {
        Some(HF_CLASSIFIER_FORMAT) => Ok(CheckpointKind::Classifier),
        // External HF checkpoints (e.g. Llama exports) carry no `format` key;
        // the chatbot importer adapts those, so treat them as chatbots.
        Some(HF_CHATBOT_FORMAT) | None => Ok(CheckpointKind::Chatbot),
        Some(other) => Err(MmnError::Other {
            message: format!("Unknown binary safetensors format {other:?}"),
        }),
    }
}

fn detect_json_kind(bytes: &[u8]) -> Result<CheckpointKind, MmnError> {
    let text = std::str::from_utf8(bytes).map_err(|e| MmnError::Other {
        message: format!("checkpoint is neither binary safetensors nor UTF-8 JSON: {e}"),
    })?;
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| MmnError::Other {
        message: format!("checkpoint JSON parse failed: {e}"),
    })?;
    match v["format"].as_str() {
        Some("mmn-safetensors-v1") => Ok(CheckpointKind::Chatbot),
        Some("mmn-classifier-v1") => Ok(CheckpointKind::Classifier),
        Some("mmn-diffusion-v1") => Ok(CheckpointKind::Diffusion),
        Some("mmn-bin-v1") => Ok(CheckpointKind::ChatbotBin),
        Some(other) => Err(MmnError::Other {
            message: format!("Unknown checkpoint format {other:?}"),
        }),
        None => Err(MmnError::Other {
            message: "checkpoint JSON has no \"format\" field".into(),
        }),
    }
}

/// Inspect a checkpoint file and report which model family it stores.
///
/// Handles every format MagicMindNet can write: JSON `mmn-safetensors-v1` /
/// `mmn-classifier-v1` / `mmn-diffusion-v1` / `mmn-bin-v1` wrappers and binary
/// HF safetensors (chatbot and classifier).
pub fn detect_checkpoint_kind(path: &str) -> Result<CheckpointKind, MmnError> {
    let bytes = fs::read(path).map_err(|e| MmnError::Other {
        message: format!("cannot read checkpoint {path}: {e}"),
    })?;
    if bytes.is_empty() {
        return Err(MmnError::Other {
            message: format!("checkpoint {path} is empty"),
        });
    }
    if is_gguf_bytes(&bytes) {
        return Ok(CheckpointKind::ChatbotGguf);
    }
    if is_zip_bytes(&bytes) {
        return detect_zip_kind(&bytes);
    }
    if is_hf_binary_bytes(&bytes) {
        detect_binary_kind(&bytes)
    } else {
        detect_json_kind(&bytes)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{
        export_bin, export_classifier, export_diffusion, export_hf_classifier_safetensors,
        export_hf_safetensors, export_safetensors,
    };
    use mmn_models::{Chatbot, Classifier, Diffusion};
    use std::path::PathBuf;

    fn tmp_path(name: &str) -> PathBuf {
        let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("target")
            .join("detect_tests");
        let _ = std::fs::create_dir_all(&dir);
        dir.join(name)
    }

    #[test]
    fn detects_chatbot_json_and_binary() {
        let model = Chatbot::new_with_seed(false, None, 64, Some(1), Some(16), Some(1));
        let json = tmp_path("bot.mmn");
        export_safetensors(&model, json.to_str().unwrap(), None).unwrap();
        assert_eq!(
            detect_checkpoint_kind(json.to_str().unwrap()).unwrap(),
            CheckpointKind::Chatbot
        );
        let bin = tmp_path("bot.safetensors");
        export_hf_safetensors(&model, bin.to_str().unwrap(), None).unwrap();
        assert_eq!(
            detect_checkpoint_kind(bin.to_str().unwrap()).unwrap(),
            CheckpointKind::Chatbot
        );
    }

    #[test]
    fn detects_classifier_json_and_binary() {
        let model = Classifier::with_labels_seed(vec!["A".into(), "B".into()], 16, Some(2));
        let json = tmp_path("clf.mmn");
        export_classifier(&model, json.to_str().unwrap()).unwrap();
        assert_eq!(
            detect_checkpoint_kind(json.to_str().unwrap()).unwrap(),
            CheckpointKind::Classifier
        );
        let bin = tmp_path("clf.safetensors");
        export_hf_classifier_safetensors(&model, bin.to_str().unwrap()).unwrap();
        assert_eq!(
            detect_checkpoint_kind(bin.to_str().unwrap()).unwrap(),
            CheckpointKind::Classifier
        );
    }

    #[test]
    fn detects_diffusion_and_bin_stub() {
        let diff = Diffusion::new();
        let dpath = tmp_path("diff.mmn");
        export_diffusion(&diff, dpath.to_str().unwrap()).unwrap();
        assert_eq!(
            detect_checkpoint_kind(dpath.to_str().unwrap()).unwrap(),
            CheckpointKind::Diffusion
        );
        let bot = Chatbot::new_with_seed(false, None, 64, Some(1), Some(16), Some(3));
        let bpath = tmp_path("bot.bin");
        export_bin(&bot, bpath.to_str().unwrap()).unwrap();
        assert_eq!(
            detect_checkpoint_kind(bpath.to_str().unwrap()).unwrap(),
            CheckpointKind::ChatbotBin
        );
    }

    #[test]
    fn detects_gguf_npz_and_torch() {
        let model = Chatbot::new_with_seed(false, None, 32, Some(1), Some(8), Some(4));
        let gguf = tmp_path("bot.gguf");
        crate::export_gguf(&model, gguf.to_str().unwrap(), "f32").unwrap();
        assert_eq!(
            detect_checkpoint_kind(gguf.to_str().unwrap()).unwrap(),
            CheckpointKind::ChatbotGguf
        );
        let npz = tmp_path("bot.npz");
        crate::export_npz(&model, npz.to_str().unwrap()).unwrap();
        assert_eq!(
            detect_checkpoint_kind(npz.to_str().unwrap()).unwrap(),
            CheckpointKind::ChatbotNpz
        );
        let pt = tmp_path("bot.pt");
        crate::export_torch_pt(&model, pt.to_str().unwrap()).unwrap();
        assert_eq!(
            detect_checkpoint_kind(pt.to_str().unwrap()).unwrap(),
            CheckpointKind::ChatbotTorch
        );
        for kind in [
            CheckpointKind::ChatbotGguf,
            CheckpointKind::ChatbotNpz,
            CheckpointKind::ChatbotTorch,
        ] {
            assert_eq!(kind.family(), "Chatbot");
        }
    }

    #[test]
    fn unrecognized_zip_errors() {
        let path = tmp_path("weird.zip");
        let bytes =
            crate::write_zip_stored(&[("readme.txt".to_string(), b"hi".to_vec())]).unwrap();
        std::fs::write(&path, bytes).unwrap();
        let err = detect_checkpoint_kind(path.to_str().unwrap()).unwrap_err();
        assert!(err.message().contains("neither a torch checkpoint"));
    }

    #[test]
    fn unknown_json_format_errors() {
        let path = tmp_path("weird.json");
        std::fs::write(&path, r#"{"format":"not-a-checkpoint"}"#).unwrap();
        let err = detect_checkpoint_kind(path.to_str().unwrap()).unwrap_err();
        assert!(err.message().contains("not-a-checkpoint"));
    }

    #[test]
    fn missing_file_errors_with_path() {
        let err = detect_checkpoint_kind("/nonexistent/model.mmn").unwrap_err();
        assert!(err.message().contains("/nonexistent/model.mmn"));
    }
}
