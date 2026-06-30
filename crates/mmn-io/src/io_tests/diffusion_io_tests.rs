use crate::{export_diffusion, import_diffusion, merge_diffusion};
use mmn_models::Diffusion;

#[test]
fn diffusion_export_import_roundtrip_preserves_sample() {
    let dir = std::env::temp_dir();
    let path = dir.join("mmn_diffusion_roundtrip.mmn");
    let model = Diffusion::new();
    let before = model.sample_image(3, Some(42)).unwrap();
    export_diffusion(&model, path.to_str().unwrap()).unwrap();
    let loaded = import_diffusion(path.to_str().unwrap()).unwrap();
    let after = loaded.sample_image(3, Some(42)).unwrap();
    assert_eq!(
        before.data.as_slice().unwrap(),
        after.data.as_slice().unwrap()
    );
}

#[test]
fn import_diffusion_rejects_chatbot_checkpoint() {
    use crate::export_safetensors;
    use mmn_models::Chatbot;
    let dir = std::env::temp_dir();
    let path = dir.join("mmn_diffusion_bot.mmn");
    let bot = Chatbot::new(false, None, 64, Some(1), Some(16));
    export_safetensors(&bot, path.to_str().unwrap(), None).unwrap();
    match import_diffusion(path.to_str().unwrap()) {
        Err(e) => assert!(e.to_string().contains("mmn-diffusion-v1")),
        Ok(_) => panic!("expected import_diffusion to reject chatbot checkpoint"),
    }
}

#[test]
fn import_diffusion_rejects_missing_unet_tensor() {
    use crate::checkpoint_util::write_file_create_parents;
    let dir = std::env::temp_dir();
    let path = dir.join("mmn_diffusion_incomplete.mmn");
    let wrapper = serde_json::json!({
        "format": "mmn-diffusion-v1",
        "meta": { "latent_channels": 4 },
        "tensors": {}
    });
    write_file_create_parents(path.to_str().unwrap(), wrapper.to_string()).unwrap();
    assert!(import_diffusion(path.to_str().unwrap()).is_err());
}

#[test]
fn merge_diffusion_averages_unet_down_weight() {
    let a = Diffusion::new();
    let b = Diffusion::new();
    let mut a_data = (*a.unet.down.weight.data).clone();
    a_data[[0, 0, 0, 0]] = 0.2;
    let mut b_data = (*b.unet.down.weight.data).clone();
    b_data[[0, 0, 0, 0]] = 0.6;
    let mut a2 = Diffusion::new();
    let mut b2 = Diffusion::new();
    a2.unet.down.weight = mmn_core::Tensor::from_array(a_data, false);
    b2.unet.down.weight = mmn_core::Tensor::from_array(b_data, false);
    let merged = merge_diffusion(&a2, &b2).unwrap();
    let expected = (0.2 + 0.6) / 2.0;
    assert!((merged.unet.down.weight.data[[0, 0, 0, 0]] - expected).abs() < 1e-6);
}

#[test]
fn merge_diffusion_rejects_latent_channel_mismatch() {
    let a = Diffusion::new();
    let mut b = Diffusion::new();
    b.latent_channels = 8;
    assert!(merge_diffusion(&a, &b).is_err());
}
