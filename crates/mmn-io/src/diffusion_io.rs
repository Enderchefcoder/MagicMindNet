use crate::checkpoint_util::{
    expect_tensor_shape, require_tensor_entry, tensor_from_entry, tensor_to_entry,
    write_file_create_parents,
};
use mmn_core::MmnError;
use mmn_models::Diffusion;
use std::collections::HashMap;
use std::fs;

const DIFFUSION_FORMAT: &str = "mmn-diffusion-v1";
const LATENT_SPATIAL: usize = 8;

pub fn export_diffusion(model: &Diffusion, path: &str) -> Result<(), MmnError> {
    let mut map = HashMap::new();
    map.insert(
        "vae_enc_conv1".to_string(),
        tensor_to_entry(&model.vae.conv1.weight),
    );
    map.insert(
        "vae_enc_conv2".to_string(),
        tensor_to_entry(&model.vae.conv2.weight),
    );
    map.insert(
        "vae_dec_conv1".to_string(),
        tensor_to_entry(&model.vae_decoder.conv1.weight),
    );
    map.insert(
        "vae_dec_conv2".to_string(),
        tensor_to_entry(&model.vae_decoder.conv2.weight),
    );
    map.insert(
        "unet_down".to_string(),
        tensor_to_entry(&model.unet.down.weight),
    );
    map.insert(
        "unet_mid".to_string(),
        tensor_to_entry(&model.unet.mid.weight),
    );
    map.insert(
        "unet_up".to_string(),
        tensor_to_entry(&model.unet.up.weight),
    );
    let wrapper = serde_json::json!({
        "tensors": map,
        "format": DIFFUSION_FORMAT,
        "meta": {
            "latent_channels": model.latent_channels,
            "spatial": LATENT_SPATIAL,
        },
    });
    write_file_create_parents(path, wrapper.to_string())?;
    Ok(())
}

pub fn import_diffusion(path: &str) -> Result<Diffusion, MmnError> {
    let text = fs::read_to_string(path).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    import_diffusion_json(&text)
}

fn import_diffusion_json(text: &str) -> Result<Diffusion, MmnError> {
    let v: serde_json::Value = serde_json::from_str(text).map_err(|e| MmnError::Other {
        message: e.to_string(),
    })?;
    if v["format"].as_str() != Some(DIFFUSION_FORMAT) {
        return Err(MmnError::Other {
            message: format!("Expected {DIFFUSION_FORMAT} checkpoint"),
        });
    }
    let latent_channels = v["meta"]["latent_channels"]
        .as_u64()
        .unwrap_or(4) as usize;
    let mut model = Diffusion::new();
    model.latent_channels = latent_channels;
    let tensors = &v["tensors"];
    model.vae.conv1.weight =
        tensor_from_entry(require_tensor_entry(tensors, "vae_enc_conv1")?)?;
    model.vae.conv2.weight =
        tensor_from_entry(require_tensor_entry(tensors, "vae_enc_conv2")?)?;
    model.vae_decoder.conv1.weight =
        tensor_from_entry(require_tensor_entry(tensors, "vae_dec_conv1")?)?;
    model.vae_decoder.conv2.weight =
        tensor_from_entry(require_tensor_entry(tensors, "vae_dec_conv2")?)?;
    model.unet.down.weight = tensor_from_entry(require_tensor_entry(tensors, "unet_down")?)?;
    model.unet.mid.weight = tensor_from_entry(require_tensor_entry(tensors, "unet_mid")?)?;
    model.unet.up.weight = tensor_from_entry(require_tensor_entry(tensors, "unet_up")?)?;
    validate_diffusion_shapes(&model)?;
    Ok(model)
}

fn validate_diffusion_shapes(model: &Diffusion) -> Result<(), MmnError> {
    let lc = model.latent_channels;
    expect_tensor_shape(&model.vae.conv1.weight, &[64, 3, 3, 3], "vae_enc_conv1")?;
    expect_tensor_shape(&model.vae.conv2.weight, &[4, 64, 3, 3], "vae_enc_conv2")?;
    expect_tensor_shape(&model.vae_decoder.conv1.weight, &[64, lc, 3, 3], "vae_dec_conv1")?;
    expect_tensor_shape(&model.vae_decoder.conv2.weight, &[3, 64, 3, 3], "vae_dec_conv2")?;
    expect_tensor_shape(&model.unet.down.weight, &[64, lc, 3, 3], "unet_down")?;
    expect_tensor_shape(&model.unet.mid.weight, &[64, 64, 3, 3], "unet_mid")?;
    expect_tensor_shape(&model.unet.up.weight, &[lc, 64, 3, 3], "unet_up")?;
    Ok(())
}
