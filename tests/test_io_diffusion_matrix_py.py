"""Parametric coverage of diffusion checkpoint IO (mmn-diffusion-v1)."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai
from conftest import load_checkpoint_tensors, tensor_entry_first_f32

DIFFUSION_TENSOR_KEYS = [
    "vae_enc_conv1",
    "vae_enc_conv2",
    "vae_dec_conv1",
    "vae_dec_conv2",
    "unet_down",
    "unet_mid",
    "unet_up",
]

# Wrong shapes with the same element count as exported tensors (latent_channels=4).
WRONG_SHAPES = {
    "vae_enc_conv1": [32, 6, 3, 3],
    "vae_enc_conv2": [8, 32, 3, 3],
    "vae_dec_conv1": [32, 8, 3, 3],
    "vae_dec_conv2": [6, 32, 3, 3],
    "unet_down": [32, 8, 3, 3],
    "unet_mid": [128, 32, 3, 3],
    "unet_up": [8, 32, 3, 3],
}


def _export_diffusion(tmp_path: Path) -> Path:
    d = ai.Diffusion()
    path = tmp_path / "diff.mmn"
    ai.export_diffusion(d, "safetensors", str(path))
    return path


@pytest.mark.parametrize("tensor_key", DIFFUSION_TENSOR_KEYS)
def test_import_diffusion_rejects_missing_tensor_matrix_py(tmp_path: Path, tensor_key: str):
    path = _export_diffusion(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"].pop(tensor_key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_diffusion("safetensors", [str(path)])
    assert tensor_key.lower() in str(exc.value).lower()


@pytest.mark.parametrize("tensor_key", DIFFUSION_TENSOR_KEYS)
def test_import_diffusion_rejects_shape_mismatch_matrix_py(tmp_path: Path, tensor_key: str):
    path = _export_diffusion(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    entry = payload["tensors"][tensor_key]
    entry["shape"] = WRONG_SHAPES[tensor_key]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_diffusion("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert tensor_key.lower() in msg and "shape" in msg


@pytest.mark.parametrize("tensor_key", DIFFUSION_TENSOR_KEYS)
def test_merge_diffusion_averages_tensor_matrix_py(tmp_path: Path, tensor_key: str):
    a = ai.Diffusion()
    b = ai.Diffusion()
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    path_m = tmp_path / "m.mmn"
    ai.export_diffusion(a, "safetensors", str(path_a))
    ai.export_diffusion(b, "safetensors", str(path_b))
    merged = ai.merge_diffusion(a, b)
    ai.export_diffusion(merged, "safetensors", str(path_m))
    wa = tensor_entry_first_f32(load_checkpoint_tensors(path_a)[tensor_key])
    wb = tensor_entry_first_f32(load_checkpoint_tensors(path_b)[tensor_key])
    wm = tensor_entry_first_f32(load_checkpoint_tensors(path_m)[tensor_key])
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


@pytest.mark.parametrize("mode", ["int8", "int4"])
@pytest.mark.parametrize("tensor_key", ["unet_down", "vae_enc_conv1"])
def test_quantize_diffusion_changes_tensor_matrix_py(tmp_path: Path, mode: str, tensor_key: str):
    d = ai.Diffusion()
    before_path = tmp_path / f"before_{mode}_{tensor_key}.mmn"
    ai.export_diffusion(d, "safetensors", str(before_path))
    before = load_checkpoint_tensors(before_path)
    ai.quantize_diffusion(d, mode)
    after_path = tmp_path / f"after_{mode}_{tensor_key}.mmn"
    ai.export_diffusion(d, "safetensors", str(after_path))
    after = load_checkpoint_tensors(after_path)
    assert before[tensor_key]["data"] != after[tensor_key]["data"]


def test_quantize_diffusion_export_import_preserves_sample(tmp_path: Path):
    d = ai.Diffusion()
    patch_before = d.sample_rgb_patch(steps=4, seed=9)
    ai.quantize_diffusion(d, "int8")
    path = tmp_path / "quant.mmn"
    ai.export_diffusion(d, "safetensors", str(path))
    loaded = ai.import_diffusion("safetensors", [str(path)])
    patch_after = loaded.sample_rgb_patch(steps=4, seed=9)
    assert patch_before == patch_after
