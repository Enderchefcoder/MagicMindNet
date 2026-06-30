"""Inpainting diffusion training on image_edit fixtures."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_train_diffusion_edit_fixture_image():
    ds = ai.DatasetImageEdit(file=str(FIXTURES / "image_edit.json"))
    d = ai.Diffusion()
    cfg = ai.TrainConfig(epochs=6, batch_size=1, learning_rate=0.05, cuda=False)
    ai.TrainDiffusion(d, ds, cfg)
    assert d.smoke_step()


def test_sample_rgb_patch_values_in_unit_interval():
    d = ai.Diffusion()
    patch = d.sample_rgb_patch(steps=4, seed=42)
    assert len(patch) == 192
    assert all(0.0 <= v <= 1.0 for v in patch)


def test_merge_diffusion_averages_weights():
    a = ai.Diffusion()
    b = ai.Diffusion()
    merged = ai.merge_diffusion(a, b)
    assert merged.latent_channels == a.latent_channels
