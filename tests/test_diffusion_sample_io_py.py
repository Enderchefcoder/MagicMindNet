"""Diffusion sampling and checkpoint IO."""

import math
from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_diffusion_sample_rgb_patch_finite_and_deterministic():
    d = ai.Diffusion()
    a = d.sample_rgb_patch(steps=6, seed=11)
    b = d.sample_rgb_patch(steps=6, seed=11)
    assert len(a) == ai.VISION_RGB_DIM
    assert all(math.isfinite(v) for v in a)
    assert a == b


def test_diffusion_export_import_roundtrip(tmp_path: Path):
    d = ai.Diffusion()
    patch_before = d.sample_rgb_patch(steps=4, seed=3)
    path = tmp_path / "diff.mmn"
    ai.export_diffusion(d, "safetensors", str(path))
    loaded = ai.import_diffusion("safetensors", [str(path)])
    patch_after = loaded.sample_rgb_patch(steps=4, seed=3)
    assert patch_before == patch_after


def test_import_diffusion_rejects_chatbot_checkpoint(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    try:
        ai.import_diffusion("safetensors", [str(path)])
    except Exception as exc:
        assert "diffusion" in str(exc).lower() or "mmn-diffusion" in str(exc)
    else:
        raise AssertionError("expected import_diffusion to reject chatbot checkpoint")
