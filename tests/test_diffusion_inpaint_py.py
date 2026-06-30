"""Diffusion inpainting sample, masked loss, quantize, and PNG export."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_denoise_loss_on_image_masked_fixture():
    d = ai.Diffusion()
    img = FIXTURES / "samples" / "photo.png"
    mask = FIXTURES / "samples" / "mask.png"
    loss = d.denoise_loss_on_image_masked(str(img), str(mask), 5)
    assert loss == loss


def test_sample_inpaint_rgb_patch_finite_and_clamped():
    d = ai.Diffusion()
    img = FIXTURES / "samples" / "photo.png"
    mask = FIXTURES / "samples" / "mask.png"
    patch = d.sample_inpaint_rgb_patch(str(img), str(mask), steps=4, seed=7)
    assert len(patch) == ai.VISION_RGB_DIM
    assert all(0.0 <= v <= 1.0 for v in patch)


def test_sample_rgb_patch_to_png_roundtrip(tmp_path: Path):
    d = ai.Diffusion()
    out = tmp_path / "sample.png"
    d.sample_rgb_patch_to_png(str(out), steps=3, seed=1)
    assert out.is_file()
    assert out.stat().st_size > 0


def test_quantize_diffusion_int8_runs():
    d = ai.Diffusion()
    ai.quantize_diffusion(d, "int8")
    assert d.smoke_step()


def test_quantize_diffusion_rejects_unknown_mode():
    d = ai.Diffusion()
    try:
        ai.quantize_diffusion(d, "bf16")
    except Exception:
        pass
    else:
        raise AssertionError("expected quantize_diffusion to reject unknown mode")
