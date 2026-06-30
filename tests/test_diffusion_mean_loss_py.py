"""Diffusion compute_mean_denoise_loss on image datasets."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_compute_mean_denoise_loss_image_gen_finite():
    ds = ai.DatasetImageGen(file=str(FIXTURES / "image_gen.json"))
    d = ai.Diffusion()
    loss = d.compute_mean_denoise_loss(ds, t=7)
    assert loss == loss and loss >= 0.0


def test_compute_mean_denoise_loss_image_edit_finite():
    ds = ai.DatasetImageEdit(file=str(FIXTURES / "image_edit.json"))
    d = ai.Diffusion()
    loss = d.compute_mean_denoise_loss(ds, t=5)
    assert loss == loss and loss >= 0.0


def test_compute_mean_denoise_loss_rejects_qa_dataset():
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    d = ai.Diffusion()
    try:
        d.compute_mean_denoise_loss(ds)
    except ai.DataMismatchError:
        pass
    else:
        raise AssertionError("expected DataMismatchError for QA dataset")
