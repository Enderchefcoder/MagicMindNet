"""Diffusion training on image_gen fixtures."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_train_diffusion_fixture_image():
    ds = ai.DatasetImageGen(file=str(FIXTURES / "image_gen.json"))
    img = FIXTURES / "samples" / "cat.png"
    d = ai.Diffusion()
    before = d.denoise_loss_on_image(str(img), 3)
    cfg = ai.TrainConfig(epochs=12, batch_size=1, learning_rate=0.05, cuda=False)
    ai.TrainDiffusion(d, ds, cfg)
    after = d.denoise_loss_on_image(str(img), 3)
    assert before == before and after == after
    assert d.smoke_step()


def test_train_diffusion_rejects_qa_dataset():
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    d = ai.Diffusion()
    cfg = ai.TrainConfig(epochs=1)
    try:
        ai.TrainDiffusion(d, ds, cfg)
    except ai.DataMismatchError:
        pass
    else:
        raise AssertionError("expected DataMismatchError for QA dataset")

