"""Vision patch encoder forward path and checkpoint roundtrip."""

from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_vision_chatbot_has_patch_encoder():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, vision=True, seed=1)
    assert bot.has_vision is True
    assert bot.has_vision_patch_encoder is True
    assert bot.vision_patch_dim == ai.VISION_PATCH_DIM == 64


def test_vision_patch_changes_compute_loss():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, vision=True, seed=2)
    loss_default_patch = bot.compute_loss("hi", "hello")
    patch = ai.vision_patch_from_text("alternate patch bytes")
    loss_custom_patch = bot.compute_loss("hi", "hello", image_patch=patch)
    assert loss_default_patch != loss_custom_patch


def test_vision_patch_wrong_length_raises():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, vision=True, seed=3)
    with pytest.raises(ai.DataMismatchError):
        bot.compute_loss("a", "b", image_patch=[0.1, 0.2])


def test_vision_patch_proj_checkpoint_roundtrip(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, vision=True, seed=4)
    w_before = bot.compute_loss("x", "y")  # touch forward
    del w_before
    path = tmp_path / "vision_patch.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.has_vision_patch_encoder is True
    assert loaded.compute_loss("x", "y") == pytest.approx(bot.compute_loss("x", "y"))


def test_vision_chatbot_trains_patch_encoder(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, vision=True, seed=5)
    loss_before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    loss_after = bot.compute_mean_loss(ds)
    assert loss_after < loss_before
