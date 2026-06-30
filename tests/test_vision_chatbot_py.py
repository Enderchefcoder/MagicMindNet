"""Vision-flag chatbot path: train, quantize, and checkpoint roundtrip."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_vision_chatbot_trains_and_keeps_flag():
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, vision=True, seed=5)
    loss_before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    assert bot.has_vision is True
    loss_after = bot.compute_mean_loss(ds)
    assert loss_after < loss_before


def test_vision_chatbot_quantize_preserves_flag(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, vision=True, seed=2)
    ai.quantize(bot, "int8")
    path = tmp_path / "vision_quant.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.has_vision is True


def test_vision_repr_includes_flag():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, vision=True, seed=1)
    assert "vision=true" in repr(bot).lower()
