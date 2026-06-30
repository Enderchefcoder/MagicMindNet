"""Optimizer integration smoke — hybrid default TrainConfig runs on fixture QA."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_hybrid_default_train_config_runs():
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=99)
    cfg = ai.TrainConfig(epochs=2, batch_size=2, learning_rate=0.05)
    assert cfg.optimizer == "hybrid"
    before = bot.compute_mean_loss(ds)
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    assert after <= before


def test_adamw_train_config_runs():
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=7)
    cfg = ai.TrainConfig(epochs=3, batch_size=2, learning_rate=0.05, optimizer="adamw")
    before = bot.compute_mean_loss(ds)
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    assert after < before
