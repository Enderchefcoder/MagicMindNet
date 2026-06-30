"""Chatbot loss API for training verification."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_chatbot_compute_loss_decreases_after_train():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    import json

    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32)
    row = json.loads(path.read_text(encoding="utf-8"))[0]
    before = bot.compute_loss(row["input"], row["output"])
    cfg = ai.TrainConfig(epochs=5, batch_size=1, cuda=False, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after = bot.compute_loss(row["input"], row["output"])
    assert after <= before, f"loss should not increase: before={before} after={after}"


def test_learned_pos_embed_compute_mean_loss_decreases_after_train():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=4,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=5, batch_size=1, cuda=False, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    assert after < before, f"mean loss should decrease: before={before} after={after}"

