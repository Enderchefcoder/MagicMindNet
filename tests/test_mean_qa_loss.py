from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_compute_mean_loss_decreases_after_train():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32)
    before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=5, batch_size=1, cuda=False, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    assert after <= before, f"mean loss before={before} after={after}"
