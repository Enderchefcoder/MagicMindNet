from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_train_batch_size_two_reduces_mean_loss():
    ds = ai.DatasetQA(file=str(FIXTURES / "qa_valid.json"), user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=7)
    before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=5, batch_size=2, learning_rate=0.05, optimizer="adamw")
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    assert after < before, f"mean loss before={before} after={after}"
