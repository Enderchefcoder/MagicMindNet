import math
from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_chatbot_compute_mean_loss_is_finite_before_train():
    ds = ai.DatasetQA(file=str(FIXTURES / "qa_valid.json"), user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=1)
    loss = bot.compute_mean_loss(ds)
    assert math.isfinite(loss)
    assert loss > 0.0
