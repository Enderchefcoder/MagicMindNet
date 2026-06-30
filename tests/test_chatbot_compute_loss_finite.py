import math

import magicmindnet as ai


def test_chatbot_compute_loss_is_finite_positive():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    loss = bot.compute_loss("hello", "world")
    assert math.isfinite(loss)
    assert loss > 0.0
