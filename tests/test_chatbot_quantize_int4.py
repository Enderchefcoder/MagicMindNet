import math

import magicmindnet as ai


def test_quantize_chatbot_int4_runs():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, seed=1)
    ai.quantize(bot, "int4")
    loss = bot.compute_loss("ab", "cd")
    assert math.isfinite(loss)
