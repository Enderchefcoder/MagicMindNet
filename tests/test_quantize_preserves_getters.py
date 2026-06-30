import magicmindnet as ai


def test_quantize_int8_preserves_chatbot_shape_getters():
    bot = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, seed=1)
    before = (bot.vocab_size, bot.n_layer, bot.d_model, bot.parameters)
    ai.quantize(bot, "int8")
    after = (bot.vocab_size, bot.n_layer, bot.d_model, bot.parameters)
    assert after == before
