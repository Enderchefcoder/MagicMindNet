import magicmindnet as ai


def test_chatbot_exposes_vocab_size_n_layer_d_model_getters():
    bot = ai.Chatbot(vocab_size=200, n_layer=3, d_model=48, seed=1)
    assert bot.vocab_size == 200
    assert bot.n_layer == 3
    assert bot.d_model == 48
