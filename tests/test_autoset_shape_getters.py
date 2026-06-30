import magicmindnet as ai


def test_autoset_chatbot_exposes_positive_shape_getters():
    bot = ai.Chatbot(autoset="sub-100M", vocab_size=8000)
    assert bot.vocab_size == 8000
    assert bot.n_layer >= 1
    assert bot.d_model >= 1
