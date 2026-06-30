import magicmindnet as ai


def test_chatbot_layer_size_matches_n_layer():
    bot = ai.Chatbot(vocab_size=128, n_layer=4, d_model=32, seed=1)
    assert bot.layer_size == 4
    assert bot.n_layer == 4
    assert bot.parameters > 0
