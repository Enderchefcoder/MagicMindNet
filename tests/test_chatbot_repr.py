import magicmindnet as ai


def test_chatbot_repr_lists_shape():
    bot = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, seed=9)
    text = repr(bot)
    assert "Chatbot" in text
    assert "vocab_size=128" in text
    assert "n_layer=2" in text
    assert "init_seed=9" in text
