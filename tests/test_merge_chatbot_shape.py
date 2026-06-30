import magicmindnet as ai


def test_merge_chatbot_preserves_shape_getters():
    a = ai.Chatbot(vocab_size=100, n_layer=2, d_model=32, seed=1)
    b = ai.Chatbot(vocab_size=100, n_layer=2, d_model=32, seed=2)
    merged = ai.merge(a, b)
    assert merged.vocab_size == 100
    assert merged.n_layer == 2
    assert merged.d_model == 32
