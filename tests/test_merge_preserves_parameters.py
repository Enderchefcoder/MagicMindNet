import magicmindnet as ai


def test_merge_chatbot_preserves_parameter_count():
    a = ai.Chatbot(vocab_size=80, n_layer=2, d_model=24, seed=1)
    b = ai.Chatbot(vocab_size=80, n_layer=2, d_model=24, seed=2)
    assert a.parameters == b.parameters
    merged = ai.merge(a, b)
    assert merged.parameters == a.parameters
