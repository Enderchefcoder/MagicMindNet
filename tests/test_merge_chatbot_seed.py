import magicmindnet as ai


def test_merge_preserves_init_seed_from_first_model():
    a = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, seed=11)
    b = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, seed=22)
    merged = ai.merge(a, b)
    assert merged.init_seed == 11
