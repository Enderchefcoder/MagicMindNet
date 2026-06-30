import magicmindnet as ai


def test_chatbot_seed_is_deterministic():
    a = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=123)
    b = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=123)
    la = a.compute_loss("hi", "there")
    lb = b.compute_loss("hi", "there")
    assert la == lb


def test_chatbot_different_seeds_differ():
    a = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    b = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=2)
    assert a.compute_loss("hi", "there") != b.compute_loss("hi", "there")
