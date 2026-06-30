import magicmindnet as ai


def test_chatbot_same_seed_same_compute_loss():
    a = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, seed=99)
    b = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, seed=99)
    loss_a = a.compute_loss("hello", "world")
    loss_b = b.compute_loss("hello", "world")
    assert loss_a == loss_b
