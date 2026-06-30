import magicmindnet as ai


def test_merge_chatbot_or_vision_flag():
    plain = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, vision=False, seed=1)
    vision = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, vision=True, seed=2)
    merged = ai.merge(plain, vision)
    assert merged.has_vision is True
