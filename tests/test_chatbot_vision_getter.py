import magicmindnet as ai


def test_chatbot_has_vision_getter_reflects_constructor():
    plain = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, vision=False)
    vision = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, vision=True)
    assert plain.has_vision is False
    assert vision.has_vision is True
