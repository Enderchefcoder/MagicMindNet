import magicmindnet as ai


def test_autoset_sub_1b_within_parameter_budget():
    bot = ai.Chatbot(autoset="sub-1B", vocab_size=8000)
    assert bot.parameters <= 1_050_000_000
    assert bot.n_layer >= 2
    assert bot.d_model >= 64
