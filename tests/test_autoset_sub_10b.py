import magicmindnet as ai


def test_autoset_sub_10b_within_parameter_budget():
    bot = ai.Chatbot(autoset="sub-10B", vocab_size=8000)
    assert bot.parameters <= 10_500_000_000
    assert bot.n_layer >= 2
