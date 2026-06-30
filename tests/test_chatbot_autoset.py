import magicmindnet as ai


def test_autoset_sub_100m_within_parameter_budget():
    bot = ai.Chatbot(autoset="sub-100M")
    assert bot.parameters < 100_000_000
    assert bot.layer_size >= 1


def test_autoset_repr_includes_parameters():
    bot = ai.Chatbot(autoset="sub-100M", seed=1)
    text = repr(bot)
    assert "Chatbot" in text
    assert "init_seed=1" in text
