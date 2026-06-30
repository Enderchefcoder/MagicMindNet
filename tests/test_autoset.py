import magicmindnet as ai


def test_autoset_sub_100m_within_budget():
    bot = ai.Chatbot(autoset="sub-100M", vocab_size=8000)
    assert bot.parameters <= 105_000_000
    assert bot.layer_size >= 1
