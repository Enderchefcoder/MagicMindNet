import magicmindnet as ai


def test_chatbot_autoset_respects_init_seed():
    bot = ai.Chatbot(autoset="sub-100M", vocab_size=8000, seed=9)
    assert bot.init_seed == 9
