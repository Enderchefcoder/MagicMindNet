import magicmindnet as ai


def test_chatbot_tokenizer_getter_is_non_empty_string():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16)
    assert isinstance(bot.tokenizer, str)
    assert len(bot.tokenizer) > 0
