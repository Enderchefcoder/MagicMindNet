"""Chatbot position encoding and causal attention getters."""

import magicmindnet as ai


def test_chatbot_uses_causal_attention_by_default():
    bot = ai.Chatbot(vocab_size=128, n_layer=2, d_model=16, seed=1)
    assert bot.uses_causal_attention is True


def test_same_token_different_positions_different_loss():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=2)
    loss_a = bot.compute_loss("a", "b")
    loss_b = bot.compute_loss("aa", "bb")
    assert loss_a != loss_b
