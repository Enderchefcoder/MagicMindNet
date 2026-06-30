"""Learned position embedding parameter count."""

import magicmindnet as ai


def test_learned_pos_embed_increases_parameters():
    sinusoidal = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=1)
    learned = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=16,
        seed=1,
        use_learned_pos_embed=True,
        max_seq_len=32,
    )
    assert learned.parameters == sinusoidal.parameters + 32 * 16
