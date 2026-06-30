"""Learned position embedding max_seq_len guard."""

import pytest

import magicmindnet as ai


def test_compute_loss_rejects_sequence_longer_than_max_seq_len():
    bot = ai.Chatbot(
        vocab_size=512,
        n_layer=1,
        d_model=16,
        seed=1,
        use_learned_pos_embed=True,
        max_seq_len=4,
    )
    long_input = "a" * 20
    long_target = "b" * 20
    with pytest.raises(Exception) as exc:
        bot.compute_loss(long_input, long_target)
    msg = str(exc.value).lower()
    assert "max_seq_len" in msg or "sequence" in msg or "exceeds" in msg
