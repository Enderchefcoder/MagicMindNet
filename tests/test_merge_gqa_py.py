"""Merge rejects mismatched GQA head counts."""

import pytest

import magicmindnet as ai


def test_merge_rejects_n_kv_heads_mismatch():
    a = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, n_heads=4, n_kv_heads=2, seed=1)
    b = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, n_heads=4, n_kv_heads=4, seed=2)
    with pytest.raises(ai.ModelMismatchError):
        ai.merge(a, b)
