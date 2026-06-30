import pytest

import magicmindnet as ai


def test_merge_rejects_n_layer_mismatch():
    a = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64, seed=1)
    b = ai.Chatbot(vocab_size=512, n_layer=4, d_model=64, seed=2)
    with pytest.raises(ai.ModelMismatchError):
        ai.merge(a, b)
