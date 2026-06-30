import pytest

import magicmindnet as ai


def test_merge_rejects_vocab_size_mismatch():
    a = ai.Chatbot(vocab_size=64, n_layer=2, d_model=32, seed=1)
    b = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, seed=2)
    with pytest.raises(ai.ModelMismatchError) as exc:
        ai.merge(a, b)
    msg = str(exc.value).lower()
    assert "merge" in msg or "size" in msg or "vocab" in msg or len(msg) > 0
