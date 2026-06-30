import magicmindnet as ai


def test_model_mismatch_error_carries_message():
    a = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16)
    b = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16)
    try:
        ai.merge(a, b)
    except ai.ModelMismatchError as exc:
        assert len(str(exc)) > 0
    else:
        raise AssertionError("expected ModelMismatchError")
