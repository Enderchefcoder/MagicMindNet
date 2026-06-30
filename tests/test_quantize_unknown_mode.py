import pytest

import magicmindnet as ai


def test_quantize_chatbot_rejects_unknown_mode():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    with pytest.raises(RuntimeError, match="Unknown quant mode"):
        ai.quantize(bot, "fp16")


def test_quantize_classifier_rejects_unknown_mode():
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=1)
    with pytest.raises(RuntimeError, match="Unknown quant mode"):
        ai.quantize_classifier(clf, "bf16")
