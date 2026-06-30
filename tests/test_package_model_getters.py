import magicmindnet as ai


def test_chatbot_exposes_documented_getters():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, vision=False, seed=1)
    for name in (
        "vocab_size",
        "n_layer",
        "d_model",
        "parameters",
        "layer_size",
        "tokenizer",
        "has_vision",
        "init_seed",
    ):
        assert hasattr(bot, name), name


def test_classifier_exposes_documented_getters():
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=1)
    for name in ("labels", "num_labels", "input_dim", "init_seed"):
        assert hasattr(clf, name), name


def test_version_in_public_module():
    assert ai.__version__ == "0.1.0"
