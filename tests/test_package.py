"""Package metadata and public API surface."""

import magicmindnet as ai


def test_version():
    assert ai.__version__ == "0.1.0"


def test_public_train_api_names():
    assert callable(ai.Train)
    assert callable(ai.RL)
    assert callable(ai.SPIN)
    assert callable(ai.export)
    assert callable(ai.import_model)
    assert callable(ai.merge_classifier)


def test_chatbot_has_loss_helpers():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    assert hasattr(bot, "compute_mean_loss")
    assert hasattr(bot, "compute_loss")


def test_classifier_has_mean_loss():
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=1)
    assert hasattr(clf, "compute_mean_loss")


def test_diffusion_has_mean_denoise_loss():
    d = ai.Diffusion()
    assert hasattr(d, "compute_mean_denoise_loss")
