"""Method-style training: `bot.train(...)`, `clf.train(...)`, `diff.train(...)`."""

from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"

QA_ROWS = [
    {"input": "repeat repeat one", "output": "repeat repeat two"},
    {"input": "say hi", "output": "hello"},
]


def make_qa():
    return ai.DatasetQA(data=QA_ROWS)


def test_chatbot_train_returns_loss_per_epoch():
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=42)
    losses = bot.train(make_qa(), epochs=3, learning_rate=0.05, optimizer="adamw")
    assert len(losses) == 3
    assert all(loss > 0 for loss in losses)
    assert losses[-1] < losses[0]


def test_chatbot_train_accepts_train_config_object():
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=1)
    cfg = ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05, optimizer="adamw")
    losses = bot.train(make_qa(), cfg)
    assert len(losses) == 2


def test_chatbot_train_kwargs_override_config():
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=1)
    cfg = ai.TrainConfig(epochs=1)
    losses = bot.train(make_qa(), cfg, epochs=4)
    assert len(losses) == 4


def test_chatbot_train_defaults_run_one_epoch():
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=1)
    losses = bot.train(make_qa())
    assert len(losses) == 1


def test_chatbot_train_rejects_wrong_dataset():
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32)
    cds = ai.DatasetClassification(data=[{"text": "x", "label": "a"}])
    with pytest.raises(ai.DataMismatchError):
        bot.train(cds)


def test_chatbot_train_rejects_unknown_optimizer():
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32)
    with pytest.raises(ValueError, match="sgd"):
        bot.train(make_qa(), optimizer="sgd")


def test_train_function_returns_losses():
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=2)
    cfg = ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05, optimizer="adamw")
    losses = ai.Train(bot, make_qa(), cfg)
    assert len(losses) == 2


def test_train_verbose_prints_epoch_loss(capfd):
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=2)
    bot.train(make_qa(), epochs=2, verbose=True)
    out = capfd.readouterr().out
    assert "epoch 1/2" in out and "mean loss" in out


def test_train_not_verbose_by_default(capfd):
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=2)
    bot.train(make_qa(), epochs=1)
    assert "epoch" not in capfd.readouterr().out


def test_classifier_train_returns_loss_per_epoch():
    ds = ai.DatasetClassification(
        data=[{"text": "sun", "label": "happy"}, {"text": "rain", "label": "sad"}]
    )
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=7)
    losses = clf.train(ds, epochs=3, learning_rate=0.05)
    assert len(losses) == 3
    assert all(loss > 0 for loss in losses)


def test_train_classifier_function_returns_losses():
    ds = ai.DatasetClassification(
        data=[{"text": "sun", "label": "happy"}, {"text": "rain", "label": "sad"}]
    )
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=7)
    cfg = ai.TrainConfig(epochs=2, learning_rate=0.05)
    losses = ai.TrainClassifier(clf, ds, cfg)
    assert len(losses) == 2


def test_classifier_train_rejects_wrong_dataset():
    clf = ai.Classifier(num_labels=2, input_dim=16)
    with pytest.raises(ai.DataMismatchError):
        clf.train(make_qa())


def test_diffusion_train_returns_loss_per_epoch():
    ds = ai.DatasetImageGen(file=str(FIXTURES / "image_gen.json"))
    diff = ai.Diffusion()
    losses = diff.train(ds, epochs=2, learning_rate=0.05)
    assert len(losses) == 2
    assert all(loss >= 0 for loss in losses)


def test_diffusion_train_rejects_wrong_dataset():
    diff = ai.Diffusion()
    with pytest.raises(ai.DataMismatchError):
        diff.train(make_qa())
