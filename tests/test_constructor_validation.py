"""Constructor validation raises friendly `ValueError`s (never Rust panics)."""

import pytest

import magicmindnet as ai


def test_chatbot_rejects_learned_pe_and_rope_together():
    with pytest.raises(ValueError, match="use_learned_pos_embed"):
        ai.Chatbot(vocab_size=128, use_learned_pos_embed=True, use_rope=True)


def test_chatbot_rejects_unknown_autoset_preset():
    with pytest.raises(ValueError, match="sub-100M"):
        ai.Chatbot(autoset="mega-10T")


def test_chatbot_rejects_zero_vocab():
    with pytest.raises(ValueError, match="vocab_size"):
        ai.Chatbot(vocab_size=0)


def test_chatbot_accepts_underscore_autoset_spelling():
    bot = ai.Chatbot(autoset="sub_100m", vocab_size=8000)
    assert bot.parameters > 0


def test_train_config_rejects_unknown_optimizer():
    with pytest.raises(ValueError, match='"adamw", "muon", "hybrid"'):
        ai.TrainConfig(optimizer="sgd")


@pytest.mark.parametrize("name", ["adamw", "muon", "hybrid"])
def test_train_config_accepts_documented_optimizers(name):
    cfg = ai.TrainConfig(optimizer=name)
    assert cfg.optimizer == name


def test_train_config_verbose_defaults_false():
    cfg = ai.TrainConfig()
    assert cfg.verbose is False


def test_train_config_repr_shows_verbose():
    cfg = ai.TrainConfig(verbose=True)
    assert "verbose=True" in repr(cfg)


def test_muon_optimizer_trains():
    data = ai.DatasetQA(data=[{"input": "aa bb", "output": "cc dd"}])
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    losses = bot.train(data, epochs=1, optimizer="muon", learning_rate=0.05)
    assert len(losses) == 1
