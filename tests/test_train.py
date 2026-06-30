from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_train_smoke():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    cfg = ai.TrainConfig(epochs=1, batch_size=2, cuda=False, optimizer="hybrid")
    ai.Train(bot, ds, cfg)


def test_rl_and_spin_smoke():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    cfg = ai.TrainConfig(epochs=1, batch_size=2, cuda=False)
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy")
    ai.SPIN(bot, 1, ds)


def test_merge_mismatch_raises():
    a = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    b = ai.Chatbot(vocab_size=512, n_layer=4, d_model=64)
    with pytest.raises(ai.ModelMismatchError):
        ai.merge(a, b)


def test_merge_same_shape():
    a = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    b = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    m = ai.merge(a, b)
    assert m.parameters > 0
