"""BytePairEncoder Python binding and Train(bpe_encoder=...) smoke."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_byte_pair_encoder_train_encode():
    enc = ai.BytePairEncoder.train(
        ["hello world", "hello there", "world peace"],
        vocab_size=300,
        num_merges=8,
    )
    assert enc.merge_count > 0
    assert enc.vocab_size == 300
    ids = enc.encode("hello world")
    assert len(ids) >= 2
    assert all(0 <= i < 300 for i in ids)


def test_byte_pair_encoder_train_from_qa_smoke():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    enc = ai.BytePairEncoder.train_from_qa(ds, vocab_size=400, num_merges=12)
    assert enc.vocab_size == 400
    assert enc.encode("fixture text")


def test_train_with_bpe_encoder():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    texts = ["repeat repeat repeat token"] * 12
    enc = ai.BytePairEncoder.train(texts, vocab_size=512, num_merges=16)
    assert enc.merge_count > 0
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=64)
    cfg = ai.TrainConfig(epochs=2, batch_size=1, cuda=False, optimizer="adamw", learning_rate=0.05)
    ai.Train(bot, ds, cfg, bpe_encoder=enc)
