"""Unigram tokenizer training and generation."""

from pathlib import Path

import magicmindnet as ai


def test_unigram_roundtrip():
    enc = ai.UnigramEncoder.train(["hello world", "hello there"], vocab_size=400)
    assert enc.piece_count >= 256
    ids = enc.encode("hello")
    assert enc.decode(ids) == "hello"


def test_unigram_train_from_qa():
    data = ai.DatasetQA(
        file="tests/fixtures/qa_valid.json",
        user_row="input",
        ai_row="output",
    )
    enc = ai.UnigramEncoder.train_from_qa(data, vocab_size=400)
    ids = enc.encode("What")
    assert isinstance(ids, list)
    assert len(ids) >= 1


def test_unigram_checkpoint_roundtrip(tmp_path: Path):
    enc = ai.UnigramEncoder.train(["abc abc abc"], vocab_size=320)
    path = tmp_path / "uni.json"
    enc.save(str(path))
    loaded = ai.UnigramEncoder.load(str(path))
    assert loaded.encode("abc") == enc.encode("abc")


def test_train_with_unigram_encoder():
    data = ai.DatasetQA(
        file="tests/fixtures/qa_valid.json",
        user_row="input",
        ai_row="output",
    )
    enc = ai.UnigramEncoder.train_from_qa(data, vocab_size=512)
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=7)
    before = bot.compute_mean_loss(data, unigram_encoder=enc)
    ai.Train(bot, data, ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05), unigram_encoder=enc)
    after = bot.compute_mean_loss(data, unigram_encoder=enc)
    assert after < before


def test_generate_top_p_and_repetition_penalty():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=21)
    out = bot.generate(
        "hi",
        max_new_tokens=6,
        temperature=0.8,
        top_p=0.9,
        repetition_penalty=1.2,
    )
    assert isinstance(out, str)
