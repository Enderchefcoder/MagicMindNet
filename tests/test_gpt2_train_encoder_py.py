"""Tests: Gpt2BpeEncoder wired into training/generation (gpt2_encoder= kwarg)."""

from __future__ import annotations

import magicmindnet as ai
from magicmindnet._native import DataMismatchError, Gpt2BpeEncoder, TrainConfig
import pathlib
import pytest

FIXTURE_DIR = pathlib.Path(__file__).parent / "fixtures"
GLINT_MINI = FIXTURE_DIR / "glint_tokenizer_mini.json"

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _tiny_gpt2_encoder() -> Gpt2BpeEncoder:
    """Build a Gpt2BpeEncoder with 256 base byte tokens and a few merges."""
    def byte_to_unicode():
        printable = set(range(0x21, 0x7F)) | set(range(0xA1, 0xAD)) | set(range(0xAE, 0x100))
        table = {}
        n = 0
        for b in range(256):
            if b in printable:
                table[b] = chr(b)
            else:
                table[b] = chr(256 + n)
                n += 1
        return table

    table = byte_to_unicode()
    tokens = [table[b] for b in range(256)]
    tokens.append("he")
    tokens.append("hel")
    tokens.append("hello")
    merges = ["h e", "he l", "hel lo"]
    return Gpt2BpeEncoder.from_vocab(tokens, merges)


def _tiny_chatbot(vocab_size: int = 259) -> ai.Chatbot:
    return ai.Chatbot(
        vocab_size=vocab_size,
        n_layer=1,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        max_seq_len=32,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Encoder basics
# ---------------------------------------------------------------------------


def test_gpt2_encoder_from_vocab_basic():
    enc = _tiny_gpt2_encoder()
    assert enc.vocab_size == 259
    ids = enc.encode("hello")
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)
    text = enc.decode(ids)
    assert "hello" in text


def test_gpt2_encoder_repr():
    enc = _tiny_gpt2_encoder()
    r = repr(enc)
    assert "Gpt2BpeEncoder" in r
    assert "259" in r


def test_gpt2_encoder_from_hf_tokenizer_json():
    enc = Gpt2BpeEncoder.from_hf_tokenizer_json(str(GLINT_MINI))
    assert enc.vocab_size == 259
    ids = enc.encode("hello")
    text = enc.decode(ids)
    assert "hello" in text


def test_gpt2_encoder_from_hf_tokenizer_json_bad_path():
    with pytest.raises((ValueError, RuntimeError, OSError)):
        Gpt2BpeEncoder.from_hf_tokenizer_json("/nonexistent/path/tokenizer.json")


def test_gpt2_encoder_merges_as_strings(tmp_path):
    """from_hf_tokenizer_json also accepts 'left right' strings for merges."""
    import json

    def byte_to_unicode():
        printable = set(range(0x21, 0x7F)) | set(range(0xA1, 0xAD)) | set(range(0xAE, 0x100))
        table = {}
        n = 0
        for b in range(256):
            if b in printable:
                table[b] = chr(b)
            else:
                table[b] = chr(256 + n)
                n += 1
        return table

    table = byte_to_unicode()
    tokens = [table[b] for b in range(256)]
    tokens.append("ab")
    vocab = {tok: i for i, tok in enumerate(tokens)}
    data = {
        "model": {
            "type": "BPE",
            "vocab": vocab,
            "merges": ["a b"],  # string form
        }
    }
    p = tmp_path / "tok.json"
    p.write_text(json.dumps(data))
    enc = Gpt2BpeEncoder.from_hf_tokenizer_json(str(p))
    assert enc.vocab_size == 257


# ---------------------------------------------------------------------------
# Training with gpt2_encoder=
# ---------------------------------------------------------------------------


def test_chatbot_train_with_gpt2_encoder():
    enc = _tiny_gpt2_encoder()
    bot = _tiny_chatbot(vocab_size=259)
    data = ai.DatasetQA(
        data=[
            {"input": "hello", "output": "hello world"},
            {"input": "hi", "output": "hi there"},
        ]
        * 4
    )
    before = bot.compute_mean_loss(data, gpt2_encoder=enc)
    losses = bot.train(
        data, epochs=2, learning_rate=1e-2, batch_size=1, optimizer="adamw", gpt2_encoder=enc
    )
    after = bot.compute_mean_loss(data, gpt2_encoder=enc)
    assert len(losses) == 2
    assert all(isinstance(v, float) for v in losses)
    assert after < before


def test_ai_train_with_gpt2_encoder():
    enc = _tiny_gpt2_encoder()
    bot = _tiny_chatbot(vocab_size=259)
    data = ai.DatasetQA(
        data=[{"input": "test", "output": "result"}] * 4
    )
    cfg = TrainConfig(epochs=1, learning_rate=1e-2, batch_size=1, optimizer="adamw")
    losses = ai.Train(bot, data, cfg, gpt2_encoder=enc)
    assert len(losses) == 1


def test_chatbot_train_corpus_with_gpt2_encoder():
    enc = _tiny_gpt2_encoder()
    bot = _tiny_chatbot(vocab_size=259)
    data = ai.DatasetCorpus(
        data=["hello world hello", "test text here"] * 4
    )
    losses = bot.train(
        data, epochs=1, learning_rate=1e-2, batch_size=1, gpt2_encoder=enc
    )
    assert len(losses) == 1


# ---------------------------------------------------------------------------
# Generation with gpt2_encoder=
# ---------------------------------------------------------------------------


def test_chatbot_generate_with_gpt2_encoder():
    enc = _tiny_gpt2_encoder()
    bot = _tiny_chatbot(vocab_size=259)
    result = bot.generate("hello", max_new_tokens=4, temperature=0.0, gpt2_encoder=enc)
    assert isinstance(result, str)


def test_chatbot_chat_with_gpt2_encoder():
    enc = _tiny_gpt2_encoder()
    bot = _tiny_chatbot(vocab_size=259)
    result = bot.chat("hello", max_new_tokens=4, gpt2_encoder=enc)
    assert isinstance(result, str)


def test_chatbot_generate_stream_with_gpt2_encoder():
    enc = _tiny_gpt2_encoder()
    bot = _tiny_chatbot(vocab_size=259)
    pieces = bot.generate_stream("hello", max_new_tokens=4, temperature=0.0, gpt2_encoder=enc)
    assert isinstance(pieces, list)


def test_chatbot_generate_tokens_with_gpt2_encoder():
    enc = _tiny_gpt2_encoder()
    bot = _tiny_chatbot(vocab_size=259)
    ids = bot.generate_tokens("hello", max_new_tokens=4, temperature=0.0, gpt2_encoder=enc)
    assert isinstance(ids, list)
    assert all(isinstance(i, int) for i in ids)


def test_chatbot_compute_loss_with_gpt2_encoder():
    enc = _tiny_gpt2_encoder()
    bot = _tiny_chatbot(vocab_size=259)
    loss = bot.compute_loss("hello", "world", gpt2_encoder=enc)
    assert isinstance(loss, float)
    assert loss > 0


def test_chatbot_compute_mean_loss_with_gpt2_encoder():
    enc = _tiny_gpt2_encoder()
    bot = _tiny_chatbot(vocab_size=259)
    data = ai.DatasetQA(data=[{"input": "hello", "output": "world"}])
    loss = bot.compute_mean_loss(data, gpt2_encoder=enc)
    assert isinstance(loss, float)
    assert loss > 0


# ---------------------------------------------------------------------------
# Error: multiple encoders
# ---------------------------------------------------------------------------


def test_multiple_encoders_error():
    enc_gpt2 = _tiny_gpt2_encoder()
    enc_bpe = ai.BytePairEncoder.train(["hello world"], vocab_size=64, num_merges=8)
    bot = _tiny_chatbot(vocab_size=259)
    data = ai.DatasetQA(data=[{"input": "hi", "output": "bye"}])
    with pytest.raises((RuntimeError, ValueError, DataMismatchError)):
        bot.train(data, epochs=1, gpt2_encoder=enc_gpt2, bpe_encoder=enc_bpe)


# ---------------------------------------------------------------------------
# from_hf_tokenizer_json — loads mini Glint fixture and trains
# ---------------------------------------------------------------------------


def test_train_with_hf_tokenizer_json_fixture():
    enc = Gpt2BpeEncoder.from_hf_tokenizer_json(str(GLINT_MINI))
    bot = _tiny_chatbot(vocab_size=enc.vocab_size)
    data = ai.DatasetQA(
        data=[
            {"input": "hello", "output": "hello world"},
        ]
        * 4
    )
    losses = bot.train(
        data, epochs=1, learning_rate=1e-2, batch_size=1, gpt2_encoder=enc
    )
    assert len(losses) == 1


def test_generate_with_hf_tokenizer_json_fixture():
    enc = Gpt2BpeEncoder.from_hf_tokenizer_json(str(GLINT_MINI))
    bot = _tiny_chatbot(vocab_size=enc.vocab_size)
    result = bot.generate("hello", max_new_tokens=4, temperature=0.0, gpt2_encoder=enc)
    assert isinstance(result, str)
