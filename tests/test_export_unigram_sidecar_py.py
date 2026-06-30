"""Chatbot export with unigram sidecar + load_unigram_sidecar helper."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_export_with_unigram_encoder_writes_sidecar_and_meta(tmp_path: Path):
    enc = ai.UnigramEncoder.train(["repeat repeat token"] * 10, vocab_size=400)
    sample = "repeat repeat hello"
    bot = ai.Chatbot(vocab_size=400, n_layer=1, d_model=32, seed=4)
    ckpt = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(ckpt), unigram_encoder=enc)
    sidecar = tmp_path / "bot.unigram.mmn"
    assert sidecar.is_file()
    meta = json.loads(ckpt.read_text(encoding="utf-8"))["meta"]
    assert meta["unigram_checkpoint"] == "bot.unigram.mmn"
    loaded = ai.load_unigram_sidecar(ckpt)
    assert loaded is not None
    assert loaded.encode(sample) == enc.encode(sample)


def test_load_unigram_sidecar_returns_none_without_meta(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32)
    ckpt = tmp_path / "plain.mmn"
    ai.export(bot, "safetensors", str(ckpt))
    assert ai.load_unigram_sidecar(ckpt) is None


def test_export_unigram_sidecar_roundtrip_loss(tmp_path: Path):
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    enc = ai.UnigramEncoder.train_from_qa(ds, vocab_size=512)
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=2)
    cfg = ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg, unigram_encoder=enc)
    loss_before = bot.compute_mean_loss(ds, unigram_encoder=enc)
    ckpt = tmp_path / "trained.mmn"
    ai.export(bot, "safetensors", str(ckpt), unigram_encoder=enc)
    loaded_bot = ai.import_model("safetensors", [str(ckpt)])
    loaded_enc = ai.load_unigram_sidecar(ckpt)
    assert loaded_enc is not None
    loss_after = loaded_bot.compute_mean_loss(ds, unigram_encoder=loaded_enc)
    assert abs(loss_before - loss_after) < 1e-3


def test_export_rejects_bpe_and_unigram_together(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32)
    bpe = ai.BytePairEncoder.train(["hello"], vocab_size=256, num_merges=4)
    uni = ai.UnigramEncoder.train(["hello"], vocab_size=256)
    ckpt = tmp_path / "bot.mmn"
    with pytest.raises(ValueError, match="at most one"):
        ai.export(bot, "safetensors", str(ckpt), bpe_encoder=bpe, unigram_encoder=uni)
