"""Corpus LM training via Train(DatasetCorpus)."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def _corpus_dataset() -> ai.DatasetCorpus:
    return ai.DatasetCorpus(
        use_two_files=True,
        rowfile=str(FIXTURES / "corpus_rows.json"),
        txtfile=str(FIXTURES / "corpus.txt"),
        sort_rows_by_complexity=True,
    )


def test_train_corpus_reduces_mean_loss():
    ds = _corpus_dataset()
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=11)
    before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=3, batch_size=2, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    assert after < before


def test_train_corpus_learned_pos_embed_reduces_mean_loss(tmp_path: Path):
    from conftest import checkpoint_tensor_bytes

    ds = _corpus_dataset()
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=12,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    before = bot.compute_mean_loss(ds)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    pe_before = checkpoint_tensor_bytes(before_path, "pos_embed")
    cfg = ai.TrainConfig(epochs=3, batch_size=2, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    assert after < before
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    pe_after = checkpoint_tensor_bytes(after_path, "pos_embed")
    assert pe_before != pe_after


def test_train_corpus_after_import_learned_pos_embed_reduces_loss(tmp_path: Path):
    from conftest import checkpoint_tensor_bytes

    ds = _corpus_dataset()
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=13,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    path = tmp_path / "learned_corpus.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.use_learned_pos_embed is True
    assert loaded.max_seq_len == 64
    pe_before = checkpoint_tensor_bytes(path, "pos_embed")
    before = loaded.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=3, batch_size=2, learning_rate=0.05)
    ai.Train(loaded, ds, cfg)
    after = loaded.compute_mean_loss(ds)
    assert after < before, f"post-import corpus loss before={before} after={after}"
    out = tmp_path / "after_train.mmn"
    ai.export(loaded, "safetensors", str(out))
    pe_after = checkpoint_tensor_bytes(out, "pos_embed")
    assert pe_before != pe_after


def test_compute_mean_loss_corpus_finite():
    ds = _corpus_dataset()
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    loss = bot.compute_mean_loss(ds)
    assert loss > 0.0
