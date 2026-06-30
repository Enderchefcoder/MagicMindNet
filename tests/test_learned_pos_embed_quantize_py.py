"""Quantize learned position embedding: loss stays finite and within tolerance."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def _qa_dataset() -> ai.DatasetQA:
    return ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )


def test_quantize_int8_learned_pos_embed_preserves_mean_loss_within_tolerance():
    ds = _qa_dataset()
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=21,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    before = bot.compute_mean_loss(ds)
    ai.quantize(bot, "int8")
    after = bot.compute_mean_loss(ds)
    assert after > 0.0 and after == after
    rel = abs(after - before) / max(before, 1e-6)
    assert rel < 0.5, f"int8 quantize mean loss drift: {before} -> {after} (rel={rel})"


def test_quantize_int4_learned_pos_embed_preserves_mean_loss_within_tolerance():
    ds = _qa_dataset()
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=22,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    before = bot.compute_mean_loss(ds)
    ai.quantize(bot, "int4")
    after = bot.compute_mean_loss(ds)
    assert after > 0.0 and after == after
    rel = abs(after - before) / max(before, 1e-6)
    assert rel < 0.5, f"int4 quantize mean loss drift: {before} -> {after} (rel={rel})"


def test_quantize_learned_pos_embed_meta_unchanged(tmp_path: Path):
    bot = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=16,
        seed=23,
        use_learned_pos_embed=True,
        max_seq_len=32,
    )
    path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(path))
    ai.quantize(bot, "int8")
    assert bot.use_learned_pos_embed is True
    assert bot.max_seq_len == 32
    out = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(out))
    import json

    meta = json.loads(out.read_text(encoding="utf-8"))["meta"]
    assert meta.get("use_learned_pos_embed") is True
    assert meta.get("max_seq_len") == 32


def test_quantize_int8_learned_pos_embed_after_train_within_tolerance():
    ds = _qa_dataset()
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=24,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    ai.Train(bot, ds, ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05))
    before = bot.compute_mean_loss(ds)
    ai.quantize(bot, "int8")
    after = bot.compute_mean_loss(ds)
    assert after > 0.0 and after == after
    rel = abs(after - before) / max(before, 1e-6)
    assert rel < 0.5, f"post-train int8 quantize drift: {before} -> {after} (rel={rel})"


def test_quantize_int4_learned_pos_embed_after_train_within_tolerance():
    ds = _qa_dataset()
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=25,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    ai.Train(bot, ds, ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05))
    before = bot.compute_mean_loss(ds)
    ai.quantize(bot, "int4")
    after = bot.compute_mean_loss(ds)
    assert after > 0.0 and after == after
    rel = abs(after - before) / max(before, 1e-6)
    assert rel < 0.5, f"post-train int4 quantize drift: {before} -> {after} (rel={rel})"
