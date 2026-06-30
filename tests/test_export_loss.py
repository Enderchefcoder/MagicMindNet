"""Checkpoint roundtrip preserves loss metrics."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_export_import_preserves_compute_loss(tmp_path: Path):
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=11)
    before = bot.compute_mean_loss(ds)
    ckpt = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(ckpt))
    loaded = ai.import_model("safetensors", [str(ckpt)])
    after = loaded.compute_mean_loss(ds)
    assert abs(before - after) < 1e-4, f"mean loss drift: {before} vs {after}"


def test_export_import_preserves_learned_pos_embed_compute_loss(tmp_path: Path):
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=12,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    before = bot.compute_mean_loss(ds)
    ckpt = tmp_path / "learned.mmn"
    ai.export(bot, "safetensors", str(ckpt))
    loaded = ai.import_model("safetensors", [str(ckpt)])
    assert loaded.use_learned_pos_embed is True
    assert loaded.max_seq_len == 64
    after = loaded.compute_mean_loss(ds)
    assert abs(before - after) < 1e-4, f"learned PE mean loss drift: {before} vs {after}"


def test_export_import_preserves_learned_pos_embed_after_train(tmp_path: Path):
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=13,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    ai.Train(bot, ds, ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05))
    before = bot.compute_mean_loss(ds)
    ckpt = tmp_path / "learned_trained.mmn"
    ai.export(bot, "safetensors", str(ckpt))
    loaded = ai.import_model("safetensors", [str(ckpt)])
    assert loaded.use_learned_pos_embed is True
    after = loaded.compute_mean_loss(ds)
    assert abs(before - after) < 1e-4, f"trained learned PE drift: {before} vs {after}"


def test_export_import_preserves_classifier_compute_loss_with_seed(tmp_path: Path):
    import json

    cls_path = tmp_path / "cls.json"
    cls_path.write_text(
        json.dumps(
            [
                {"text": "good", "tag": "Happy"},
                {"text": "bad", "tag": "Sad"},
            ]
        ),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(cls_path), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=42)
    before = clf.compute_loss("good", "Happy")
    ckpt = tmp_path / "clf.mmn"
    ai.export_classifier(clf, "safetensors", str(ckpt))
    loaded = ai.import_classifier("safetensors", [str(ckpt)])
    after = loaded.compute_loss("good", "Happy")
    assert abs(before - after) < 1e-4
