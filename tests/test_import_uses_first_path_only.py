from pathlib import Path

import magicmindnet as ai


def test_import_model_uses_first_path_only(tmp_path: Path):
    bot_a = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=1)
    bot_b = ai.Chatbot(vocab_size=512, n_layer=2, d_model=32, seed=2)
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    ai.export(bot_a, "safetensors", str(path_a))
    ai.export(bot_b, "safetensors", str(path_b))
    loaded = ai.import_model("safetensors", [str(path_a), str(path_b)])
    assert loaded.vocab_size == 128
    assert loaded.n_layer == 1
    assert loaded.d_model == 16


def test_import_classifier_uses_first_path_only(tmp_path: Path):
    clf_a = ai.Classifier.with_labels(["x", "y"], input_dim=8, seed=1)
    clf_b = ai.Classifier.with_labels(["a", "b", "c"], input_dim=16, seed=2)
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    ai.export_classifier(clf_a, "safetensors", str(path_a))
    ai.export_classifier(clf_b, "safetensors", str(path_b))
    loaded = ai.import_classifier("safetensors", [str(path_a), str(path_b)])
    assert loaded.input_dim == 8
    assert loaded.num_labels == 2
