"""Smoke tests for split `mmn-py` PyO3 modules (datasets, models, train, io)."""

from __future__ import annotations

import inspect

import magicmindnet._native as native

import magicmindnet as ai


def test_native_module_exports_core_symbols():
    for name in (
        "DatasetQA",
        "DatasetCorpus",
        "DatasetClassification",
        "Chatbot",
        "Classifier",
        "Diffusion",
        "TrainConfig",
        "Train",
        "RL",
        "SPIN",
        "export",
        "import_model",
        "merge",
        "quantize",
    ):
        assert hasattr(native, name), f"_native missing {name}"


def test_dataset_pyclass_names_match_public_api():
    expected_names = {
        ai.DatasetQA: "DatasetQA",
        ai.DatasetCorpus: "DatasetCorpus",
        ai.DatasetClassification: "DatasetClassification",
        ai.DatasetImageGen: "DatasetImageGen",
        ai.DatasetImageEdit: "DatasetImageEdit",
    }
    for cls, name in expected_names.items():
        assert cls.__name__ == name


def test_chatbot_and_classifier_construct_from_split_modules():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=32, seed=3)
    clf = ai.Classifier.with_labels(["a", "b"], input_dim=8, seed=3)
    assert "Chatbot" in repr(bot)
    assert "Classifier" in repr(clf)
    assert bot.init_seed == 3
    assert bot.uses_causal_attention is True
    assert bot.use_learned_pos_embed is False
    assert bot.max_seq_len == 512
    assert clf.labels == ["a", "b"]


def test_chatbot_learned_pos_embed_option():
    bot = ai.Chatbot(
        vocab_size=64,
        n_layer=1,
        d_model=16,
        seed=4,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    assert bot.use_learned_pos_embed is True
    assert bot.max_seq_len == 64


def test_diffusion_from_models_module():
    d = ai.Diffusion()
    assert d.latent_channels > 0
    assert d.smoke_step() is True


def test_train_config_to_string_includes_epochs():
    cfg = ai.TrainConfig(epochs=2, batch_size=4, cuda=False)
    text = repr(cfg)
    assert "epochs" in text.lower() or "2" in text


def test_data_mismatch_error_is_exception_subclass():
    assert issubclass(ai.DataMismatchError, Exception)


def test_io_wrappers_accept_split_model_types(tmp_path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=32, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.vocab_size == bot.vocab_size
    assert inspect.isclass(type(loaded))
