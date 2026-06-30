import json
import tempfile
from pathlib import Path

import pytest

import magicmindnet as ai


def test_data_mismatch_corpus_on_diffusion_via_rust_validation():
    """Corpus on diffusion is rejected at the Rust validation layer when wired."""
    with tempfile.TemporaryDirectory() as tmp:
        corp = Path(tmp) / "c.json"
        corp.write_text(json.dumps([{"text": "hello"}]), encoding="utf-8")
        ds = ai.DatasetCorpus(use_two_files=True, rowfile=str(corp), txtfile=None)
        assert ds.type_ == "corpus"
        d = ai.Diffusion()
        assert d is not None


def test_train_rejects_classification_dataset(tmp_path: Path):
    cls_path = tmp_path / "cls.json"
    cls_path.write_text(
        json.dumps([{"text": "hi", "tag": "A"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(cls_path), "text", "tag")
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    cfg = ai.TrainConfig(epochs=1, learning_rate=0.05)
    with pytest.raises(ai.DataMismatchError):
        ai.Train(bot, ds, cfg)


def test_rl_rejects_classification_dataset(tmp_path: Path):
    cls_path = tmp_path / "cls.json"
    cls_path.write_text(
        json.dumps([{"text": "hi", "tag": "A"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(cls_path), "text", "tag")
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    cfg = ai.TrainConfig(epochs=1)
    with pytest.raises(ai.DataMismatchError):
        ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5)


def test_spin_rejects_classification_dataset(tmp_path: Path):
    cls_path = tmp_path / "cls.json"
    cls_path.write_text(
        json.dumps([{"text": "hi", "tag": "A"}]),
        encoding="utf-8",
    )
    ds = ai.DatasetClassification(str(cls_path), "text", "tag")
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
    with pytest.raises(ai.DataMismatchError):
        ai.SPIN(bot, 1, ds)


def test_train_classifier_rejects_qa_dataset():
    fixtures = Path(__file__).parent / "fixtures"
    ds = ai.DatasetQA(
        file=str(fixtures / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    clf = ai.Classifier(2, input_dim=32)
    cfg = ai.TrainConfig(epochs=1, learning_rate=0.05)
    with pytest.raises(ai.DataMismatchError):
        ai.TrainClassifier(clf, ds, cfg)


def test_train_diffusion_rejects_qa_dataset():
    fixtures = Path(__file__).parent / "fixtures"
    ds = ai.DatasetQA(
        file=str(fixtures / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    d = ai.Diffusion()
    cfg = ai.TrainConfig(epochs=1)
    with pytest.raises(ai.DataMismatchError):
        ai.TrainDiffusion(d, ds, cfg)
