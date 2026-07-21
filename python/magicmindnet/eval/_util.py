"""Shared helpers for eval tasks."""

from __future__ import annotations

import json
import tempfile
from pathlib import Path

import magicmindnet as ai

_PKG_ROOT = Path(__file__).resolve().parents[3]
FIXTURES = _PKG_ROOT / "tests" / "fixtures"


def fixtures_dir() -> Path:
    return FIXTURES


def qa_dataset() -> ai.DatasetQA:
    return ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )


def corpus_dataset() -> ai.DatasetCorpus:
    return ai.DatasetCorpus(
        use_two_files=True,
        rowfile=str(FIXTURES / "corpus_rows.json"),
        txtfile=str(FIXTURES / "corpus.txt"),
    )


def classification_dataset(work_dir: Path | None = None) -> ai.DatasetClassification:
    """Prefer the repo fixture; fall back to a tiny temp JSON set."""
    fixture = FIXTURES / "labels_small.json"
    if fixture.is_file():
        # labels_small.json uses text/tag columns (pos/neg).
        return ai.DatasetClassification(str(fixture), "text", "tag")
    data = [
        {"text": "great day", "tag": "Happy"},
        {"text": "awful day", "tag": "Sad"},
        {"text": "nice day", "tag": "Happy"},
        {"text": "terrible day", "tag": "Sad"},
        {"text": "wonderful", "tag": "Happy"},
        {"text": "horrible", "tag": "Sad"},
    ]
    if work_dir is not None:
        path = Path(work_dir) / "eval_cls.json"
        path.write_text(json.dumps(data), encoding="utf-8")
        return ai.DatasetClassification(str(path), "text", "tag")
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(data, f)
        path = f.name
    return ai.DatasetClassification(path, "text", "tag")


def tiny_chatbot(
    seed: int,
    *,
    vocab_size: int = 256,
    n_layer: int = 2,
    d_model: int = 32,
    **kwargs,
) -> ai.Chatbot:
    return ai.Chatbot(
        vocab_size=vocab_size,
        n_layer=n_layer,
        d_model=d_model,
        seed=seed,
        **kwargs,
    )


def train_cfg(epochs: int = 3, lr: float = 0.05) -> ai.TrainConfig:
    return ai.TrainConfig(epochs=epochs, batch_size=1, learning_rate=lr, cuda=False)


def ensure_work(work_dir: Path | None) -> Path:
    if work_dir is None:
        return Path(tempfile.mkdtemp(prefix="mmn-eval-"))
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    return work_dir
