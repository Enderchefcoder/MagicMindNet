"""Classifier eval tasks."""

from __future__ import annotations

import time
from pathlib import Path

import magicmindnet as ai
from magicmindnet.eval._util import classification_dataset
from magicmindnet.eval.registry import register
from magicmindnet.eval.types import Metric, TaskResult


@register("cls_mean_loss", suites=("smoke", "cls"), description="Classifier mean CE")
def cls_mean_loss(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    t0 = time.perf_counter()
    ds = classification_dataset(work_dir)
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=seed)
    loss = float(clf.compute_mean_loss(ds))
    return TaskResult(
        name="cls_mean_loss",
        ok=loss > 0 and loss == loss,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[Metric("mean_loss", loss, higher_is_better=False)],
    )


@register("cls_train", suites=("smoke", "cls", "train"), description="Classifier train delta")
def cls_train(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    t0 = time.perf_counter()
    ds = classification_dataset(work_dir)
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=seed)
    before = float(clf.compute_mean_loss(ds))
    cfg = ai.TrainConfig(
        epochs=20,
        batch_size=2,
        learning_rate=0.05,
        optimizer="adamw",
        cuda=False,
    )
    ai.TrainClassifier(clf, ds, cfg)
    after = float(clf.compute_mean_loss(ds))
    return TaskResult(
        name="cls_train",
        ok=after <= before + 1e-6,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("loss_before", before, higher_is_better=False),
            Metric("loss_after", after, higher_is_better=False),
            Metric("loss_delta", after - before, higher_is_better=False),
        ],
    )


@register("cls_accuracy", suites=("cls",), description="Classifier label accuracy on train set")
def cls_accuracy(*, seed: int = 1, work_dir: Path | None = None) -> TaskResult:
    t0 = time.perf_counter()
    ds = classification_dataset(work_dir)
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=seed)
    cfg = ai.TrainConfig(
        epochs=20,
        batch_size=2,
        learning_rate=0.05,
        optimizer="adamw",
        cuda=False,
    )
    ai.TrainClassifier(clf, ds, cfg)
    pairs = list(ds.as_pairs())
    correct = 0
    for text, label in pairs:
        pred = clf.predict_label(text)
        if pred == label:
            correct += 1
    acc = correct / max(len(pairs), 1)
    return TaskResult(
        name="cls_accuracy",
        ok=0.0 <= acc <= 1.0,
        elapsed_ms=(time.perf_counter() - t0) * 1000,
        metrics=[
            Metric("accuracy", acc, higher_is_better=True),
            Metric("n_examples", float(len(pairs))),
        ],
    )
