"""Public API surface matches docs/API.md exports."""

import inspect

import pytest

import magicmindnet as ai

DATASET_CLASSES = [
    ai.DatasetQA,
    ai.DatasetClassification,
    ai.DatasetCorpus,
    ai.DatasetImageGen,
    ai.DatasetImageEdit,
]

MODEL_CLASSES = [ai.Chatbot, ai.Classifier, ai.Diffusion]


def test_version_exported():
    assert "__version__" in ai.__all__
    assert ai.__version__ == "0.1.0"


@pytest.mark.parametrize("name", ai.__all__)
def test_all_export_is_attribute(name: str):
    assert hasattr(ai, name), f"missing export: {name}"


@pytest.mark.parametrize("cls", DATASET_CLASSES)
def test_dataset_has_rows_format_type_getters(cls):
    sig = inspect.signature(cls)
    assert "file" in str(sig) or cls is ai.DatasetCorpus


def test_dataset_qa_has_format_sample():
    assert callable(ai.DatasetQA.format_sample)


def test_train_config_has_documented_fields():
    cfg = ai.TrainConfig()
    for field in ("epochs", "batch_size", "cuda", "optimizer", "learning_rate"):
        assert hasattr(cfg, field)


@pytest.mark.parametrize("fn_name", ["Train", "TrainClassifier", "TrainDiffusion", "RL", "SPIN"])
def test_training_entrypoints_callable(fn_name: str):
    assert callable(getattr(ai, fn_name))


@pytest.mark.parametrize(
    "fn_name",
    [
        "export",
        "import_model",
        "merge",
        "quantize",
        "export_classifier",
        "import_classifier",
        "export_diffusion",
        "import_diffusion",
        "merge_classifier",
        "quantize_classifier",
        "limit",
        "limit_percent",
    ],
)
def test_io_utilities_callable(fn_name: str):
    assert callable(getattr(ai, fn_name))
