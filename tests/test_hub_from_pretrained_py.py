"""Universal hub loading: from_pretrained for HF / ModelScope / Ollama / local.

Any repo id or local path should resolve to a runnable model with
generate / predict / train / finetune when the task supports it.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import magicmindnet as ai


def test_from_pretrained_exported():
    assert callable(ai.from_pretrained)
    assert "from_pretrained" in ai.__all__


def test_inspect_source_classifies_pipeline_tags():
    from magicmindnet.hub import inspect_source

    card = inspect_source(
        {
            "pipeline_tag": "text-generation",
            "architectures": ["Qwen3ForCausalLM"],
            "files": ["config.json", "model.safetensors"],
        }
    )
    assert card.family in {"causal-lm", "chatbot", "llm"}
    card2 = inspect_source(
        {
            "pipeline_tag": "text-classification",
            "architectures": ["RobertaForSequenceClassification"],
            "files": ["config.json", "pytorch_model.bin"],
        }
    )
    assert card2.family in {"classifier", "seq-cls", "text-classification"}
    card3 = inspect_source(
        {
            "pipeline_tag": "text-to-image",
            "model_index": {"_class_name": "StableDiffusionPipeline"},
            "files": ["model_index.json"],
        }
    )
    assert card3.family in {"diffusion", "text-to-image", "image"}
    card4 = inspect_source(
        {
            "pipeline_tag": "text-to-video",
            "model_index": {"_class_name": "WanPipeline"},
            "files": ["model_index.json"],
        }
    )
    assert card4.family == "video"
    card5 = inspect_source(
        {
            "pipeline_tag": "text-to-speech",
            "files": ["config.json", "model.pth"],
            "tags": ["text-to-speech"],
        }
    )
    assert card5.family == "tts"


def test_resolve_local_checkpoint_via_from_pretrained(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "tiny.mmn"
    bot.save(str(path))
    loaded = ai.from_pretrained(str(path))
    assert getattr(loaded, "vocab_size", None) == 64
    assert hasattr(loaded, "generate") or hasattr(loaded, "chat") or hasattr(loaded, "predict")


def test_hub_model_finetune_classifier_api():
    """Emotion-style seq-cls finetune surface (no network — synthetic HubModel)."""
    from magicmindnet.hub import HubModel

    model = HubModel.synthetic_classifier(labels=["anger", "joy", "sadness"])
    data = ai.DatasetClassification(
        data=[
            {"text": "I am so happy today", "label": "joy"},
            {"text": "this makes me angry", "label": "anger"},
            {"text": "I feel sad and empty", "label": "sadness"},
        ]
        * 4
    )
    assert hasattr(data, "as_pairs")
    losses = model.finetune(data, epochs=2, learning_rate=1e-2)
    assert len(losses) >= 1
    label = model.predict_label("I am so happy today")
    assert label in {"anger", "joy", "sadness"}


def test_hub_model_finetune_causal_lm_api():
    from magicmindnet.hub import HubModel

    model = HubModel.synthetic_causal_lm(vocab_size=128, max_seq_len=64)
    data = ai.DatasetCorpus(data=["hello world from hub", "finetune tiny language model"] * 8)
    assert hasattr(data, "as_texts")
    losses = model.finetune(data, epochs=1, learning_rate=1e-2)
    assert len(losses) == 1
    text = model.generate("hello", max_new_tokens=8)
    assert isinstance(text, str)


def test_dataset_as_pairs_and_as_texts():
    cls = ai.DatasetClassification(
        data=[{"text": "a", "label": "x"}, {"text": "b", "label": "y"}]
    )
    assert cls.as_pairs() == [("a", "x"), ("b", "y")]
    corp = ai.DatasetCorpus(data=["one", "two"])
    assert corp.as_texts() == ["one", "two"]


@pytest.mark.network
def test_from_pretrained_emotion_classifier_live():
    model = ai.from_pretrained("j-hartmann/emotion-english-distilroberta-base")
    assert model.family in {"classifier", "seq-cls", "text-classification"}
    assert model.predict_label("I love this!") == "joy"


@pytest.mark.network
def test_from_pretrained_qwen_gguf_live(tmp_path: Path):
    model = ai.from_pretrained(
        "Qwen/Qwen3-0.6B-GGUF",
        filename="Qwen3-0.6B-Q8_0.gguf",
        cache_dir=str(tmp_path / "hf"),
    )
    assert model.native is not None
    assert model.head_dim == 128
    text = model.generate("Hi", max_new_tokens=4)
    assert isinstance(text, str)
