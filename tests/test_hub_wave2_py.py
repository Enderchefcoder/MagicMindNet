"""Hub wave-2 offline coverage: routing, capabilities, diffusion, rerank, Ollama mocks."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

import magicmindnet as ai
from magicmindnet.hub import (
    HubModel,
    ModelCard,
    _parse_spec,
    from_pretrained,
    inspect_source,
    list_hub_families,
)


def test_hub_public_exports():
    assert callable(ai.from_pretrained)
    assert callable(ai.inspect_source)
    assert callable(ai.resolve_source)
    assert "HubModel" in ai.__all__
    assert "ModelCard" in ai.__all__
    assert "inspect_source" in ai.__all__
    assert "resolve_source" in ai.__all__


@pytest.mark.parametrize(
    "spec,kind,rest",
    [
        ("hf://org/name", "hf", "org/name"),
        ("https://huggingface.co/org/name", "hf", "org/name"),
        ("ms://org/name", "modelscope", "org/name"),
        ("modelscope://org/name", "modelscope", "org/name"),
        ("https://www.modelscope.cn/models/org/name", "modelscope", "org/name"),
        ("ollama://llama3.2:1b", "ollama", "llama3.2:1b"),
        ("ollama:llama3.2", "ollama", "llama3.2"),
        ("org/name", "hf", "org/name"),
    ],
)
def test_parse_spec_prefixes(spec, kind, rest):
    assert _parse_spec(spec) == (kind, rest)


def test_parse_spec_local_suffix(tmp_path: Path):
    p = tmp_path / "x.mmn"
    p.write_text("{}", encoding="utf-8")
    kind, path = _parse_spec(str(p))
    assert kind == "local"
    assert path == str(p)


def test_inspect_asr_reranker_arrays_mlx_and_nlp_families():
    assert inspect_source({"pipeline_tag": "automatic-speech-recognition"}).family == "asr"
    assert (
        inspect_source(
            {
                "pipeline_tag": "text-classification",
                "tags": ["rerank"],
                "architectures": ["XLMRobertaForSequenceClassification"],
                "source": "BAAI/bge-reranker-v2-m3",
            }
        ).family
        == "reranker"
    )
    assert (
        inspect_source({"files": ["weights.safetensors"], "pipeline_tag": None}).family == "arrays"
    )
    mlx = inspect_source(
        {
            "pipeline_tag": "text-generation",
            "tags": ["mlx"],
            "architectures": ["Qwen3ForCausalLM"],
            "files": ["config.json"],
        }
    )
    assert any("MLX" in n for n in mlx.notes)
    assert inspect_source({"pipeline_tag": "feature-extraction"}).family == "embedding"
    assert inspect_source({"pipeline_tag": "sentence-similarity"}).family == "embedding"
    assert inspect_source({"pipeline_tag": "zero-shot-classification"}).family == "zero-shot"
    assert inspect_source({"pipeline_tag": "fill-mask"}).family == "fill-mask"
    assert inspect_source({"pipeline_tag": "question-answering"}).family == "question-answering"
    assert inspect_source({"pipeline_tag": "image-text-to-text"}).family == "vlm"


def test_model_card_from_dict_roundtrip():
    card = ModelCard(family="classifier", pipeline_tag="text-classification", notes=["n"])
    restored = ModelCard.from_dict(card.to_dict())
    assert restored.family == "classifier"
    assert restored.notes == ["n"]


def test_list_hub_families_contains_core():
    fams = list_hub_families()
    for name in (
        "causal-lm",
        "classifier",
        "reranker",
        "seq2seq",
        "diffusion",
        "video",
        "tts",
        "asr",
        "embedding",
        "gguf",
    ):
        assert name in fams


def test_capabilities_native_models():
    clf = HubModel.synthetic_classifier(["a", "b"])
    caps = clf.capabilities()
    assert caps["predict"] is True
    assert caps["finetune"] is True
    assert caps["generate"] is False
    lm = HubModel.synthetic_causal_lm()
    assert lm.capabilities()["generate"] is True
    diff = HubModel.synthetic_diffusion()
    assert diff.capabilities()["generate"] is True
    assert diff.capabilities()["finetune"] is True


def test_synthetic_diffusion_generate_and_finetune():
    fixtures = Path(__file__).parent / "fixtures"
    model = HubModel.synthetic_diffusion()
    out = model.generate("ignored", steps=1)
    assert out is not None
    ds = ai.DatasetImageGen(file=str(fixtures / "image_gen.json"))
    losses = model.finetune(ds, epochs=1, learning_rate=0.05, batch_size=1)
    assert len(losses) == 1


def test_synthetic_reranker_score_pairs_and_rerank():
    model = HubModel.synthetic_reranker()
    scores = model.score_pairs("query", ["good doc", "bad doc"])
    assert len(scores) == 2
    assert all(isinstance(s, float) for s in scores)
    ranked = model.rerank("query", ["a", "b", "c"], top_k=2)
    assert len(ranked) == 2
    assert all(isinstance(x, tuple) and len(x) == 2 for x in ranked)


def test_video_generate_raises_clear_not_implemented():
    card = ModelCard(family="video", backend="diffusers")
    model = HubModel(family="video", card=card, source="x")
    with pytest.raises(RuntimeError, match="video"):
        model.generate("a cat walking")


def test_gguf_shell_generate_raises_clear_error():
    card = ModelCard(family="gguf", backend="native", notes=["failed"])
    model = HubModel(family="gguf", card=card, source="broken.gguf")
    with pytest.raises(RuntimeError, match="GGUF|native|backend"):
        model.generate("hi")


def test_ollama_generate_and_chat_mocked():
    card = ModelCard(family="causal-lm", backend="ollama")
    model = HubModel(family="causal-lm", card=card, source="llama3.2:1b")

    class _Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"response": "hello-ollama", "message": {"content": "chat-hi"}}).encode()

    with patch("urllib.request.urlopen", return_value=_Resp()):
        assert model.generate("hi", max_new_tokens=4) == "hello-ollama"
        assert model.chat("hi", system="be brief") == "chat-hi"


def test_foreign_finetune_trainer_mock():
    card = ModelCard(family="classifier", backend="transformers")
    foreign = MagicMock()
    foreign.config = MagicMock()
    foreign.config.label2id = {"joy": 0, "anger": 1}
    foreign.config.id2label = {0: "joy", 1: "anger"}
    tok = MagicMock()

    def _tok(text, **kwargs):
        import torch

        return {
            "input_ids": torch.zeros(1, 4, dtype=torch.long),
            "attention_mask": torch.ones(1, 4, dtype=torch.long),
        }

    tok.side_effect = _tok
    model = HubModel(
        family="classifier",
        card=card,
        foreign=foreign,
        tokenizer=tok,
        labels=["joy", "anger"],
    )
    data = ai.DatasetClassification(
        data=[{"text": "happy", "label": "joy"}, {"text": "mad", "label": "anger"}] * 2
    )

    class _Result:
        training_loss = 0.5

    mock_trainer = MagicMock()
    mock_trainer.train.return_value = _Result()
    mock_args = MagicMock(seed=0, full_determinism=False)

    with (
        patch("transformers.trainer.Trainer", return_value=mock_trainer) as Trainer,
        patch("transformers.trainer.TrainingArguments", return_value=mock_args),
        patch("transformers.Trainer", return_value=mock_trainer),
        patch("transformers.TrainingArguments", return_value=mock_args),
    ):
        losses = model.finetune(
            data, epochs=1, batch_size=2, learning_rate=1e-4, verbose=True
        )
    assert losses == [0.5]
    assert Trainer.called or mock_trainer.train.called
    assert mock_trainer.train.called


def test_dataset_qa_as_pairs():
    qa = ai.DatasetQA(data=[{"input": "hi", "output": "hello"}, {"input": "a", "output": "b"}])
    assert qa.as_pairs() == [("hi", "hello"), ("a", "b")]


def test_cot_true_empty_thinktag_defaults_to_think_wrappers():
    qa = ai.DatasetQA(data=[{"input": "q", "output": "a"}], cot=True, thinktag="")
    formatted = qa.format_sample(0)
    assert "<think>" in formatted and "</think>" in formatted


def test_hub_save_native_roundtrip(tmp_path: Path):
    model = HubModel.synthetic_causal_lm(vocab_size=64)
    path = tmp_path / "hub_bot.mmn"
    model.save(str(path))
    loaded = from_pretrained(str(path))
    assert loaded.vocab_size == 64


def test_hub_chat_aliases_generate_native():
    model = HubModel.synthetic_causal_lm()
    text = model.chat("hi", max_new_tokens=4)
    assert isinstance(text, str)


def test_diffusion_foreign_generate_mock_pipe():
    card = ModelCard(family="diffusion", backend="diffusers")

    class _Pipe:
        def __call__(self, prompt, **kwargs):
            class _Out:
                images = [f"img:{prompt}"]

            return _Out()

    model = HubModel(family="diffusion", card=card, foreign=_Pipe())
    assert model.generate("sunset") == "img:sunset"


def test_resolve_local_dir_via_from_pretrained(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=32, n_layer=1, d_model=16, seed=2)
    path = tmp_path / "nested"
    path.mkdir()
    bot.save(str(path / "bot.mmn"))
    loaded = from_pretrained(str(path))
    assert loaded.native is not None


def test_download_false_local_ok(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=32, n_layer=1, d_model=16, seed=3)
    path = tmp_path / "x.mmn"
    bot.save(str(path))
    loaded = from_pretrained(str(path), download=False)
    assert loaded.vocab_size == 32


def test_hands_on_local_subprocess():
    """CI-friendly smoke: hub hands-on local case only."""
    import subprocess
    import sys

    root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, str(root / "scripts" / "hub_hands_on.py"), "--only", "local"],
        cwd=str(root),
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr + proc.stdout
    assert "local-native-chatbot" in proc.stdout or "OK" in proc.stdout
