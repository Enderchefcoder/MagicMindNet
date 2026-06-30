"""Parametric coverage of the chatbot safetensors IO contract (see docs/checkpoint_coverage.md)."""

import json
from pathlib import Path

import pytest

import magicmindnet as ai
from conftest import load_checkpoint_tensors, tensor_entry_first_f32

# All exported chatbot tensor keys for a single-block model (d_model=16, vocab=256).
CHATBOT_TENSOR_KEYS = [
    "embed",
    "lm_head",
    "blocks.0.attn.q",
    "blocks.0.attn.k",
    "blocks.0.attn.v",
    "blocks.0.attn.out",
    "blocks.0.ffn",
    "blocks.0.ffn2",
    "blocks.0.ln1.gamma",
    "blocks.0.ln1.beta",
    "blocks.0.ln2.gamma",
    "blocks.0.ln2.beta",
]

# Learned position embedding (only exported when use_learned_pos_embed=True).
LEARNED_POS_EMBED_TENSOR_KEYS = ["pos_embed"]
LEARNED_POS_EMBED_MAX_SEQ_LEN = 32

# Layernorm γ/β are quantized in code but stay byte-identical at init (γ=1, β=0).
QUANTIZE_EXPORT_CHANGE_KEYS = [
    k
    for k in CHATBOT_TENSOR_KEYS
    if not k.endswith((".ln1.gamma", ".ln1.beta", ".ln2.gamma", ".ln2.beta"))
]

# Wrong shapes with the same element count as exported data (reaches expect_tensor_shape).
WRONG_SHAPES = {
    "embed": [128, 32],
    "lm_head": [128, 32],
    "pos_embed": [64, 8],
    "blocks.0.attn.q": [8, 32],
    "blocks.0.attn.k": [8, 32],
    "blocks.0.attn.v": [8, 32],
    "blocks.0.attn.out": [8, 32],
    "blocks.0.ffn": [128, 8],
    "blocks.0.ffn2": [32, 32],
    "blocks.0.ln1.gamma": [4, 4],
    "blocks.0.ln1.beta": [4, 4],
    "blocks.0.ln2.gamma": [4, 4],
    "blocks.0.ln2.beta": [4, 4],
}


def _export_chatbot(tmp_path: Path) -> Path:
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    return path


def _export_learned_pos_embed_chatbot(tmp_path: Path) -> Path:
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=16,
        seed=2,
        use_learned_pos_embed=True,
        max_seq_len=LEARNED_POS_EMBED_MAX_SEQ_LEN,
    )
    path = tmp_path / "learned_pos.mmn"
    ai.export(bot, "safetensors", str(path))
    return path


def _learned_pos_embed_bot(**kwargs) -> ai.Chatbot:
    defaults = {
        "vocab_size": 128,
        "n_layer": 1,
        "d_model": 16,
        "use_learned_pos_embed": True,
        "max_seq_len": LEARNED_POS_EMBED_MAX_SEQ_LEN,
    }
    defaults.update(kwargs)
    return ai.Chatbot(**defaults)


@pytest.mark.parametrize("tensor_key", CHATBOT_TENSOR_KEYS)
def test_import_rejects_missing_tensor_matrix_py(tmp_path: Path, tensor_key: str):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"].pop(tensor_key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert tensor_key.lower() in msg


@pytest.mark.parametrize("tensor_key", CHATBOT_TENSOR_KEYS)
def test_import_rejects_shape_mismatch_matrix_py(tmp_path: Path, tensor_key: str):
    path = _export_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"][tensor_key]["shape"] = WRONG_SHAPES[tensor_key]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert tensor_key.lower() in msg and "shape" in msg


@pytest.mark.parametrize("tensor_key", CHATBOT_TENSOR_KEYS)
def test_merge_averages_tensor_matrix_py(tmp_path: Path, tensor_key: str):
    a = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=1)
    b = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=2)
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    path_m = tmp_path / "m.mmn"
    ai.export(a, "safetensors", str(path_a))
    ai.export(b, "safetensors", str(path_b))
    merged = ai.merge(a, b)
    ai.export(merged, "safetensors", str(path_m))
    tensors_a = load_checkpoint_tensors(path_a)
    tensors_b = load_checkpoint_tensors(path_b)
    tensors_m = load_checkpoint_tensors(path_m)
    wa = tensor_entry_first_f32(tensors_a[tensor_key])
    wb = tensor_entry_first_f32(tensors_b[tensor_key])
    wm = tensor_entry_first_f32(tensors_m[tensor_key])
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


@pytest.mark.parametrize("mode", ["int8", "int4"])
@pytest.mark.parametrize("tensor_key", QUANTIZE_EXPORT_CHANGE_KEYS)
def test_quantize_changes_tensor_matrix_py(tmp_path: Path, mode: str, tensor_key: str):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before_path = tmp_path / f"before_{mode}_{tensor_key.replace('.', '_')}.mmn"
    ai.export(bot, "safetensors", str(before_path))
    before = load_checkpoint_tensors(before_path)
    ai.quantize(bot, mode)
    after_path = tmp_path / f"after_{mode}_{tensor_key.replace('.', '_')}.mmn"
    ai.export(bot, "safetensors", str(after_path))
    after = load_checkpoint_tensors(after_path)
    assert before[tensor_key]["data"] != after[tensor_key]["data"]


@pytest.mark.parametrize("tensor_key", LEARNED_POS_EMBED_TENSOR_KEYS)
def test_import_rejects_missing_learned_pos_embed_matrix_py(tmp_path: Path, tensor_key: str):
    path = _export_learned_pos_embed_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert payload["meta"].get("use_learned_pos_embed") is True
    payload["tensors"].pop(tensor_key, None)
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert tensor_key.lower() in msg


@pytest.mark.parametrize("tensor_key", LEARNED_POS_EMBED_TENSOR_KEYS)
def test_import_rejects_shape_mismatch_learned_pos_embed_matrix_py(
    tmp_path: Path, tensor_key: str
):
    path = _export_learned_pos_embed_chatbot(tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload["tensors"][tensor_key]["shape"] = WRONG_SHAPES[tensor_key]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises((ValueError, RuntimeError)) as exc:
        ai.import_model("safetensors", [str(path)])
    msg = str(exc.value).lower()
    assert tensor_key.lower() in msg and "shape" in msg


@pytest.mark.parametrize("tensor_key", LEARNED_POS_EMBED_TENSOR_KEYS)
def test_merge_averages_learned_pos_embed_matrix_py(tmp_path: Path, tensor_key: str):
    a = _learned_pos_embed_bot(seed=1)
    b = _learned_pos_embed_bot(seed=2)
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    path_m = tmp_path / "m.mmn"
    ai.export(a, "safetensors", str(path_a))
    ai.export(b, "safetensors", str(path_b))
    merged = ai.merge(a, b)
    ai.export(merged, "safetensors", str(path_m))
    tensors_a = load_checkpoint_tensors(path_a)
    tensors_b = load_checkpoint_tensors(path_b)
    tensors_m = load_checkpoint_tensors(path_m)
    wa = tensor_entry_first_f32(tensors_a[tensor_key])
    wb = tensor_entry_first_f32(tensors_b[tensor_key])
    wm = tensor_entry_first_f32(tensors_m[tensor_key])
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


@pytest.mark.parametrize("mode", ["int8", "int4"])
@pytest.mark.parametrize("tensor_key", LEARNED_POS_EMBED_TENSOR_KEYS)
def test_quantize_changes_learned_pos_embed_matrix_py(
    tmp_path: Path, mode: str, tensor_key: str
):
    bot = _learned_pos_embed_bot(vocab_size=256, seed=3)
    before_path = tmp_path / f"before_{mode}_{tensor_key}.mmn"
    ai.export(bot, "safetensors", str(before_path))
    before = load_checkpoint_tensors(before_path)
    ai.quantize(bot, mode)
    after_path = tmp_path / f"after_{mode}_{tensor_key}.mmn"
    ai.export(bot, "safetensors", str(after_path))
    after = load_checkpoint_tensors(after_path)
    assert before[tensor_key]["data"] != after[tensor_key]["data"]

