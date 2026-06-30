"""Learned position embedding IO roundtrip."""

from pathlib import Path

import pytest

import magicmindnet as ai
from conftest import checkpoint_tensor_bytes, load_checkpoint_tensors, tensor_entry_first_f32


def test_learned_pos_embed_export_import_roundtrip(tmp_path: Path):
    bot = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=16,
        seed=11,
        use_learned_pos_embed=True,
        max_seq_len=32,
    )
    path = tmp_path / "learned_pos.mmn"
    ai.export(bot, "safetensors", str(path))
    before = checkpoint_tensor_bytes(path, "pos_embed")
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.use_learned_pos_embed is True
    assert loaded.max_seq_len == 32
    out = tmp_path / "reexport.mmn"
    ai.export(loaded, "safetensors", str(out))
    after = checkpoint_tensor_bytes(out, "pos_embed")
    assert before == after


def test_merge_learned_pos_embed_averages_weights(tmp_path: Path):
    a = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=16,
        seed=1,
        use_learned_pos_embed=True,
        max_seq_len=32,
    )
    b = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=16,
        seed=2,
        use_learned_pos_embed=True,
        max_seq_len=32,
    )
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
    wa = tensor_entry_first_f32(tensors_a["pos_embed"])
    wb = tensor_entry_first_f32(tensors_b["pos_embed"])
    wm = tensor_entry_first_f32(tensors_m["pos_embed"])
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_rejects_learned_vs_sinusoidal_pos_embed():
    learned = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=16,
        seed=1,
        use_learned_pos_embed=True,
        max_seq_len=32,
    )
    sinusoidal = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, seed=2)
    with pytest.raises(ai.ModelMismatchError):
        ai.merge(learned, sinusoidal)


def test_merge_trained_learned_pos_embed_averages_weights(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(Path(__file__).parent / "fixtures" / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    cfg = ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05)
    a = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=30,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    b = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=31,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    ai.Train(a, ds, cfg)
    ai.Train(b, ds, cfg)
    path_a = tmp_path / "a.mmn"
    path_b = tmp_path / "b.mmn"
    path_m = tmp_path / "m.mmn"
    ai.export(a, "safetensors", str(path_a))
    ai.export(b, "safetensors", str(path_b))
    merged = ai.merge(a, b)
    ai.export(merged, "safetensors", str(path_m))
    wa = tensor_entry_first_f32(load_checkpoint_tensors(path_a)["pos_embed"])
    wb = tensor_entry_first_f32(load_checkpoint_tensors(path_b)["pos_embed"])
    wm = tensor_entry_first_f32(load_checkpoint_tensors(path_m)["pos_embed"])
    assert abs(wm - (wa + wb) / 2.0) < 1e-5
