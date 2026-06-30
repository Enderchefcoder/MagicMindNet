from pathlib import Path

import magicmindnet as ai
from conftest import load_checkpoint_tensors, tensor_entry_first_f32


def _merge_export_first_f32(tmp_path: Path, tensor_key: str) -> tuple[float, float, float]:
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
    return wa, wb, wm


def test_merge_chatbot_averages_block_attn_q_weight(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.attn.q")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_block_attn_k_weight(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.attn.k")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_block_attn_v_weight(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.attn.v")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_block_attn_out_weight(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.attn.out")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_block_ffn_weight(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.ffn")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_block_ffn2_weight(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.ffn2")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_block_ln1_gamma(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.ln1.gamma")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_block_ln1_beta(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.ln1.beta")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_block_ln2_gamma(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.ln2.gamma")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_block_ln2_beta(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "blocks.0.ln2.beta")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_lm_head_weight(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "lm_head")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5


def test_merge_chatbot_averages_embed_weight(tmp_path: Path):
    wa, wb, wm = _merge_export_first_f32(tmp_path, "embed")
    assert abs(wm - (wa + wb) / 2.0) < 1e-5
