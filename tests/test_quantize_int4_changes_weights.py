import json
from pathlib import Path

import magicmindnet as ai


def test_quantize_int4_changes_embed_tensor_data(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, seed=1)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    before_embed = json.loads(before_path.read_text(encoding="utf-8"))["tensors"]["embed"]["data"]
    ai.quantize(bot, "int4")
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    after_embed = json.loads(after_path.read_text(encoding="utf-8"))["tensors"]["embed"]["data"]
    assert before_embed != after_embed
