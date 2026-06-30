import json
from pathlib import Path

import magicmindnet as ai


def test_quantize_int4_changes_lm_head_and_block_weights(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=1)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    before = json.loads(before_path.read_text(encoding="utf-8"))["tensors"]
    ai.quantize(bot, "int4")
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    after = json.loads(after_path.read_text(encoding="utf-8"))["tensors"]
    assert before["lm_head"]["data"] != after["lm_head"]["data"]
    assert before["blocks.0.ffn"]["data"] != after["blocks.0.ffn"]["data"]
    assert before["blocks.0.attn.q"]["data"] != after["blocks.0.attn.q"]["data"]
