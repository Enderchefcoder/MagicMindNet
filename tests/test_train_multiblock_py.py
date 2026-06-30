"""Multi-block training and checkpoint continuity."""

import json
from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def _qa_dataset() -> ai.DatasetQA:
    return ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )


def test_train_two_layer_block1_ffn_changes_after_train(tmp_path: Path):
    ds = _qa_dataset()
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=16, seed=5)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    before = json.loads(before_path.read_text(encoding="utf-8"))["tensors"]["blocks.1.ffn"]["data"]
    cfg = ai.TrainConfig(epochs=8, batch_size=2, learning_rate=0.05, optimizer="adamw")
    ai.Train(bot, ds, cfg)
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    after = json.loads(after_path.read_text(encoding="utf-8"))["tensors"]["blocks.1.ffn"]["data"]
    assert before != after, "block 1 FFN should update during training"


def test_train_after_import_roundtrip_reduces_loss(tmp_path: Path):
    ds = _qa_dataset()
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=16, seed=3)
    path = tmp_path / "bot.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.n_layer == 2
    before = loaded.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=5, batch_size=1, learning_rate=0.05, optimizer="adamw")
    ai.Train(loaded, ds, cfg)
    after = loaded.compute_mean_loss(ds)
    assert after < before, f"post-import train loss before={before} after={after}"


def test_train_after_import_learned_pos_embed_reduces_loss(tmp_path: Path):
    from conftest import checkpoint_tensor_bytes

    ds = _qa_dataset()
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=4,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    path = tmp_path / "learned_qa.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.use_learned_pos_embed is True
    pe_before = checkpoint_tensor_bytes(path, "pos_embed")
    before = loaded.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=5, batch_size=1, learning_rate=0.05, optimizer="adamw")
    ai.Train(loaded, ds, cfg)
    after = loaded.compute_mean_loss(ds)
    assert after < before, f"post-import QA loss before={before} after={after}"
    out = tmp_path / "after_train.mmn"
    ai.export(loaded, "safetensors", str(out))
    pe_after = checkpoint_tensor_bytes(out, "pos_embed")
    assert pe_before != pe_after
