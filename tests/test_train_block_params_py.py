"""Train() updates block params (FFN, attn, LN); RL keeps block weights frozen."""

from pathlib import Path

import magicmindnet as ai
from conftest import checkpoint_tensor_bytes

FIXTURES = Path(__file__).parent / "fixtures"


def _qa_dataset() -> ai.DatasetQA:
    return ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )


def test_train_changes_attn_q_weights(tmp_path: Path):
    ds = _qa_dataset()
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=12)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    q_before = checkpoint_tensor_bytes(before_path, "blocks.0.attn.q")
    cfg = ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    q_after = checkpoint_tensor_bytes(after_path, "blocks.0.attn.q")
    assert q_before != q_after


def test_train_changes_ln1_gamma(tmp_path: Path):
    ds = _qa_dataset()
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=13)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    gamma_before = checkpoint_tensor_bytes(before_path, "blocks.1.ln1.gamma")
    cfg = ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    gamma_after = checkpoint_tensor_bytes(after_path, "blocks.1.ln1.gamma")
    assert gamma_before != gamma_after


def test_train_changes_ffn2_as_positive_control(tmp_path: Path):
    ds = _qa_dataset()
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=14)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    ffn_before = checkpoint_tensor_bytes(before_path, "blocks.1.ffn2")
    cfg = ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    ffn_after = checkpoint_tensor_bytes(after_path, "blocks.1.ffn2")
    assert ffn_before != ffn_after


def test_train_changes_learned_pos_embed(tmp_path: Path):
    ds = _qa_dataset()
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=15,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    pe_before = checkpoint_tensor_bytes(before_path, "pos_embed")
    cfg = ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    pe_after = checkpoint_tensor_bytes(after_path, "pos_embed")
    assert pe_before != pe_after


def test_train_changes_embed_and_lm_head(tmp_path: Path):
    ds = _qa_dataset()
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=15)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    embed_before = checkpoint_tensor_bytes(before_path, "embed")
    head_before = checkpoint_tensor_bytes(before_path, "lm_head")
    cfg = ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    embed_after = checkpoint_tensor_bytes(after_path, "embed")
    head_after = checkpoint_tensor_bytes(after_path, "lm_head")
    assert embed_before != embed_after
    assert head_before != head_after
