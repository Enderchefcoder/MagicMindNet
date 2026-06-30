"""RL / SPIN training behavior regression tests."""

from pathlib import Path

import magicmindnet as ai
from conftest import checkpoint_tensor_bytes, checkpoint_tensor_first_f32

FIXTURES = Path(__file__).parent / "fixtures"


def test_rl_changes_lm_head_weights(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=7)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    w_before = checkpoint_tensor_first_f32(before_path, "lm_head")
    cfg = ai.TrainConfig(epochs=1, batch_size=1, learning_rate=0.05)
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy")
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    w_after = checkpoint_tensor_first_f32(after_path, "lm_head")
    assert w_before != w_after


def test_rl_reward_only_changes_lm_head(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=9)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    w_before = checkpoint_tensor_first_f32(before_path, "lm_head")
    cfg = ai.TrainConfig(epochs=1, batch_size=1, learning_rate=0.05)
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="reward_only")
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    w_after = checkpoint_tensor_first_f32(after_path, "lm_head")
    assert w_before != w_after


def test_rl_punish_only_changes_lm_head(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=10)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    w_before = checkpoint_tensor_first_f32(before_path, "lm_head")
    cfg = ai.TrainConfig(epochs=1, batch_size=1, learning_rate=0.05)
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="punish_only")
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    w_after = checkpoint_tensor_first_f32(after_path, "lm_head")
    assert w_before != w_after


def test_rl_does_not_change_attn_q_weights(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=15)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    q_before = checkpoint_tensor_bytes(before_path, "blocks.0.attn.q")
    cfg = ai.TrainConfig(epochs=1, batch_size=1, learning_rate=0.05)
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy")
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    q_after = checkpoint_tensor_bytes(after_path, "blocks.0.attn.q")
    assert q_before == q_after


def test_rl_does_not_change_ln1_gamma(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=16)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    gamma_before = checkpoint_tensor_bytes(before_path, "blocks.1.ln1.gamma")
    cfg = ai.TrainConfig(epochs=1, batch_size=1, learning_rate=0.05)
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy")
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    gamma_after = checkpoint_tensor_bytes(after_path, "blocks.1.ln1.gamma")
    assert gamma_before == gamma_after


def test_rl_does_not_change_learned_pos_embed(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=19,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    pe_before = checkpoint_tensor_bytes(before_path, "pos_embed")
    cfg = ai.TrainConfig(epochs=1, batch_size=1, learning_rate=0.05)
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy")
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    pe_after = checkpoint_tensor_bytes(after_path, "pos_embed")
    assert pe_before == pe_after


def test_spin_changes_learned_pos_embed(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=20,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    pe_before = checkpoint_tensor_bytes(before_path, "pos_embed")
    ai.SPIN(bot, 2, ds)
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    pe_after = checkpoint_tensor_bytes(after_path, "pos_embed")
    assert pe_before != pe_after


def test_spin_changes_attn_q_weights(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=17)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    q_before = checkpoint_tensor_bytes(before_path, "blocks.0.attn.q")
    ai.SPIN(bot, 2, ds)
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    q_after = checkpoint_tensor_bytes(after_path, "blocks.0.attn.q")
    assert q_before != q_after


def test_spin_changes_ln1_gamma(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=18)
    before_path = tmp_path / "before.mmn"
    ai.export(bot, "safetensors", str(before_path))
    gamma_before = checkpoint_tensor_bytes(before_path, "blocks.1.ln1.gamma")
    ai.SPIN(bot, 2, ds)
    after_path = tmp_path / "after.mmn"
    ai.export(bot, "safetensors", str(after_path))
    gamma_after = checkpoint_tensor_bytes(after_path, "blocks.1.ln1.gamma")
    assert gamma_before != gamma_after


def test_spin_completes_with_finite_mean_loss():
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=11)
    ai.SPIN(bot, 2, ds)
    loss = bot.compute_mean_loss(ds)
    assert loss > 0.0
    assert loss == loss  # finite (not NaN)
