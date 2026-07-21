"""final_norm + LoopLoRA QKV adapters (Glint-2 parity knobs)."""

from __future__ import annotations

import magicmindnet as ai
import pytest


def test_qwen_gguf_output_norm_weight_only_loads():
    """Live Qwen GGUF has output_norm.weight only — must not require beta."""
    from pathlib import Path

    gguf = Path(
        "/tmp/mmn_hub_cache/qwen_gguf/models--Qwen--Qwen3-0.6B-GGUF/"
        "snapshots/23749fefcc72300e3a2ad315e1317431b06b590a/Qwen3-0.6B-Q8_0.gguf"
    )
    if not gguf.is_file():
        pytest.skip("Qwen GGUF cache missing")
    bot = ai.load(str(gguf))
    assert bot.final_norm is True
    assert bot.head_dim == 128
    text = bot.generate("Hi", max_new_tokens=2)
    assert isinstance(text, str)


def test_final_norm_default_off():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=32, seed=1)
    assert bot.final_norm is False


def test_final_norm_rms_construct_train_roundtrip(tmp_path):
    bot = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=32,
        n_heads=4,
        seed=2,
        norm="rms",
        final_norm=True,
        tie_embeddings=True,
        use_rope=True,
    )
    assert bot.final_norm is True
    assert bot.norm == "rms"
    ds = ai.DatasetQA(
        data=[{"input": "hi", "output": "yo"}, {"input": "hey", "output": "hi"}]
    )
    before = bot.compute_mean_loss(ds)
    ai.Train(bot, ds, ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05, cuda=False))
    after = bot.compute_mean_loss(ds)
    assert after == after
    assert after <= before + 1.0  # should train; allow mild noise
    path = tmp_path / "fn.mmn"
    bot.save(str(path))
    loaded = ai.load(str(path))
    assert loaded.final_norm is True
    assert loaded.norm == "rms"
    assert abs(loaded.compute_mean_loss(ds) - after) < 1e-3


def test_loop_lora_default_rank_zero():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=32, seed=1)
    assert bot.lora_rank == 0


def test_loop_lora_zero_init_matches_no_lora_logits():
    kwargs = dict(
        vocab_size=64,
        n_layer=1,
        d_model=32,
        n_heads=4,
        n_loops=3,
        seed=7,
        norm="rms",
        ffn="swiglu",
        tie_embeddings=True,
        loop_embed=True,
        use_rope=True,
    )
    plain = ai.Chatbot(**kwargs)
    lora = ai.Chatbot(**kwargs, lora_rank=4)
    assert lora.lora_rank == 4
    assert lora.parameters > plain.parameters
    ids_prompt = "ab"
    a = plain.generate(ids_prompt, max_new_tokens=4)
    b = lora.generate(ids_prompt, max_new_tokens=4)
    # Zero-init up → identical generation at init
    assert a == b


def test_loop_lora_train_and_roundtrip(tmp_path):
    bot = ai.Chatbot(
        vocab_size=128,
        n_layer=1,
        d_model=32,
        n_heads=4,
        n_loops=3,
        lora_rank=4,
        seed=3,
        norm="rms",
        ffn="swiglu",
        tie_embeddings=True,
        loop_embed=True,
        final_norm=True,
        use_rope=True,
    )
    assert bot.lora_rank == 4
    assert bot.final_norm is True
    ds = ai.DatasetCorpus(data=["loop lora tiny train line one", "loop lora tiny train line two"])
    losses = bot.train(ds, epochs=1, learning_rate=0.02, verbose=False)
    assert len(losses) == 1
    assert losses[0] == losses[0]
    path = tmp_path / "lora.mmn"
    bot.save(str(path))
    loaded = ai.load(str(path))
    assert loaded.lora_rank == 4
    assert loaded.final_norm is True
    assert loaded.n_loops == 3
    assert loaded.loop_embed is True


def test_lora_rank_rejects_negative():
    with pytest.raises(ValueError):
        ai.Chatbot(vocab_size=32, n_layer=1, d_model=16, seed=1, lora_rank=-1)


def test_glint_full_stack_with_final_norm_and_lora(tmp_path):
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        n_loops=4,
        d_model=32,
        n_heads=4,
        ffn_dim=64,
        ffn="swiglu",
        norm="rms",
        use_rope=True,
        tie_embeddings=True,
        loop_embed=True,
        final_norm=True,
        lora_rank=2,
        seed=0,
        max_seq_len=64,
    )
    data = ai.DatasetCorpus(data=["hello world from glint tiny lm", "another short corpus line"])
    losses = bot.train(data, epochs=1, learning_rate=0.01, verbose=False)
    assert losses[0] == losses[0]
    text = bot.chat("hi", max_new_tokens=6)
    assert isinstance(text, str) and len(text) > 0
    path = tmp_path / "glint_full.safetensors"
    bot.save(str(path))
    loaded = ai.load(str(path))
    assert loaded.final_norm is True
    assert loaded.lora_rank == 2
