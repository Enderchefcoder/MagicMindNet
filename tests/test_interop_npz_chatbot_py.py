"""NumPy .npz chatbot checkpoints — the numpy/TensorFlow interchange path."""

import zipfile

import pytest

import magicmindnet as ai


def make_bot(seed=17):
    return ai.Chatbot(vocab_size=48, n_layer=2, d_model=16, seed=seed)


def test_npz_chatbot_roundtrip(tmp_path):
    path = str(tmp_path / "bot.npz")
    bot = make_bot()
    bot.save(path, format="npz")
    loaded = ai.Chatbot.load(path)
    assert loaded.vocab_size == bot.vocab_size
    assert loaded.n_layer == bot.n_layer
    assert loaded.d_model == bot.d_model


def test_npz_universal_load(tmp_path):
    path = str(tmp_path / "bot.npz")
    make_bot().save(path, format="npz")
    assert isinstance(ai.load(path), ai.Chatbot)


def test_npz_roundtrip_preserves_loss(tmp_path):
    data = ai.DatasetQA(data=[{"input": "hi", "output": "hello"}])
    bot = make_bot(seed=23)
    before = bot.compute_mean_loss(data)
    path = str(tmp_path / "bot.npz")
    bot.save(path, format="numpy")
    after = ai.load(path).compute_mean_loss(data)
    assert abs(before - after) < 1e-4


def test_npz_checkpoint_is_valid_zip_of_npy(tmp_path):
    path = tmp_path / "bot.npz"
    make_bot().save(str(path), format="npz")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert "meta.json" in names
        assert "embed.npy" in names
        assert "blocks.0.attn.q.npy" in names
        assert zf.testzip() is None


def test_npz_numpy_can_read_checkpoint(tmp_path):
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "bot.npz")
    bot = make_bot()
    bot.save(path, format="npz")
    arrays = np.load(path)
    assert arrays["embed"].shape == (48, 16)
    assert arrays["lm_head"].dtype == np.float32


def test_npz_keras_style_weights_import(tmp_path):
    """np.savez of HF/MMN-named weights (e.g. exported from TF/Keras) loads."""
    np = pytest.importorskip("numpy")
    d, vocab = 8, 16
    path = str(tmp_path / "tf_weights.npz")
    np.savez(
        path,
        **{
            "model.embed_tokens.weight": np.full((vocab, d), 0.5, dtype=np.float32),
            "model.layers.0.self_attn.q_proj.weight": np.full((d, d), 0.1, dtype=np.float32),
            "model.layers.0.self_attn.k_proj.weight": np.full((d, d), 0.1, dtype=np.float32),
            "model.layers.0.self_attn.v_proj.weight": np.full((d, d), 0.1, dtype=np.float32),
            "model.layers.0.self_attn.o_proj.weight": np.full((d, d), 0.1, dtype=np.float32),
            "model.layers.0.mlp.up_proj.weight": np.full((d * 4, d), 0.2, dtype=np.float32),
            "model.layers.0.mlp.down_proj.weight": np.full((d, d * 4), 0.3, dtype=np.float32),
        },
    )
    loaded = ai.load(path)
    assert isinstance(loaded, ai.Chatbot)
    assert loaded.vocab_size == vocab
    assert loaded.d_model == d


def test_import_model_npz_format(tmp_path):
    path = str(tmp_path / "bot.npz")
    make_bot().save(path, format="npz")
    loaded = ai.import_model("npz", [path])
    assert loaded.vocab_size == 48
