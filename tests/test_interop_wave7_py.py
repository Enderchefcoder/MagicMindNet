"""Wave-7 interop: from-scratch safetensors codec + complete k-quant encoders."""

import pytest

import magicmindnet as ai


def make_bot(seed=43):
    return ai.Chatbot(vocab_size=64, n_layer=1, d_model=256, seed=seed)


def test_hf_safetensors_read_by_official_package(tmp_path):
    """The official safetensors package must open files our codec writes."""
    st = pytest.importorskip("safetensors")
    from safetensors import safe_open

    path = str(tmp_path / "bot.safetensors")
    bot = ai.Chatbot(vocab_size=48, n_layer=1, d_model=16, seed=3)
    bot.save(path, format="hf-safetensors")
    with safe_open(path, framework="numpy") as f:
        names = set(f.keys())
        assert "embed" in names
        assert "blocks.0.attn.q" in names
        embed = f.get_tensor("embed")
        assert embed.shape == (48, 16)
        meta = f.metadata()
        assert meta["format"] == "mmn-hf-safetensors-v1"
    assert st is not None


def test_official_package_files_read_by_us(tmp_path):
    """Files written by safetensors.numpy must import as chatbots."""
    pytest.importorskip("safetensors")
    np = pytest.importorskip("numpy")
    from safetensors.numpy import save_file

    d, vocab = 8, 16
    tensors = {
        "model.embed_tokens.weight": np.full((vocab, d), 0.5, dtype=np.float32),
        "model.layers.0.self_attn.q_proj.weight": np.full((d, d), 0.1, dtype=np.float32),
        "model.layers.0.self_attn.k_proj.weight": np.full((d, d), 0.1, dtype=np.float32),
        "model.layers.0.self_attn.v_proj.weight": np.full((d, d), 0.1, dtype=np.float32),
        "model.layers.0.self_attn.o_proj.weight": np.full((d, d), 0.1, dtype=np.float32),
        "model.layers.0.mlp.up_proj.weight": np.full((d * 4, d), 0.2, dtype=np.float32),
        "model.layers.0.mlp.down_proj.weight": np.full((d, d * 4), 0.3, dtype=np.float32),
    }
    path = str(tmp_path / "external.safetensors")
    save_file(tensors, path)
    loaded = ai.load(path)
    assert isinstance(loaded, ai.Chatbot)
    assert loaded.vocab_size == vocab
    assert loaded.d_model == d


def test_hf_safetensors_roundtrip_after_codec_swap(tmp_path):
    """Loss parity through the from-scratch container."""
    data = ai.DatasetQA(data=[{"input": "hi", "output": "hello"}])
    bot = ai.Chatbot(vocab_size=64, n_layer=2, d_model=16, seed=7)
    before = bot.compute_mean_loss(data)
    path = str(tmp_path / "bot.safetensors")
    bot.save(path, format="hf-safetensors")
    after = ai.load(path).compute_mean_loss(data)
    assert abs(before - after) < 1e-4


@pytest.mark.parametrize("format", ["gguf-q2_k", "gguf-q3_k"])
def test_remaining_kquant_exports_roundtrip(tmp_path, format):
    data = ai.DatasetQA(data=[{"input": "hi", "output": "hello"}])
    bot = make_bot()
    before = bot.compute_mean_loss(data)
    path = str(tmp_path / "bot.gguf")
    bot.save(path, format=format)
    after = ai.load(path).compute_mean_loss(data)
    # 2-3 bit quantization is coarse; the model must stay usable.
    assert abs(before - after) < 2.0, f"{format}: {before} vs {after}"


@pytest.mark.parametrize("name", ["Q2_K", "Q3_K", "Q8_K"])
def test_remaining_kquant_blocks_decode_identically_in_reference(tmp_path, name):
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    import gguf.quants as gq

    from magicmindnet import _native

    format = {"Q2_K": "gguf-q2_k", "Q3_K": "gguf-q3_k"}.get(name)
    if format is None:
        pytest.skip("Q8_K has no chatbot export shorthand")
    path = str(tmp_path / f"{name}.gguf")
    make_bot(seed=47).save(path, format=format)
    reader = gguf.GGUFReader(path)
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType[name]
    payload = bytes(tensor.data.tobytes())
    numel = 64 * 256
    theirs = gq.dequantize(np.frombuffer(payload, np.uint8), tensor.tensor_type).reshape(-1)
    ours = np.array(_native.dequantize_ggml(name, list(payload), numel), dtype=np.float32)
    np.testing.assert_allclose(ours, theirs.astype(np.float32), rtol=2e-5, atol=1e-6)
