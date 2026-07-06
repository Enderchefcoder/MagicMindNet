"""Wave-8 interop: IQ4 encoders, vision GGUF export, SavedModel dirs."""

import pytest

import magicmindnet as ai


@pytest.mark.parametrize("format", ["gguf-iq4_nl", "gguf-iq4_xs"])
def test_iq4_exports_roundtrip(tmp_path, format):
    data = ai.DatasetQA(data=[{"input": "hi", "output": "hello"}])
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=256, seed=53)
    before = bot.compute_mean_loss(data)
    path = str(tmp_path / "bot.gguf")
    bot.save(path, format=format)
    after = ai.load(path).compute_mean_loss(data)
    assert abs(before - after) < 1.0, f"{format}: {before} vs {after}"


@pytest.mark.parametrize("name", ["IQ4_NL", "IQ4_XS"])
def test_iq4_blocks_decode_identically_in_reference(tmp_path, name):
    """gguf-py can't encode IQ4, but it must decode our blocks like we do."""
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    import gguf.quants as gq

    from magicmindnet import _native

    format = {"IQ4_NL": "gguf-iq4_nl", "IQ4_XS": "gguf-iq4_xs"}[name]
    path = str(tmp_path / f"{name}.gguf")
    ai.Chatbot(vocab_size=64, n_layer=1, d_model=256, seed=59).save(path, format=format)
    reader = gguf.GGUFReader(path)
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    assert tensor.tensor_type == gguf.GGMLQuantizationType[name]
    payload = bytes(tensor.data.tobytes())
    numel = 64 * 256
    theirs = gq.dequantize(np.frombuffer(payload, np.uint8), tensor.tensor_type).reshape(-1)
    ours = np.array(_native.dequantize_ggml(name, list(payload), numel), dtype=np.float32)
    np.testing.assert_allclose(ours, theirs.astype(np.float32), rtol=2e-5, atol=1e-6)


def test_vision_gguf_universal_load(tmp_path):
    bot = ai.Chatbot(vision=True, vocab_size=96, n_layer=1, d_model=16, seed=61)
    path = str(tmp_path / "vision.gguf")
    bot.save(path, format="gguf")
    loaded = ai.load(path)
    assert loaded.has_vision is True
    reply = loaded.chat("hello")
    assert isinstance(reply, str)


def test_savedmodel_directory_loads(tmp_path):
    """A SavedModel-style directory resolves to its variables bundle."""
    import shutil

    variables = tmp_path / "saved" / "variables"
    variables.mkdir(parents=True)
    shutil.copy("tests/fixtures/tf/ckpt.index", variables / "variables.index")
    shutil.copy(
        "tests/fixtures/tf/ckpt.data-00000-of-00001",
        variables / "variables.data-00000-of-00001",
    )
    arrays = ai.load_tf_checkpoint(str(tmp_path / "saved"))
    assert arrays["b"] == [0.5, -0.5]


def test_live_savedmodel_loads(tmp_path):
    tf = pytest.importorskip("tensorflow")
    np = pytest.importorskip("numpy")

    class Holder(tf.Module):
        def __init__(self):
            super().__init__()
            self.w = tf.Variable(np.arange(6, dtype=np.float32).reshape(2, 3))

    module = Holder()
    path = str(tmp_path / "sm")
    tf.saved_model.save(module, path)
    arrays = ai.load_tf_checkpoint(path)
    kernel = next(v for k, v in arrays.items() if k.endswith("w") or "w" in k)
    assert kernel == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
