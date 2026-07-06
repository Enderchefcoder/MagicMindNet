"""Wave-4 interop: k-quant encoders, TF checkpoint v2, real-TF validation."""

import pytest

import magicmindnet as ai

TF_FIXTURES = "tests/fixtures/tf"


def make_bot(seed=7):
    # k-quants require rows (d_model) to be 256-multiples.
    return ai.Chatbot(vocab_size=64, n_layer=1, d_model=256, seed=seed)


@pytest.mark.parametrize("format", ["gguf-q4_k", "gguf-q5_k", "gguf-q6_k"])
def test_kquant_export_roundtrips(tmp_path, format):
    data = ai.DatasetQA(data=[{"input": "hi", "output": "hello"}])
    bot = make_bot(seed=29)
    before = bot.compute_mean_loss(data)
    path = str(tmp_path / "bot.gguf")
    bot.save(path, format=format)
    after = ai.load(path).compute_mean_loss(data)
    assert abs(before - after) < 1.0, f"{format}: {before} vs {after}"


def test_kquant_export_types_reported(tmp_path):
    path = str(tmp_path / "bot.gguf")
    make_bot().save(path, format="gguf-q6_k")
    info = ai.gguf_info(path)
    assert "Q6K" in {t["type"] for t in info["tensors"]}


def test_kquant_blocks_decode_identically_in_reference(tmp_path):
    """Our Q4_K/Q6_K encoder output must decode the same in gguf-py and here."""
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    import gguf.quants as gq

    from magicmindnet import _native

    for format, type_name in [("gguf-q4_k", "Q4_K"), ("gguf-q5_k", "Q5_K"), ("gguf-q6_k", "Q6_K")]:
        path = str(tmp_path / f"{type_name}.gguf")
        make_bot(seed=31).save(path, format=format)
        reader = gguf.GGUFReader(path)
        tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
        assert tensor.tensor_type == gguf.GGMLQuantizationType[type_name]
        payload = bytes(tensor.data.tobytes())
        numel = 64 * 256
        theirs = gq.dequantize(np.frombuffer(payload, dtype=np.uint8), tensor.tensor_type)
        ours = _native.dequantize_ggml(type_name, list(payload), numel)
        np.testing.assert_allclose(
            np.array(ours, dtype=np.float32),
            theirs.astype(np.float32).reshape(-1),
            rtol=2e-5,
            atol=1e-6,
        )


def test_kquant_reconstruction_quality(tmp_path):
    """Reference-decoded k-quant weights stay close to the original floats."""
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    import gguf.quants as gq

    bot = make_bot(seed=37)
    ref_path = str(tmp_path / "ref.mmn")
    bot.save(ref_path)
    import struct
    from pathlib import Path

    from conftest import load_checkpoint_tensors

    entry = load_checkpoint_tensors(Path(ref_path))["embed"]
    original = np.array(
        struct.unpack(f"<{len(entry['data']) // 4}f", bytes(entry["data"])),
        dtype=np.float32,
    )
    path = str(tmp_path / "q6.gguf")
    bot.save(path, format="gguf-q6_k")
    reader = gguf.GGUFReader(path)
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    decoded = gq.dequantize(tensor.data, tensor.tensor_type).reshape(-1).astype(np.float32)
    scale = np.abs(original).max()
    rmse = float(np.sqrt(np.mean((original - decoded) ** 2))) / max(scale, 1e-9)
    assert rmse < 0.02, f"Q6_K relative RMSE {rmse}"


def test_tf_checkpoint_fixture_reads_without_tensorflow():
    arrays = ai.load_tf_checkpoint(f"{TF_FIXTURES}/ckpt")
    assert arrays["w"] == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
    ]
    assert arrays["b"] == [0.5, -0.5]
    assert arrays["steps"] == [1.0, 2.0, 3.0]
    # `.index` suffix also accepted.
    again = ai.load_tf_checkpoint(f"{TF_FIXTURES}/ckpt.index")
    assert again["b"] == [0.5, -0.5]


def test_tf_checkpoint_missing_errors():
    with pytest.raises((RuntimeError, ValueError), match="cannot read"):
        ai.load_tf_checkpoint("/nonexistent/ckpt")


def test_real_keras_fixture_reads():
    weights = ai.load_keras(f"{TF_FIXTURES}/model.keras")
    assert "layers/dense/vars/0" in weights
    kernel = weights["layers/dense/vars/0"]
    assert len(kernel) == 3 and len(kernel[0]) == 4
    h5 = ai.load_h5(f"{TF_FIXTURES}/model.weights.h5")
    assert h5.keys() == weights.keys()


def test_live_tensorflow_checkpoint_matches_reader(tmp_path):
    tf = pytest.importorskip("tensorflow")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(3)
    w = rng.standard_normal((4, 5)).astype(np.float32)
    ckpt = tf.train.Checkpoint(w=tf.Variable(w))
    prefix = str(tmp_path / "live")
    ckpt.write(prefix)
    ours = ai.load_tf_checkpoint(prefix)
    np.testing.assert_allclose(np.array(ours["w"], dtype=np.float32), w, rtol=0, atol=0)


def test_live_keras_weights_match_reader(tmp_path):
    tf = pytest.importorskip("tensorflow")
    np = pytest.importorskip("numpy")
    keras = tf.keras
    model = keras.Sequential(
        [keras.Input(shape=(6,)), keras.layers.Dense(3, name="probe")]
    )
    path = str(tmp_path / "live.weights.h5")
    model.save_weights(path)
    ours = ai.load_h5(path)
    kernel_key = next(k for k in ours if k.endswith("vars/0"))
    np.testing.assert_allclose(
        np.array(ours[kernel_key], dtype=np.float32),
        model.get_weights()[0],
        rtol=1e-6,
        atol=1e-7,
    )
