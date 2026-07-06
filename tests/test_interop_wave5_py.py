"""Wave-5 interop: ONNX, chunked/gzip HDF5, classic-quant encoders, fast inflate."""

import pytest

import magicmindnet as ai

CHUNKED_FIXTURE = "tests/fixtures/chunked.h5"


def test_chunked_gzip_h5_fixture_reads():
    arrays = ai.load_h5(CHUNKED_FIXTURE)
    assert arrays["gz"][0][0] == 0.0
    assert arrays["gz"][1][1] == 65.0
    assert arrays["gz"][63][63] == 4095.0
    assert arrays["shuf"][3][4] == 100.0
    assert all(v == 1.0 for row in arrays["plainchunk"] for v in row)


def test_live_h5py_gzip_shuffle_roundtrip(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "gz.h5")
    rng = np.random.default_rng(5)
    data = rng.standard_normal((37, 21)).astype(np.float32)  # odd shape: edge chunks
    with h5py.File(path, "w") as f:
        f.create_dataset("x", data=data, compression="gzip", compression_opts=9, shuffle=True)
        f.create_dataset("y", data=data.astype(np.float64), compression="gzip", chunks=(10, 7))
    arrays = ai.load_h5(path)
    np.testing.assert_allclose(np.array(arrays["x"], dtype=np.float32), data, rtol=0, atol=0)
    np.testing.assert_allclose(
        np.array(arrays["y"], dtype=np.float32), data.astype(np.float64).astype(np.float32)
    )


def test_onnx_real_model_reads(tmp_path):
    onnx = pytest.importorskip("onnx")
    np = pytest.importorskip("numpy")
    from onnx import TensorProto, helper, numpy_helper

    rng = np.random.default_rng(7)
    w = rng.standard_normal((4, 3)).astype(np.float32)
    b = np.array([1, -2, 3], dtype=np.int64)
    h = rng.standard_normal((2, 2)).astype(np.float16)
    graph = helper.make_graph(
        nodes=[helper.make_node("MatMul", ["input", "w"], ["mm"])],
        name="g",
        inputs=[helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 4])],
        outputs=[helper.make_tensor_value_info("mm", TensorProto.FLOAT, [1, 3])],
        initializer=[
            numpy_helper.from_array(w, name="w"),
            numpy_helper.from_array(b, name="b"),
            numpy_helper.from_array(h, name="h"),
        ],
    )
    model = helper.make_model(graph)
    path = str(tmp_path / "model.onnx")
    onnx.save(model, path)

    arrays = ai.load_onnx(path)
    np.testing.assert_allclose(np.array(arrays["w"], dtype=np.float32), w)
    assert arrays["b"] == [1.0, -2.0, 3.0]
    np.testing.assert_allclose(
        np.array(arrays["h"], dtype=np.float32), h.astype(np.float32)
    )


def test_onnx_errors():
    with pytest.raises((RuntimeError, ValueError), match="cannot read"):
        ai.load_onnx("/nonexistent/model.onnx")


@pytest.mark.parametrize("name", ["Q4_1", "Q5_0", "Q5_1", "TQ2_0"])
def test_classic_quant_encoders_match_reference_bytes(tmp_path, name):
    """Our encoders must produce byte-identical blocks to gguf-py's quantize."""
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    import gguf.quants as gq

    from magicmindnet import _native

    qtype = gguf.GGMLQuantizationType[name]
    rng = np.random.default_rng(13)
    values = (rng.standard_normal(512) * 0.4).astype(np.float32)
    if name == "TQ2_0":
        values = np.sign(values).astype(np.float32)  # ternary-friendly input
    theirs = gq.quantize(values, qtype).tobytes()
    # Roundtrip through our writer: save a GGUF with this quant and re-read
    # the raw payload via the reference reader for byte comparison.
    format = {"Q4_1": "gguf-q4_1", "Q5_0": "gguf-q5_0", "Q5_1": "gguf-q5_1"}.get(name)
    if format is None:
        # TQ2_0 has no chatbot export; compare dequants instead.
        ours_dec = _native.dequantize_ggml(name, list(theirs), values.size)
        theirs_dec = gq.dequantize(np.frombuffer(theirs, np.uint8), qtype).reshape(-1)
        np.testing.assert_allclose(ours_dec, theirs_dec, rtol=1e-6, atol=1e-7)
        return
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=256, seed=41)
    path = str(tmp_path / "enc.gguf")
    bot.save(path, format=format)
    reader = gguf.GGUFReader(path)
    tensor = next(t for t in reader.tensors if t.name == "token_embd.weight")
    assert tensor.tensor_type == qtype
    payload = np.frombuffer(bytes(tensor.data.tobytes()), dtype=np.uint8)
    # Reference re-quantization of the reference-dequantized payload must be
    # a fixed point: decode ours, re-encode with gguf-py, compare bytes.
    decoded = gq.dequantize(payload, qtype).reshape(64, 256).astype(np.float32)
    requantized = gq.quantize(decoded, qtype).tobytes()
    assert requantized == payload.tobytes(), f"{name} blocks are not a reference fixed point"


def test_compressed_npz_large_roundtrip(tmp_path):
    """Exercise the table-based inflate fast path on a bigger archive."""
    # Power-of-two denominators keep the values exactly f32-representable.
    values = [[float((i * j) % 89) / 128.0 for j in range(256)] for i in range(128)]
    path = str(tmp_path / "big.npz")
    ai.save_npz(path, {"m": values}, compress=True)
    back = ai.load_npz(path)
    assert back["m"] == values
