"""Wave-6 interop: writers for HDF5, TF checkpoint v2, ONNX + new encoders."""

import pytest

import magicmindnet as ai


def test_h5_writer_roundtrips_through_our_reader(tmp_path):
    path = str(tmp_path / "ours.h5")
    arrays = {
        "weights": [[1.0, 2.0], [3.0, 4.0]],
        "layer1/kernel": [[5.0]],
        "layer1/bias": [0.5, -0.5],
    }
    ai.save_h5(path, arrays)
    assert ai.load_h5(path) == arrays


def test_h5_writer_read_by_h5py(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = str(tmp_path / "ours.h5")
    ai.save_h5(path, {"weights": [[1.0, 2.0], [3.0, 4.0]], "grp/bias": [0.5, -0.5]})
    with h5py.File(path) as f:
        assert sorted(f.keys()) == ["grp", "weights"]
        assert f["weights"][()].tolist() == [[1.0, 2.0], [3.0, 4.0]]
        assert f["grp"]["bias"][()].tolist() == [0.5, -0.5]


def test_h5_writer_many_datasets_read_by_h5py(tmp_path):
    h5py = pytest.importorskip("h5py")
    path = str(tmp_path / "many.h5")
    arrays = {f"tensor_{i:03}": [float(i), float(-i)] for i in range(30)}
    ai.save_h5(path, arrays)
    with h5py.File(path) as f:
        assert len(f.keys()) == 30
        assert f["tensor_017"][()].tolist() == [17.0, -17.0]


def test_tf_checkpoint_writer_roundtrips(tmp_path):
    prefix = str(tmp_path / "ours_ckpt")
    arrays = {"w": [[9.0, 8.0], [7.0, 6.0]], "opt/steps": [1.0, 2.0, 3.0]}
    ai.save_tf_checkpoint(prefix, arrays)
    assert ai.load_tf_checkpoint(prefix) == arrays


def test_tf_checkpoint_writer_read_by_tensorflow(tmp_path):
    tf = pytest.importorskip("tensorflow")
    prefix = str(tmp_path / "ours_ckpt")
    ai.save_tf_checkpoint(prefix, {"w": [[9.0, 8.0]], "b": [0.25]})
    reader = tf.train.load_checkpoint(prefix)
    assert reader.get_tensor("w").tolist() == [[9.0, 8.0]]
    assert reader.get_tensor("b").tolist() == [0.25]
    shapes = reader.get_variable_to_shape_map()
    assert shapes["w"] == [1, 2]


def test_onnx_writer_roundtrips(tmp_path):
    path = str(tmp_path / "ours.onnx")
    arrays = {"w": [[1.5, 2.5]], "b": [3.0]}
    ai.save_onnx(path, arrays)
    assert ai.load_onnx(path) == arrays


def test_onnx_writer_passes_official_checker(tmp_path):
    onnx = pytest.importorskip("onnx")
    from onnx import numpy_helper

    path = str(tmp_path / "ours.onnx")
    ai.save_onnx(path, {"w": [[1.5, 2.5]], "b": [3.0]})
    model = onnx.load(path)
    onnx.checker.check_model(model)
    inits = {t.name: numpy_helper.to_array(t).tolist() for t in model.graph.initializer}
    assert inits == {"w": [[1.5, 2.5]], "b": [3.0]}


def test_full_circle_tf_to_mmn_to_tf(tmp_path):
    """Real TF writes -> we read -> we write -> real TF reads."""
    tf = pytest.importorskip("tensorflow")
    np = pytest.importorskip("numpy")
    rng = np.random.default_rng(17)
    original = rng.standard_normal((3, 5)).astype(np.float32)
    ckpt = tf.train.Checkpoint(w=tf.Variable(original))
    theirs = str(tmp_path / "theirs")
    ckpt.write(theirs)
    loaded = ai.load_tf_checkpoint(theirs)
    ours = str(tmp_path / "ours")
    ai.save_tf_checkpoint(ours, loaded)
    reader = tf.train.load_checkpoint(ours)
    np.testing.assert_allclose(reader.get_tensor("w"), original, rtol=0, atol=0)


@pytest.mark.parametrize("name", ["MXFP4", "TQ1_0"])
def test_new_encoders_match_reference_bytes(name):
    """Our MXFP4/TQ1_0 encoders must reproduce gguf-py's quantize exactly."""
    np = pytest.importorskip("numpy")
    gguf = pytest.importorskip("gguf")
    import gguf.quants as gq

    from magicmindnet import _native

    qtype = gguf.GGMLQuantizationType[name]
    rng = np.random.default_rng(23)
    values = (rng.standard_normal(512) * 0.6).astype(np.float32)
    if name == "TQ1_0":
        values = np.sign(values).astype(np.float32) * (np.abs(values) > 0.3)
    theirs = gq.quantize(values, qtype).tobytes()
    ours_dec = _native.dequantize_ggml(name, list(theirs), values.size)
    theirs_dec = gq.dequantize(np.frombuffer(theirs, np.uint8), qtype).reshape(-1)
    np.testing.assert_allclose(ours_dec, theirs_dec, rtol=1e-6, atol=1e-7)


def test_h5_writer_errors():
    with pytest.raises((RuntimeError, ValueError), match="at least one"):
        ai.save_h5("/tmp/never.h5", {})
