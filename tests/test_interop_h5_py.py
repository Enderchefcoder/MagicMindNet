"""HDF5 / TensorFlow-Keras interop — from-scratch reader, no h5py required."""

import zipfile

import pytest

import magicmindnet as ai

FIXTURE = "tests/fixtures/simple.h5"


def test_fixture_reads_without_h5py():
    arrays = ai.load_h5(FIXTURE)
    assert arrays["weights"] == [
        [0.0, 1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0, 7.0],
        [8.0, 9.0, 10.0, 11.0],
    ]
    assert arrays["bias"] == [-1.0, 0.5]
    assert arrays["layer1/kernel"] == [[7.0, 7.0], [7.0, 7.0]]
    assert arrays["layer1/ints"] == [1.0, 2.0, 3.0]


def test_h5py_written_file_roundtrips(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "weights.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("dense/kernel", data=np.arange(6, dtype=np.float32).reshape(2, 3))
        f.create_dataset("dense/bias", data=np.array([0.5, -0.5, 1.0], dtype=np.float32))
        f.create_dataset("half", data=np.array([1.5, 2.5], dtype=np.float16))
        f.create_dataset("scalar_ish", data=np.array([42], dtype=np.uint8))
    arrays = ai.load_h5(path)
    assert arrays["dense/kernel"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert arrays["dense/bias"] == [0.5, -0.5, 1.0]
    assert arrays["half"] == [1.5, 2.5]
    assert arrays["scalar_ish"] == [42.0]


def test_keras_v3_archive_reads(tmp_path):
    path = tmp_path / "model.keras"
    with open(FIXTURE, "rb") as fh:
        h5_bytes = fh.read()
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("metadata.json", "{}")
        zf.writestr("config.json", "{}")
        zf.writestr("model.weights.h5", h5_bytes)
    arrays = ai.load_keras(str(path))
    assert "layer1/kernel" in arrays


def test_keras_plain_h5_passthrough():
    arrays = ai.load_keras(FIXTURE)
    assert "weights" in arrays


def test_compressed_dataset_rejected(tmp_path):
    h5py = pytest.importorskip("h5py")
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "gz.h5")
    with h5py.File(path, "w") as f:
        f.create_dataset("big", data=np.zeros((64, 64), dtype=np.float32), compression="gzip")
    with pytest.raises((RuntimeError, ValueError), match="chunked|compression"):
        ai.load_h5(path)


def test_not_hdf5_errors(tmp_path):
    path = tmp_path / "fake.h5"
    path.write_bytes(b"definitely not hdf5")
    with pytest.raises((RuntimeError, ValueError), match="HDF5"):
        ai.load_h5(str(path))


def test_keras_zip_without_weights_errors(tmp_path):
    path = tmp_path / "empty.keras"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("config.json", "{}")
    with pytest.raises((RuntimeError, ValueError), match="model.weights.h5"):
        ai.load_keras(str(path))
