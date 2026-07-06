"""NumPy .npy/.npz array IO — from-scratch codec, no numpy required."""

import pytest

import magicmindnet as ai


def test_npy_roundtrip_matrix(tmp_path):
    path = str(tmp_path / "m.npy")
    data = [[1.0, -2.5, 3.0], [4.0, 5.5, -6.0]]
    ai.save_npy(path, data)
    assert ai.load_npy(path) == data


def test_npy_roundtrip_vector_and_scalar(tmp_path):
    vec = str(tmp_path / "v.npy")
    ai.save_npy(vec, [1.0, 2.0, 3.0])
    assert ai.load_npy(vec) == [1.0, 2.0, 3.0]

    scalar = str(tmp_path / "s.npy")
    ai.save_npy(scalar, 42.0)
    assert ai.load_npy(scalar) == 42.0


def test_npy_accepts_tolist_objects(tmp_path):
    class FakeArray:
        """Anything exposing .tolist() (numpy arrays, torch tensors) works."""

        def tolist(self):
            return [[1.0, 2.0], [3.0, 4.0]]

    path = str(tmp_path / "fake.npy")
    ai.save_npy(path, FakeArray())
    assert ai.load_npy(path) == [[1.0, 2.0], [3.0, 4.0]]


def test_npy_ragged_input_rejected(tmp_path):
    with pytest.raises(ValueError, match="ragged"):
        ai.save_npy(str(tmp_path / "bad.npy"), [[1.0, 2.0], [3.0]])


def test_npy_missing_file_errors():
    with pytest.raises((RuntimeError, ValueError), match="cannot read"):
        ai.load_npy("/nonexistent/never.npy")


def test_npz_roundtrip_dict(tmp_path):
    path = str(tmp_path / "arrays.npz")
    arrays = {"weights": [[0.5, 1.5]], "bias": [0.25, -0.75]}
    ai.save_npz(path, arrays)
    back = ai.load_npz(path)
    assert back == arrays


def test_npz_numpy_compatibility(tmp_path):
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "np.npz")
    ai.save_npz(path, {"a": [[1.0, 2.0], [3.0, 4.0]]})
    loaded = np.load(path)
    assert loaded["a"].shape == (2, 2)
    assert loaded["a"][1][0] == 3.0

    out = str(tmp_path / "from_np.npz")
    np.savez(out, b=np.array([9.0, 8.0], dtype=np.float32))
    assert ai.load_npz(out) == {"b": [9.0, 8.0]}


def test_npy_numpy_compatibility(tmp_path):
    np = pytest.importorskip("numpy")
    path = str(tmp_path / "x.npy")
    ai.save_npy(path, [[7.0, 6.0]])
    arr = np.load(path)
    assert arr.dtype == np.float32
    assert arr.tolist() == [[7.0, 6.0]]

    out = str(tmp_path / "y.npy")
    np.save(out, np.arange(6, dtype=np.float64).reshape(2, 3))
    assert ai.load_npy(out) == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
