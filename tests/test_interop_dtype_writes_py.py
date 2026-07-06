"""dtype-parameterized writes: npy/npz element types + half-precision
safetensors — cross-validated against numpy.load and the official
safetensors package."""

import numpy as np
import pytest
from safetensors.numpy import load_file as st_load

import magicmindnet as ai


@pytest.mark.parametrize(
    "dtype,np_dtype",
    [
        ("f2", np.float16),
        ("f4", np.float32),
        ("f8", np.float64),
        ("i1", np.int8),
        ("i2", np.int16),
        ("i4", np.int32),
        ("i8", np.int64),
        ("u1", np.uint8),
        ("u2", np.uint16),
        ("u4", np.uint32),
        ("u8", np.uint64),
        ("float16", np.float16),
        ("float64", np.float64),
    ],
)
def test_npy_dtype_writes_load_with_numpy(tmp_path, dtype, np_dtype):
    path = str(tmp_path / f"a_{np.dtype(np_dtype).name}_{dtype}.npy")
    values = [1.0, 0.0, 3.0]
    ai.save_npy(path, values, dtype=dtype)
    arr = np.load(path)
    assert arr.dtype == np_dtype
    np.testing.assert_allclose(arr.astype(np.float32), values)
    # Our own reader agrees.
    assert ai.load_npy(path) == values


def test_npy_bool_dtype(tmp_path):
    path = str(tmp_path / "b.npy")
    ai.save_npy(path, [1.0, 0.0, 3.0], dtype="b1")
    arr = np.load(path)
    assert arr.dtype == np.bool_
    assert arr.tolist() == [True, False, True]


def test_npy_unknown_dtype_errors(tmp_path):
    with pytest.raises((ValueError, RuntimeError), match="not supported"):
        ai.save_npy(str(tmp_path / "x.npy"), [1.0], dtype="c16")


def test_npz_dtype_compressed(tmp_path):
    path = str(tmp_path / "w.npz")
    ai.save_npz(path, {"w": [[1.5, -2.5]], "b": [7.0]}, compress=True, dtype="f8")
    z = np.load(path)
    assert z["w"].dtype == np.float64
    np.testing.assert_allclose(z["w"], [[1.5, -2.5]])
    assert ai.load_npz(path)["b"] == [7.0]


@pytest.mark.parametrize("dtype,np_dtype", [("f32", np.float32), ("f16", np.float16)])
def test_safetensors_dtype_writes(tmp_path, dtype, np_dtype):
    path = str(tmp_path / f"m_{dtype}.safetensors")
    ai.save_safetensors(path, {"w": [[1.0, -2.0], [0.5, 3.0]]}, dtype=dtype)
    got = st_load(path)
    assert got["w"].dtype == np_dtype
    np.testing.assert_allclose(got["w"].astype(np.float32), [[1.0, -2.0], [0.5, 3.0]])
    assert ai.load_safetensors(path)["w"] == [[1.0, -2.0], [0.5, 3.0]]


def test_safetensors_bf16_roundtrips_through_our_reader(tmp_path):
    import os

    path = str(tmp_path / "m_bf16.safetensors")
    big = [[float(i + j) for i in range(64)] for j in range(64)]
    ai.save_safetensors(path, {"w": big}, dtype="bf16")
    assert ai.load_safetensors(path)["w"] == big
    # Tensor bytes halve vs f32 (header overhead is fixed).
    f32_path = str(tmp_path / "m_f32.safetensors")
    ai.save_safetensors(f32_path, {"w": big}, dtype="f32")
    assert os.path.getsize(path) < os.path.getsize(f32_path) * 0.6


def test_safetensors_unknown_dtype_errors(tmp_path):
    with pytest.raises((ValueError, RuntimeError), match="not supported"):
        ai.save_safetensors(str(tmp_path / "x.safetensors"), {"w": [1.0]}, dtype="i8")
