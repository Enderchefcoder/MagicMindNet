"""Flax/JAX msgpack checkpoint IO (ai.load_flax / ai.save_flax).

Cross-validated against the official ``msgpack`` package using the exact
``flax.serialization`` encoding: pytree map with ndarray leaves as
``ExtType(1, packb((shape, dtype.name, tobytes())))``.
"""

import msgpack
import numpy as np
import pytest

import magicmindnet as ai


def flax_ndarray(arr):
    """Replicate flax.serialization._ndarray_to_bytes exactly."""
    payload = msgpack.packb(
        (arr.shape, arr.dtype.name, arr.tobytes()), use_bin_type=True
    )
    return msgpack.ExtType(1, payload)


def flax_to_bytes(tree):
    """Replicate flax.serialization.to_bytes on a plain-dict pytree."""

    def enc(x):
        if isinstance(x, np.ndarray):
            return flax_ndarray(x)
        return x

    return msgpack.packb(tree, default=enc, use_bin_type=True)


def flax_from_bytes(data):
    """Replicate flax.serialization.from_bytes (ndarray leaves only)."""

    def dec(code, payload):
        assert code == 1
        shape, dtype_name, raw = msgpack.unpackb(payload, raw=True)
        return np.frombuffer(raw, dtype=np.dtype(dtype_name.decode())).reshape(shape)

    return msgpack.unpackb(data, ext_hook=dec, raw=False)


def test_load_flax_reads_flax_style_checkpoint(tmp_path):
    tree = {
        "params": {
            "dense": {
                "kernel": np.arange(6, dtype=np.float32).reshape(2, 3),
                "bias": np.array([0.5, -0.5, 1.5], dtype=np.float32),
            }
        },
        "step": 7,
    }
    path = tmp_path / "state.msgpack"
    path.write_bytes(flax_to_bytes(tree))
    arrays = ai.load_flax(str(path))
    assert arrays["params/dense/kernel"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert arrays["params/dense/bias"] == [0.5, -0.5, 1.5]
    assert arrays["step"] == 7.0


def test_save_flax_loads_with_official_msgpack(tmp_path):
    path = str(tmp_path / "out.msgpack")
    ai.save_flax(
        path,
        {
            "params/dense/kernel": [[1.0, -2.0], [3.5, 0.25]],
            "params/dense/bias": [9.0],
        },
    )
    tree = flax_from_bytes(open(path, "rb").read())
    np.testing.assert_allclose(
        tree["params"]["dense"]["kernel"], [[1.0, -2.0], [3.5, 0.25]]
    )
    np.testing.assert_allclose(tree["params"]["dense"]["bias"], [9.0])
    assert tree["params"]["dense"]["kernel"].dtype == np.float32


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.bool_, [True, False]),
        (np.uint8, [0, 200]),
        (np.int8, [-128, 127]),
        (np.float16, [1.5, -2.25]),
        (np.int16, [-300, 400]),
        (np.uint16, [60000]),
        (np.int32, [-70000]),
        (np.uint32, [70000]),
        (np.float32, [0.25, -8.0]),
        (np.float64, [3.25]),
        (np.int64, [-5]),
        (np.uint64, [12]),
    ],
)
def test_flax_dtypes_decode(tmp_path, dtype, values):
    arr = np.array(values, dtype=dtype)
    path = tmp_path / f"dt_{np.dtype(dtype).name}.msgpack"
    path.write_bytes(flax_to_bytes({"x": arr}))
    loaded = ai.load_flax(str(path))
    np.testing.assert_allclose(loaded["x"], arr.astype(np.float32))


def test_flax_roundtrip_through_our_codec(tmp_path):
    path = str(tmp_path / "roundtrip.msgpack")
    data = {"a/b": [[0.125]], "a/c": [1.0, 2.0], "z": [3.0]}
    ai.save_flax(path, data)
    assert ai.load_flax(path) == data


def test_flax_rejects_conflicting_paths(tmp_path):
    path = str(tmp_path / "bad.msgpack")
    with pytest.raises((ValueError, RuntimeError)):
        ai.save_flax(path, {"a": [1.0], "a/b": [2.0]})


def test_flax_rejects_non_msgpack(tmp_path):
    path = tmp_path / "junk.msgpack"
    path.write_bytes(b"\xc1 not msgpack")
    with pytest.raises((ValueError, RuntimeError)):
        ai.load_flax(str(path))
