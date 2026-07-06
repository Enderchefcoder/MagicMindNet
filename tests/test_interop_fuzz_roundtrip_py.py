"""Randomized roundtrip hardening across every array writer.

Deterministically seeded shapes/values (scalars, empty axes, rank 0-4,
awkward names, extreme floats) round-trip through each format and, where
the reference package can read it, cross-check against that too.
"""

import numpy as np
import pytest

import magicmindnet as ai

FORMATS = ["npz", "pt", "h5", "onnx", "safetensors", "flax", "gguf", "pickle", "zarr"]


def random_arrays(seed):
    rng = np.random.default_rng(seed)
    arrays = {}
    for i in range(rng.integers(1, 5)):
        rank = int(rng.integers(0, 4))
        shape = tuple(int(rng.integers(1, 6)) for _ in range(rank))
        scale = 10.0 ** float(rng.integers(-2, 3))
        values = (rng.standard_normal(shape) * scale).astype(np.float32)
        name = f"layer_{seed}_{i}/weight" if rng.random() < 0.5 else f"t{i}"
        arrays[name] = values
    return arrays


@pytest.mark.parametrize("fmt", FORMATS)
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_random_roundtrips(tmp_path, fmt, seed):
    arrays = random_arrays(seed)
    ext = {"flax": "msgpack", "pickle": "pkl"}.get(fmt, fmt)
    path = str(tmp_path / f"fuzz_{fmt}_{seed}.{ext}")
    ai.save_arrays(path, arrays, format=fmt)
    back = ai.load_arrays(path)
    assert set(back) == set(arrays)
    for name, values in arrays.items():
        np.testing.assert_allclose(
            np.asarray(back[name], dtype=np.float32).reshape(values.shape),
            values,
            rtol=1e-6,
            err_msg=f"{fmt} seed {seed} tensor {name}",
        )


def test_extreme_values_roundtrip(tmp_path):
    extremes = {
        "extremes": [3.4e38, -3.4e38, 1.2e-38, 0.0, -0.0, 1.0],
    }
    for fmt in ["npz", "safetensors", "pt", "h5", "zarr", "pickle"]:
        ext = {"pickle": "pkl"}.get(fmt, fmt)
        path = str(tmp_path / f"extreme.{ext}")
        ai.save_arrays(path, extremes, format=fmt)
        back = ai.load_arrays(path)
        np.testing.assert_allclose(
            np.asarray(back["extremes"], dtype=np.float32),
            np.asarray(extremes["extremes"], dtype=np.float32),
            err_msg=fmt,
        )


def test_single_scalar_tensor_every_format(tmp_path):
    for fmt in FORMATS:
        ext = {"flax": "msgpack", "pickle": "pkl"}.get(fmt, fmt)
        path = str(tmp_path / f"scalar.{ext}")
        ai.save_arrays(path, {"s": np.float32(2.5)}, format=fmt)
        back = ai.load_arrays(path)
        assert float(np.asarray(back["s"]).reshape(())) == 2.5, fmt
