"""Zarr v2 store IO (ai.load_zarr / ai.save_zarr) — cross-validated against
zarr-python (v2 format, zlib chunks)."""

import numpy as np
import pytest
import zarr
from numcodecs import Zlib

import magicmindnet as ai


def test_official_zarr_array_reads(tmp_path):
    store = str(tmp_path / "official.zarr")
    z = zarr.create_array(
        store=store, shape=(4, 6), chunks=(2, 3), dtype="<f4",
        zarr_format=2, compressors=Zlib(level=4),
    )
    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    z[:] = values
    loaded = ai.load_zarr(store)
    np.testing.assert_allclose(loaded[""], values)
    assert ai.detect_arrays_format(store) == "zarr"
    np.testing.assert_allclose(ai.load_arrays(store)[""], values)


@pytest.mark.parametrize("dtype", ["<f8", "<i4", "<u2", "<f2"])
def test_official_zarr_dtypes_read(tmp_path, dtype):
    store = str(tmp_path / f"dt_{dtype[1:]}.zarr")
    z = zarr.create_array(
        store=store, shape=(5,), chunks=(2,), dtype=dtype,
        zarr_format=2, compressors=Zlib(level=1),
    )
    z[:] = np.array([1, 0, 3, 2, 5]).astype(dtype)
    loaded = ai.load_zarr(store)
    np.testing.assert_allclose(loaded[""], [1.0, 0.0, 3.0, 2.0, 5.0])


def test_official_uncompressed_group_reads(tmp_path):
    store = str(tmp_path / "group.zarr")
    root = zarr.open_group(store, mode="w", zarr_format=2)
    a = root.create_array("layer/kernel", shape=(2, 2), chunks=(2, 2),
                          dtype="<f4", compressors=None)
    a[:] = np.array([[1.0, -2.0], [3.5, 0.25]], dtype=np.float32)
    loaded = ai.load_zarr(store)
    np.testing.assert_allclose(loaded["layer/kernel"], [[1.0, -2.0], [3.5, 0.25]])


def test_our_writer_opens_with_zarr_python(tmp_path):
    store = str(tmp_path / "ours.zarr")
    data = {"layer/kernel": [[1.0, -2.0], [3.5, 0.25]], "bias": [0.5, -0.5]}
    ai.save_zarr(store, data)
    g = zarr.open_group(store, mode="r", zarr_format=2)
    np.testing.assert_allclose(g["layer/kernel"][:], data["layer/kernel"])
    np.testing.assert_allclose(g["bias"][:], data["bias"])
    # And through our own reader + the universal loader.
    assert ai.load_zarr(store) == data
    assert ai.load_arrays(store) == data


def test_save_arrays_zarr_extension(tmp_path):
    store = str(tmp_path / "auto.zarr")
    ai.save_arrays(store, {"w": [[7.0]]})
    assert ai.detect_arrays_format(store) == "zarr"
    assert ai.load_arrays(store)["w"] == [[7.0]]


def test_missing_chunks_use_fill_value(tmp_path):
    import json
    import os

    store = tmp_path / "sparse.zarr"
    os.makedirs(store)
    (store / ".zarray").write_text(json.dumps({
        "chunks": [2], "compressor": None, "dtype": "<f4", "fill_value": 9.0,
        "filters": None, "order": "C", "shape": [4], "zarr_format": 2,
    }))
    (store / "0").write_bytes(np.array([1.0, 2.0], dtype=np.float32).tobytes())
    # Chunk "1" missing -> fill_value.
    loaded = ai.load_zarr(str(store))
    assert loaded[""] == [1.0, 2.0, 9.0, 9.0]


def test_blosc_store_rejected_with_hint(tmp_path):
    import json
    import os

    store = tmp_path / "blosc.zarr"
    os.makedirs(store)
    (store / ".zarray").write_text(json.dumps({
        "chunks": [1], "compressor": {"id": "blosc", "cname": "lz4"},
        "dtype": "<f4", "fill_value": 0.0, "filters": None,
        "order": "C", "shape": [1], "zarr_format": 2,
    }))
    with pytest.raises((ValueError, RuntimeError), match="blosc"):
        ai.load_zarr(str(store))
