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


@pytest.mark.parametrize(
    "cname,shuffle_name",
    [("lz4", "SHUFFLE"), ("lz4", "NOSHUFFLE"), ("zlib", "SHUFFLE")],
)
def test_blosc_compressed_stores_read(tmp_path, cname, shuffle_name):
    from numcodecs import Blosc

    store = str(tmp_path / f"blosc_{cname}_{shuffle_name}.zarr")
    values = ((np.arange(30_000, dtype=np.float32) % 97) / 7.0).reshape(100, 300)
    z = zarr.create_array(
        store=store, shape=(100, 300), chunks=(40, 80), dtype="<f4",
        zarr_format=2,
        compressors=Blosc(cname=cname, clevel=5, shuffle=getattr(Blosc, shuffle_name)),
    )
    z[:] = values
    loaded = ai.load_zarr(store)
    np.testing.assert_allclose(np.array(loaded[""], dtype=np.float32), values)


def test_blosc_memcpy_small_chunks_read(tmp_path):
    from numcodecs import Blosc

    # Tiny random chunks are incompressible -> blosc memcpy mode.
    store = str(tmp_path / "memcpy.zarr")
    rng = np.random.default_rng(1)
    values = rng.standard_normal((4, 4)).astype(np.float32)
    z = zarr.create_array(
        store=store, shape=(4, 4), chunks=(2, 2), dtype="<f4",
        zarr_format=2, compressors=Blosc(cname="lz4", clevel=5, shuffle=Blosc.SHUFFLE),
    )
    z[:] = values
    loaded = ai.load_zarr(store)
    np.testing.assert_allclose(np.array(loaded[""], dtype=np.float32), values)


def test_zstd_compressed_store_rejected_clearly(tmp_path):
    # zarr-python 3.x defaults v2 stores to plain zstd — unsupported, with
    # a clear error naming the codec.
    store = str(tmp_path / "zstd.zarr")
    z = zarr.create_array(store=store, shape=(4,), chunks=(4,), dtype="<f4",
                          zarr_format=2)
    z[:] = np.ones(4, dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError), match="zstd"):
        ai.load_zarr(store)


@pytest.mark.parametrize("cname", ["blosclz", "lz4"])
@pytest.mark.parametrize("shuffle_name", ["SHUFFLE", "NOSHUFFLE", "BITSHUFFLE"])
def test_blosclz_and_bitshuffle_stores_read(tmp_path, cname, shuffle_name):
    from numcodecs import Blosc

    store = str(tmp_path / f"b_{cname}_{shuffle_name}.zarr")
    values = ((np.arange(30_000, dtype=np.float32) % 97) / 7.0)
    z = zarr.create_array(
        store=store, shape=values.shape, chunks=(8192,), dtype="<f4",
        zarr_format=2,
        compressors=Blosc(cname=cname, clevel=5, shuffle=getattr(Blosc, shuffle_name)),
    )
    z[:] = values
    loaded = ai.load_zarr(store)
    np.testing.assert_allclose(np.array(loaded[""], dtype=np.float32), values)


def test_zarr_v3_gzip_and_blosc_read(tmp_path):
    from zarr.codecs import BloscCodec, GzipCodec

    values = np.arange(24, dtype=np.float32).reshape(4, 6)
    g_store = str(tmp_path / "v3_gzip.zarr")
    z = zarr.create_array(store=g_store, shape=(4, 6), chunks=(2, 3),
                          dtype="float32", compressors=[GzipCodec(level=5)])
    z[:] = values
    np.testing.assert_allclose(np.array(ai.load_zarr(g_store)[""], dtype=np.float32), values)
    assert ai.detect_arrays_format(g_store) == "zarr"

    b_store = str(tmp_path / "v3_blosc.zarr")
    z2 = zarr.create_array(store=b_store, shape=(4, 6), chunks=(4, 6), dtype="float32",
                           compressors=[BloscCodec(cname="lz4", clevel=5, shuffle="shuffle")])
    z2[:] = values
    np.testing.assert_allclose(np.array(ai.load_zarr(b_store)[""], dtype=np.float32), values)


def test_zarr_v3_group_tree_reads(tmp_path):
    from zarr.codecs import GzipCodec

    store = str(tmp_path / "v3_group.zarr")
    root = zarr.open_group(store, mode="w")
    a = root.create_array("layer/kernel", shape=(2, 2), chunks=(2, 2),
                          dtype="float32", compressors=[GzipCodec()])
    a[:] = np.array([[1.5, -2.5], [0.25, 9.0]], dtype=np.float32)
    loaded = ai.load_zarr(store)
    np.testing.assert_allclose(
        np.array(loaded["layer/kernel"], dtype=np.float32), [[1.5, -2.5], [0.25, 9.0]]
    )


def test_zarr_v3_zstd_rejected_clearly(tmp_path):
    store = str(tmp_path / "v3_zstd.zarr")
    z = zarr.create_array(store=store, shape=(4,), chunks=(4,), dtype="float32")
    z[:] = np.ones(4, dtype=np.float32)
    with pytest.raises((ValueError, RuntimeError), match="zstd"):
        ai.load_zarr(store)


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


