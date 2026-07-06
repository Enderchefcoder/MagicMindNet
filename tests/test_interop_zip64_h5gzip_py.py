"""ZIP64 archives + gzip-compressed HDF5 writing.

ZIP64 files are produced by CPython's zipfile with ``force_zip64``; the
compressed HDF5 output is verified by real h5py.
"""

import os
import zipfile

import h5py
import numpy as np

import magicmindnet as ai


def test_zip64_npz_reads(tmp_path):
    path = str(tmp_path / "z64.zip")
    values = np.arange(100, dtype=np.float32)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_DEFLATED) as zf:
        with zf.open("data.npy", "w", force_zip64=True) as f:
            np.save(f, values)
    assert ai.detect_arrays_format(path) == "npz"
    loaded = ai.load_arrays(path)
    assert loaded["data"] == values.tolist()


def test_zip64_torch_style_archive_reads(tmp_path):
    # A zip64 archive holding stored entries with the torch layout.
    path = str(tmp_path / "model.pt")
    inner = str(tmp_path / "plain.pt")
    ai.save_pt(inner, {"w": [[1.0, 2.0]]})
    with zipfile.ZipFile(inner) as src, zipfile.ZipFile(path, "w") as dst:
        for name in src.namelist():
            data = src.read(name)
            with dst.open(name, "w", force_zip64=True) as f:
                f.write(data)
    loaded = ai.load_pt(path)
    assert loaded["w"] == [[1.0, 2.0]]


def test_h5_gzip_write_reads_with_h5py(tmp_path):
    path = str(tmp_path / "compressed.h5")
    kernel = [[float((i + j) % 7) for i in range(32)] for j in range(32)]
    ai.save_h5(path, {"layer/kernel": kernel, "bias": [0.5, -0.5]}, compress=True)
    with h5py.File(path, "r") as f:
        assert f["layer/kernel"].compression == "gzip"
        np.testing.assert_allclose(
            f["layer/kernel"][()], np.array(kernel, dtype=np.float32)
        )
        np.testing.assert_allclose(f["bias"][()], [0.5, -0.5])


def test_h5_gzip_write_shrinks_and_roundtrips(tmp_path):
    data = {"big": [[float(i % 5) for i in range(128)] for _ in range(64)]}
    plain = str(tmp_path / "plain.h5")
    packed = str(tmp_path / "packed.h5")
    ai.save_h5(plain, data)
    ai.save_h5(packed, data, compress=True)
    assert os.path.getsize(packed) < os.path.getsize(plain) / 2
    assert ai.load_h5(packed) == ai.load_h5(plain) == data


def test_h5_gzip_full_circle_with_h5py(tmp_path):
    # ours (gzip) -> h5py read -> h5py write (gzip) -> ours read.
    ours = str(tmp_path / "ours.h5")
    theirs = str(tmp_path / "theirs.h5")
    data = {"w": [[1.5, -2.5], [0.25, 9.0]]}
    ai.save_h5(ours, data, compress=True)
    with h5py.File(ours, "r") as src, h5py.File(theirs, "w") as dst:
        dst.create_dataset("w", data=src["w"][()], compression="gzip")
    assert ai.load_h5(theirs)["w"] == data["w"]
