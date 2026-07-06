"""Generic safetensors array IO (ai.load_safetensors / ai.save_safetensors).

Cross-validated against the official ``safetensors`` package: files written
by our from-scratch codec must load with ``safetensors.numpy``, and files
written by the official package (every spec dtype) must load with ours.
"""

import numpy as np
import pytest
from safetensors.numpy import load_file as st_load_file
from safetensors.numpy import save_file as st_save_file

import magicmindnet as ai


def test_save_safetensors_loads_with_official_package(tmp_path):
    path = str(tmp_path / "ours.safetensors")
    ai.save_safetensors(
        path,
        {
            "w": [[1.0, -2.0], [3.5, 0.25]],
            "b": [0.5, -0.5, 9.0],
        },
    )
    official = st_load_file(path)
    np.testing.assert_allclose(official["w"], [[1.0, -2.0], [3.5, 0.25]])
    np.testing.assert_allclose(official["b"], [0.5, -0.5, 9.0])
    assert official["w"].dtype == np.float32


def test_load_safetensors_reads_official_file(tmp_path):
    path = str(tmp_path / "official.safetensors")
    st_save_file(
        {
            "layer.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
            "layer.bias": np.array([-1.5, 2.5], dtype=np.float32),
        },
        path,
    )
    arrays = ai.load_safetensors(path)
    assert arrays["layer.weight"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert arrays["layer.bias"] == [-1.5, 2.5]


@pytest.mark.parametrize(
    "dtype,values",
    [
        (np.bool_, [True, False, True]),
        (np.uint8, [0, 200, 255]),
        (np.int8, [-128, 0, 127]),
        (np.int16, [-300, 0, 400]),
        (np.uint16, [0, 60000]),
        (np.float16, [1.5, -2.25]),
        (np.int32, [-70000, 70000]),
        (np.uint32, [0, 4000000]),
        (np.float32, [3.5, -0.125]),
        (np.float64, [3.25, -8.5]),
        (np.int64, [-5, 9]),
        (np.uint64, [0, 12]),
    ],
)
def test_every_official_dtype_decodes(tmp_path, dtype, values):
    path = str(tmp_path / f"dt_{np.dtype(dtype).name}.safetensors")
    arr = np.array(values, dtype=dtype)
    st_save_file({"x": arr}, path)
    loaded = ai.load_safetensors(path)
    np.testing.assert_allclose(loaded["x"], arr.astype(np.float32))


def test_bf16_tensor_decodes(tmp_path):
    # Hand-built BF16 container (values exactly representable in bf16), so
    # the test needs neither torch nor safetensors.torch.
    import json
    import struct

    values = [1.0, -2.0, 0.5]
    f32 = np.array(values, dtype=np.float32)
    bf16 = (f32.view(np.uint32) >> 16).astype("<u2").tobytes()
    header = json.dumps(
        {"x": {"dtype": "BF16", "shape": [3], "data_offsets": [0, len(bf16)]}}
    ).encode()
    path = tmp_path / "bf16.safetensors"
    path.write_bytes(struct.pack("<Q", len(header)) + header + bf16)
    loaded = ai.load_safetensors(str(path))
    assert loaded["x"] == values


def test_roundtrip_through_our_codec(tmp_path):
    path = str(tmp_path / "roundtrip.safetensors")
    data = {"a": [[0.125, 1.0]], "z": [3.0]}
    ai.save_safetensors(path, data)
    assert ai.load_safetensors(path) == data


def test_save_rejects_ragged_arrays(tmp_path):
    path = str(tmp_path / "bad.safetensors")
    with pytest.raises(ValueError):
        ai.save_safetensors(path, {"x": [[1.0, 2.0], [3.0]]})


def test_load_rejects_non_safetensors(tmp_path):
    path = tmp_path / "junk.safetensors"
    path.write_bytes(b"definitely not safetensors")
    with pytest.raises((ValueError, RuntimeError)):
        ai.load_safetensors(str(path))
