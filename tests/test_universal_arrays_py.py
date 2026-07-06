"""Universal array IO: ai.load_arrays / ai.save_arrays / ai.load_gguf_arrays.

One loader detects every supported container by content; save_arrays infers
the writer from the extension. Cross-checks against official packages where
they are installed (numpy, gguf).
"""

import numpy as np
import pytest

import magicmindnet as ai

DATA = {"w": [[1.0, -2.0], [3.5, 0.25]], "b": [0.5, 0.0, -0.5]}


@pytest.mark.parametrize(
    "ext,expected_format",
    [
        ("npz", "npz"),
        ("pt", "pt"),
        ("pth", "pt"),
        ("h5", "h5"),
        ("hdf5", "h5"),
        ("onnx", "onnx"),
        ("safetensors", "safetensors"),
        ("msgpack", "flax"),
        ("gguf", "gguf"),
    ],
)
def test_save_arrays_roundtrips_every_extension(tmp_path, ext, expected_format):
    path = str(tmp_path / f"arrays.{ext}")
    ai.save_arrays(path, DATA)
    assert ai.detect_arrays_format(path) == expected_format
    back = ai.load_arrays(path)
    assert back["w"] == DATA["w"]
    assert back["b"] == DATA["b"]


def test_save_arrays_explicit_format_overrides_extension(tmp_path):
    path = str(tmp_path / "weights.bin")
    ai.save_arrays(path, DATA, format="safetensors")
    assert ai.detect_arrays_format(path) == "safetensors"
    assert ai.load_arrays(path)["w"] == DATA["w"]


def test_save_arrays_tf_checkpoint_format(tmp_path):
    prefix = str(tmp_path / "ckpt")
    ai.save_arrays(prefix, DATA, format="tf-checkpoint")
    back = ai.load_arrays(prefix)
    assert back["w"] == DATA["w"]


def test_save_arrays_unknown_extension_errors(tmp_path):
    with pytest.raises(ValueError, match="cannot infer a format"):
        ai.save_arrays(str(tmp_path / "weights.xyz"), DATA)


def test_load_arrays_single_npy(tmp_path):
    path = str(tmp_path / "single.npy")
    ai.save_npy(path, [[7.0, 8.0]])
    assert ai.detect_arrays_format(path) == "npy"
    assert ai.load_arrays(path) == {"arr": [[7.0, 8.0]]}


def test_load_arrays_official_numpy_npz(tmp_path):
    path = str(tmp_path / "np.npz")
    np.savez(path, w=np.array(DATA["w"], dtype=np.float32))
    loaded = ai.load_arrays(path)
    assert loaded["w"] == DATA["w"]


def test_load_arrays_rejects_junk(tmp_path):
    path = tmp_path / "junk.bin"
    path.write_bytes(b"absolutely not a tensor container")
    with pytest.raises((ValueError, RuntimeError), match="unrecognized tensor container"):
        ai.load_arrays(str(path))


def test_gguf_arrays_roundtrip_and_official_load(tmp_path):
    path = str(tmp_path / "arrays.gguf")
    ai.save_gguf_arrays(path, DATA)
    back = ai.load_gguf_arrays(path)
    assert back == DATA

    gguf = pytest.importorskip("gguf")
    reader = gguf.GGUFReader(path)
    names = {t.name for t in reader.tensors}
    assert names == {"w", "b"}
    w = next(t for t in reader.tensors if t.name == "w")
    np.testing.assert_allclose(
        np.array(w.data, dtype=np.float32).reshape(2, 2), DATA["w"]
    )


def test_load_gguf_arrays_dequantizes_quantized_model(tmp_path):
    # A quantized chatbot checkpoint is also a plain GGUF tensor container.
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=64)
    path = str(tmp_path / "bot.gguf")
    bot.save(path, format="gguf-q8_0")
    arrays = ai.load_gguf_arrays(path)
    embed = arrays["token_embd.weight"]
    assert len(embed) == 64 and len(embed[0]) == 64
    assert ai.detect_arrays_format(path) == "gguf"
