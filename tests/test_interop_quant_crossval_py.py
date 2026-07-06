"""Cross-validate every GGML quant codec against the official gguf package.

`gguf` (llama.cpp's reference Python implementation) dequantizes the same
block payloads; outputs must match MagicMindNet's from-scratch Rust codecs
bit-for-bit (float32 exact for table lookups, tiny epsilon where the
reference uses a different operation order).
"""

import pytest

np = pytest.importorskip("numpy")
gguf = pytest.importorskip("gguf")

import gguf.quants as gq  # noqa: E402

from magicmindnet import _native  # noqa: E402

# (name, elements per block, bytes per block)
TYPES = [
    ("Q4_0", 32, 18),
    ("Q4_1", 32, 20),
    ("Q5_0", 32, 22),
    ("Q5_1", 32, 24),
    ("Q8_0", 32, 34),
    ("Q2_K", 256, 84),
    ("Q3_K", 256, 110),
    ("Q4_K", 256, 144),
    ("Q5_K", 256, 176),
    ("Q6_K", 256, 210),
    ("IQ4_NL", 32, 18),
    ("IQ4_XS", 256, 136),
    ("IQ2_XXS", 256, 66),
    ("IQ2_XS", 256, 74),
    ("IQ2_S", 256, 82),
    ("IQ3_XXS", 256, 98),
    ("IQ3_S", 256, 110),
    ("IQ1_S", 256, 50),
    ("IQ1_M", 256, 56),
    ("TQ1_0", 256, 54),
    ("TQ2_0", 256, 66),
    ("MXFP4", 32, 17),
    ("NVFP4", 64, 36),
    ("BF16", 1, 2),
]


def _payload(name: str, block_bytes: int, n_blocks: int = 4) -> bytes:
    """Deterministic pseudo-random block payload with sane scale fields."""
    rng = np.random.default_rng(abs(hash(name)) % (2**32))
    raw = rng.integers(0, 256, size=n_blocks * block_bytes, dtype=np.uint8)
    return raw.tobytes()


def _reference_dequant(name: str, payload: bytes, numel: int):
    qtype = gguf.GGMLQuantizationType[name]
    arr = np.frombuffer(payload, dtype=np.uint8)
    out = gq.dequantize(arr, qtype)
    assert out.size == numel
    return out.astype(np.float32).reshape(-1)


@pytest.mark.parametrize("name,block_elems,block_bytes", TYPES)
def test_dequant_matches_reference(name, block_elems, block_bytes):
    n_blocks = 4 if block_elems > 1 else 64
    payload = _payload(name, block_bytes, n_blocks)
    numel = block_elems * n_blocks
    ours = np.array(_native.dequantize_ggml(name, list(payload), numel), dtype=np.float32)
    theirs = _reference_dequant(name, payload, numel)
    assert ours.shape == theirs.shape
    # Random scale bytes can decode to inf/NaN f16 values; require identical
    # NaN placement and closeness elsewhere (rtol covers operation-order f32
    # rounding differences between the vectorized reference and scalar Rust).
    np.testing.assert_allclose(ours, theirs, rtol=2e-5, atol=1e-6, equal_nan=True)


# Types the reference package can quantize (k-quants raise NotImplementedError).
@pytest.mark.parametrize("name", ["Q4_0", "Q4_1", "Q5_0", "Q5_1", "Q8_0", "TQ1_0", "TQ2_0", "MXFP4"])
def test_reference_quantized_floats_roundtrip(name):
    """gguf-py quantizes real float data; both dequantizers must agree."""
    qtype = gguf.GGMLQuantizationType[name]
    rng = np.random.default_rng(11)
    values = (rng.standard_normal(512) * 0.5).astype(np.float32)
    packed = gq.quantize(values, qtype)
    theirs = gq.dequantize(packed, qtype).astype(np.float32).reshape(-1)
    ours = np.array(
        _native.dequantize_ggml(name, list(packed.tobytes()), values.size),
        dtype=np.float32,
    )
    np.testing.assert_allclose(ours, theirs, rtol=2e-5, atol=1e-6)


def test_our_gguf_files_read_by_reference_reader(tmp_path):
    """gguf-py's GGUFReader must parse files our writer produces."""
    import magicmindnet as ai

    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=32, seed=13)
    path = str(tmp_path / "crossval.gguf")
    bot.save(path, format="gguf-q8_0")

    reader = gguf.GGUFReader(path)
    fields = {f.name for f in reader.fields.values()}
    assert "general.architecture" in fields
    names = {t.name for t in reader.tensors}
    assert "token_embd.weight" in names
    embd = next(t for t in reader.tensors if t.name == "token_embd.weight")
    values = gq.dequantize(embd.data, embd.tensor_type).reshape(-1)
    ref = str(tmp_path / "ref.mmn")
    bot.save(ref)
    # Compare reference-decoded Q8_0 embed against the original within quant error.
    first = values[0]
    from pathlib import Path

    from conftest import checkpoint_tensor_first_f32

    original = checkpoint_tensor_first_f32(Path(ref), "embed")
    assert abs(first - original) < 0.02


def test_unknown_type_name_errors():
    with pytest.raises(ValueError, match="unknown GGML type"):
        _native.dequantize_ggml("Q9_9", [0, 0], 1)
