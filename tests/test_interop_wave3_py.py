"""Wave-3 interop: sharded checkpoints, GPT-2 BPE vocabs, DEFLATE compression."""

import json
import struct

import pytest

import magicmindnet as ai

D_MODEL = 8
VOCAB = 16


def _llama_tensors():
    return {
        "model.embed_tokens.weight": ([VOCAB, D_MODEL], 0.5),
        "model.layers.0.self_attn.q_proj.weight": ([D_MODEL, D_MODEL], 0.1),
        "model.layers.0.self_attn.k_proj.weight": ([D_MODEL, D_MODEL], 0.1),
        "model.layers.0.self_attn.v_proj.weight": ([D_MODEL, D_MODEL], 0.1),
        "model.layers.0.self_attn.o_proj.weight": ([D_MODEL, D_MODEL], 0.1),
        "model.layers.0.mlp.up_proj.weight": ([D_MODEL * 4, D_MODEL], 0.2),
        "model.layers.0.mlp.down_proj.weight": ([D_MODEL, D_MODEL * 4], 0.3),
    }


def _write_pt_shard(path, tensors):
    arrays = {}
    for name, (shape, fill) in tensors.items():
        numel = 1
        for dim in shape:
            numel *= dim
        flat = [fill] * numel

        def nest(dims, values):
            if len(dims) == 1:
                return values[: dims[0]]
            size = len(values) // dims[0]
            return [nest(dims[1:], values[i * size : (i + 1) * size]) for i in range(dims[0])]

        arrays[name] = nest(shape, flat)
    ai.save_pt(str(path), arrays)


def test_sharded_pt_checkpoint_loads(tmp_path):
    tensors = _llama_tensors()
    names = list(tensors)
    first = {n: tensors[n] for n in names[:4]}
    second = {n: tensors[n] for n in names[4:]}
    _write_pt_shard(tmp_path / "pytorch_model-00001-of-00002.bin", first)
    _write_pt_shard(tmp_path / "pytorch_model-00002-of-00002.bin", second)
    weight_map = {n: "pytorch_model-00001-of-00002.bin" for n in names[:4]}
    weight_map.update({n: "pytorch_model-00002-of-00002.bin" for n in names[4:]})
    index = tmp_path / "pytorch_model.bin.index.json"
    index.write_text(json.dumps({"weight_map": weight_map}))

    loaded = ai.load(str(index))
    assert isinstance(loaded, ai.Chatbot)
    assert loaded.vocab_size == VOCAB
    assert loaded.n_layer == 1

    via_chatbot = ai.Chatbot.load(str(index))
    assert via_chatbot.d_model == D_MODEL


def test_sharded_missing_shard_errors(tmp_path):
    index = tmp_path / "model.safetensors.index.json"
    index.write_text(json.dumps({"weight_map": {"w": "gone.safetensors"}}))
    with pytest.raises((RuntimeError, ValueError), match="gone.safetensors"):
        ai.load(str(index))


def test_gpt2_bpe_encoder_roundtrip():
    # Byte-single vocab in the GPT-2 unicode space + one merge.
    tokens = [chr(b) for b in range(0x21, 0x7F)]
    tokens.append("\u0120")  # Ġ (space)
    tokens.append("hi")
    enc = ai.Gpt2BpeEncoder.from_vocab(tokens, ["h i"])
    ids = enc.encode("hi!")
    assert enc.decode(ids) == "hi!"
    assert len(ids) == 2  # "hi" merged + "!"
    assert enc.vocab_size == len(tokens)
    assert enc.token(ids[0]) == "hi"
    assert repr(enc).startswith("Gpt2BpeEncoder(")


def test_gpt2_bpe_bad_inputs():
    with pytest.raises((RuntimeError, ValueError)):
        ai.Gpt2BpeEncoder.from_vocab([], [])
    with pytest.raises((RuntimeError, ValueError), match="left right"):
        ai.Gpt2BpeEncoder.from_vocab(["a"], ["nospace"])
    enc = ai.Gpt2BpeEncoder.from_vocab(["a"], [])
    with pytest.raises(ValueError, match="out of range"):
        enc.token(5)


def test_compressed_npz_smaller_and_numpy_readable(tmp_path):
    arrays = {"w": [[1.0] * 64 for _ in range(64)]}
    stored = tmp_path / "stored.npz"
    packed = tmp_path / "packed.npz"
    ai.save_npz(str(stored), arrays)
    ai.save_npz(str(packed), arrays, compress=True)
    assert packed.stat().st_size < stored.stat().st_size / 4
    assert ai.load_npz(str(packed)) == arrays

    np = pytest.importorskip("numpy")
    loaded = np.load(str(packed))
    assert loaded["w"].shape == (64, 64)
    assert float(loaded["w"][0][0]) == 1.0


def test_compressed_npz_zlib_crosscheck(tmp_path):
    """CPython's zlib must decompress our from-scratch DEFLATE streams."""
    import zipfile

    packed = tmp_path / "z.npz"
    values = [[float(i % 7) for i in range(128)] for _ in range(16)]
    ai.save_npz(str(packed), {"x": values}, compress=True)
    with zipfile.ZipFile(packed) as zf:
        info = zf.getinfo("x.npy")
        assert info.compress_type == zipfile.ZIP_DEFLATED
        raw = zf.read("x.npy")  # zipfile inflates with zlib and checks CRC
    assert raw[:6] == b"\x93NUMPY"
    assert struct.unpack("<f", raw[-4:])[0] == float(127 % 7)


def test_gguf_bpe_tokenizer_wrong_model_errors(tmp_path):
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=2)
    tok = ai.UnigramEncoder.train(["hello"], vocab_size=260)
    path = str(tmp_path / "uni.gguf")
    bot.save(path, format="gguf", unigram_encoder=tok)
    with pytest.raises((RuntimeError, ValueError), match="llama"):
        ai.load_gguf_bpe_tokenizer(path)
