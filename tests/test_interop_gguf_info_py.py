"""GGUF inspection (gguf_info) and new export quantizations."""

import pytest

import magicmindnet as ai


def make_bot(seed=9, d_model=32):
    return ai.Chatbot(vocab_size=64, n_layer=2, d_model=d_model, seed=seed)


def test_gguf_info_reports_metadata_and_tensors(tmp_path):
    path = str(tmp_path / "bot.gguf")
    make_bot().save(path, format="gguf")
    info = ai.gguf_info(path)
    assert info["version"] == 3
    assert info["metadata"]["general.architecture"] == "mmn"
    assert info["metadata"]["mmn.block_count"] == 2
    assert info["metadata"]["mmn.embedding_length"] == 32
    names = [t["name"] for t in info["tensors"]]
    assert "token_embd.weight" in names
    assert "blk.1.ffn_down.weight" in names
    embd = next(t for t in info["tensors"] if t["name"] == "token_embd.weight")
    assert embd["shape"] == [64, 32]
    assert embd["type"] == "F32"


def test_gguf_info_missing_file_errors():
    with pytest.raises((RuntimeError, ValueError), match="cannot read"):
        ai.gguf_info("/nonexistent/model.gguf")


@pytest.mark.parametrize("format", ["gguf-f16", "gguf-q4_0", "gguf-q8_0"])
def test_gguf_quantized_export_roundtrips(tmp_path, format):
    data = ai.DatasetQA(data=[{"input": "hi", "output": "hello"}])
    bot = make_bot(seed=19)
    before = bot.compute_mean_loss(data)
    path = str(tmp_path / "bot.gguf")
    bot.save(path, format=format)
    loaded = ai.load(path)
    after = loaded.compute_mean_loss(data)
    # Quantization shifts the loss slightly; f16 is near-exact.
    tolerance = 0.02 if format == "gguf-f16" else 1.0
    assert abs(before - after) < tolerance, f"{format}: {before} vs {after}"


def test_gguf_info_reports_quant_type(tmp_path):
    path = str(tmp_path / "bot_q4.gguf")
    make_bot().save(path, format="gguf-q4_0")
    info = ai.gguf_info(path)
    types = {t["type"] for t in info["tensors"]}
    assert "Q4_0" in types


def test_gguf_tokenizer_missing_errors(tmp_path):
    path = str(tmp_path / "bot.gguf")
    make_bot().save(path, format="gguf")
    with pytest.raises((RuntimeError, ValueError), match="no embedded tokenizer"):
        ai.load_gguf_tokenizer(path)


def test_gguf_self_contained_model_with_embedded_vocab(tmp_path):
    """Weights + vocabulary in one GGUF file, llama.cpp style."""
    enc = ai.UnigramEncoder.train(["hello world", "hello there general"], vocab_size=300)
    bot = ai.Chatbot(vocab_size=enc.vocab_size, n_layer=1, d_model=16, seed=3)
    path = str(tmp_path / "self_contained.gguf")
    bot.save(path, format="gguf", unigram_encoder=enc)

    info = ai.gguf_info(path)
    assert info["metadata"]["tokenizer.ggml.model"] == "llama"
    assert len(info["metadata"]["tokenizer.ggml.tokens"]) == enc.piece_count

    loaded = ai.load(path)
    tok = ai.load_gguf_tokenizer(path)
    assert tok.piece_count == enc.piece_count
    assert tok.decode(tok.encode("hello world")) == "hello world"
    reply = loaded.chat("hello", unigram_encoder=tok)
    assert isinstance(reply, str)


def test_gguf_rejects_bpe_embedding(tmp_path):
    enc = ai.BytePairEncoder.train(["hello hello"], vocab_size=300, num_merges=4)
    bot = make_bot()
    with pytest.raises(ValueError, match="unigram"):
        bot.save(str(tmp_path / "x.gguf"), format="gguf", bpe_encoder=enc)


def test_npz_pt_reject_tokenizer_sidecars(tmp_path):
    enc = ai.UnigramEncoder.train(["hello"], vocab_size=260)
    bot = make_bot()
    for format in ["npz", "pt"]:
        with pytest.raises(ValueError, match="safetensors or gguf"):
            bot.save(str(tmp_path / f"x.{format}"), format=format, unigram_encoder=enc)
