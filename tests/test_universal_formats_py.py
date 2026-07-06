"""Universal checkpoint detection: ai.load() across every supported format."""

import zipfile

import pytest

import magicmindnet as ai

FORMATS = ["safetensors", "hf-safetensors", "gguf", "npz", "pt"]


def make_bot(seed=31):
    return ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, seed=seed)


@pytest.mark.parametrize("format", FORMATS)
def test_universal_load_every_format(tmp_path, format):
    path = str(tmp_path / f"bot.{format.replace('-', '_')}")
    bot = make_bot()
    bot.save(path, format=format)
    loaded = ai.load(path)
    assert isinstance(loaded, ai.Chatbot)
    assert loaded.vocab_size == bot.vocab_size
    assert loaded.d_model == bot.d_model


@pytest.mark.parametrize("format", FORMATS)
def test_chatbot_load_autodetects_every_format(tmp_path, format):
    path = str(tmp_path / "checkpoint.model")
    make_bot().save(path, format=format)
    loaded = ai.Chatbot.load(path)
    assert loaded.n_layer == 1


@pytest.mark.parametrize("format", FORMATS)
def test_loss_survives_every_format(tmp_path, format):
    data = ai.DatasetQA(data=[{"input": "ping", "output": "pong"}])
    bot = make_bot(seed=41)
    before = bot.compute_mean_loss(data)
    path = str(tmp_path / "bot.any")
    bot.save(path, format=format)
    after = ai.load(path).compute_mean_loss(data)
    assert abs(before - after) < 1e-4


def test_import_model_error_lists_supported_formats():
    with pytest.raises(ValueError, match="gguf"):
        ai.import_model("unknown-format", ["x.bin"])


def test_load_unrecognized_zip_errors(tmp_path):
    path = tmp_path / "not_a_model.zip"
    with zipfile.ZipFile(path, "w") as zf:
        zf.writestr("readme.txt", "hello")
    with pytest.raises((RuntimeError, ValueError), match="neither a torch checkpoint"):
        ai.load(str(path))


def test_trained_model_survives_gguf(tmp_path):
    """Training then GGUF roundtrip keeps the trained weights."""
    data = ai.DatasetQA(data=[{"input": "a", "output": "b"}])
    bot = make_bot(seed=51)
    bot.train(data, epochs=2, verbose=False)
    trained_loss = bot.compute_mean_loss(data)
    path = str(tmp_path / "trained.gguf")
    bot.save(path, format="gguf")
    loaded_loss = ai.load(path).compute_mean_loss(data)
    assert abs(trained_loss - loaded_loss) < 1e-4
