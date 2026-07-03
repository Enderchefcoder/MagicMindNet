"""Model `save()` / `load()` methods and the universal `ai.load()` dispatcher."""

import pytest

import magicmindnet as ai


def make_bot():
    return ai.Chatbot(vocab_size=256, n_layer=1, d_model=16, seed=11)


def make_clf():
    return ai.Classifier.with_labels(["a", "b"], input_dim=16, seed=5)


def test_chatbot_save_load_roundtrip(tmp_path):
    bot = make_bot()
    path = tmp_path / "bot.mmn"
    bot.save(str(path))
    loaded = ai.Chatbot.load(str(path))
    assert loaded.vocab_size == bot.vocab_size
    assert loaded.n_layer == bot.n_layer
    assert loaded.d_model == bot.d_model
    assert loaded.compute_loss("hi", "yo") == bot.compute_loss("hi", "yo")


def test_chatbot_save_load_hf_binary(tmp_path):
    bot = make_bot()
    path = tmp_path / "bot.safetensors"
    bot.save(str(path), format="hf-safetensors")
    loaded = ai.Chatbot.load(str(path))
    assert loaded.d_model == bot.d_model
    assert loaded.compute_loss("hi", "yo") == pytest.approx(
        bot.compute_loss("hi", "yo"), abs=1e-5
    )


def test_chatbot_save_load_bin_stub(tmp_path):
    bot = make_bot()
    path = tmp_path / "bot.bin"
    bot.save(str(path), format="bin")
    loaded = ai.Chatbot.load(str(path))
    assert loaded.vocab_size == bot.vocab_size
    assert loaded.n_layer == bot.n_layer


def test_chatbot_save_unknown_format_raises(tmp_path):
    with pytest.raises(ValueError, match="Unknown format"):
        make_bot().save(str(tmp_path / "x.gguf"), format="gguf")


def test_chatbot_load_missing_file_mentions_path():
    with pytest.raises(Exception, match="missing/bot.mmn"):
        ai.Chatbot.load("missing/bot.mmn")


def test_classifier_save_load_roundtrip(tmp_path):
    clf = make_clf()
    path = tmp_path / "clf.mmn"
    clf.save(str(path))
    loaded = ai.Classifier.load(str(path))
    assert loaded.labels == clf.labels
    assert loaded.input_dim == clf.input_dim
    assert loaded.predict("hello") == clf.predict("hello")


def test_diffusion_save_load_roundtrip(tmp_path):
    diff = ai.Diffusion()
    path = tmp_path / "diff.mmn"
    diff.save(str(path))
    loaded = ai.Diffusion.load(str(path))
    assert loaded.latent_channels == diff.latent_channels
    assert loaded.parameters == diff.parameters


def test_universal_load_dispatches_chatbot(tmp_path):
    path = tmp_path / "bot.mmn"
    make_bot().save(str(path))
    model = ai.load(str(path))
    assert isinstance(model, ai.Chatbot)


def test_universal_load_dispatches_hf_binary_chatbot(tmp_path):
    path = tmp_path / "bot.safetensors"
    make_bot().save(str(path), format="hf-safetensors")
    model = ai.load(str(path))
    assert isinstance(model, ai.Chatbot)


def test_universal_load_dispatches_classifier(tmp_path):
    path = tmp_path / "clf.mmn"
    make_clf().save(str(path))
    model = ai.load(str(path))
    assert isinstance(model, ai.Classifier)


def test_universal_load_dispatches_diffusion(tmp_path):
    path = tmp_path / "diff.mmn"
    ai.Diffusion().save(str(path))
    model = ai.load(str(path))
    assert isinstance(model, ai.Diffusion)


def test_universal_load_dispatches_bin_stub(tmp_path):
    path = tmp_path / "bot.bin"
    make_bot().save(str(path), format="bin")
    model = ai.load(str(path))
    assert isinstance(model, ai.Chatbot)


def test_universal_load_rejects_non_checkpoint(tmp_path):
    path = tmp_path / "junk.json"
    path.write_text('{"hello": "world"}', encoding="utf-8")
    with pytest.raises(Exception, match="format"):
        ai.load(str(path))


def test_chatbot_load_rejects_classifier_checkpoint(tmp_path):
    path = tmp_path / "clf.mmn"
    make_clf().save(str(path))
    with pytest.raises(ValueError, match="Classifier"):
        ai.Chatbot.load(str(path))


def test_classifier_load_rejects_chatbot_checkpoint(tmp_path):
    path = tmp_path / "bot.mmn"
    make_bot().save(str(path))
    with pytest.raises(ValueError, match="Chatbot"):
        ai.Classifier.load(str(path))


def test_diffusion_load_rejects_chatbot_checkpoint(tmp_path):
    path = tmp_path / "bot.mmn"
    make_bot().save(str(path))
    with pytest.raises(ValueError, match="Chatbot"):
        ai.Diffusion.load(str(path))


def test_save_matches_export_function(tmp_path):
    bot = make_bot()
    method_path = tmp_path / "method.mmn"
    func_path = tmp_path / "func.mmn"
    bot.save(str(method_path))
    ai.export(bot, "safetensors", str(func_path))
    assert method_path.read_text(encoding="utf-8") == func_path.read_text(encoding="utf-8")


def test_save_with_bpe_sidecar(tmp_path):
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=16, seed=2)
    bpe = ai.BytePairEncoder.train(["hello world hello"] * 8, vocab_size=512, num_merges=8)
    path = tmp_path / "bot.mmn"
    bot.save(str(path), bpe_encoder=bpe)
    assert (tmp_path / "bot.bpe.mmn").is_file()
    assert ai.load_bpe_sidecar(path) is not None
