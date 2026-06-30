"""Vision patch encoder forward path and checkpoint roundtrip."""

import base64
from pathlib import Path

import pytest

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"

# Valid 1×1 PNG (decoded by the `image` crate in Rust).
_MINI_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)


def test_vision_chatbot_has_patch_encoder():
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, vision=True, seed=1)
    assert bot.has_vision is True
    assert bot.has_vision_patch_encoder is True
    assert bot.has_vision_rgb_conv is True
    assert bot.has_vision_cross_attn is True
    assert bot.vision_patch_dim == ai.VISION_PATCH_DIM == 64
    assert bot.vision_rgb_dim == ai.VISION_RGB_DIM == 192


def test_vision_rgb_patch_changes_compute_loss():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, vision=True, seed=2)
    gray = ai.vision_patch_from_text("gray patch")
    rgb = ai.vision_rgb_patch_from_text("rgb patch")
    loss_gray = bot.compute_loss("hi", "hello", image_patch=gray)
    loss_rgb = bot.compute_loss("hi", "hello", image_patch=rgb)
    assert loss_gray != loss_rgb


def test_vision_rgb_default_patch_from_input():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, vision=True, seed=6)
    loss_auto = bot.compute_loss("photo prompt", "caption")
    rgb = ai.vision_rgb_patch_from_text("photo prompt")
    loss_explicit = bot.compute_loss("photo prompt", "caption", image_patch=rgb)
    assert loss_auto == pytest.approx(loss_explicit)


def test_vision_cross_attn_trains(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, vision=True, seed=7)
    assert bot.has_vision_cross_attn is True
    loss_before = bot.compute_mean_loss(ds)
    ai.Train(bot, ds, ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05))
    loss_after = bot.compute_mean_loss(ds)
    assert loss_after < loss_before
    path = tmp_path / "vision_cross.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.has_vision_cross_attn is True
    assert loaded.compute_mean_loss(ds) == pytest.approx(bot.compute_mean_loss(ds))


def test_vision_patch_changes_compute_loss():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, vision=True, seed=2)
    loss_default_patch = bot.compute_loss("hi", "hello")
    patch = ai.vision_patch_from_text("alternate patch bytes")
    loss_custom_patch = bot.compute_loss("hi", "hello", image_patch=patch)
    assert loss_default_patch != loss_custom_patch


def test_vision_patch_wrong_length_raises():
    bot = ai.Chatbot(vocab_size=64, n_layer=1, d_model=16, vision=True, seed=3)
    with pytest.raises(ai.DataMismatchError):
        bot.compute_loss("a", "b", image_patch=[0.1, 0.2])


def test_vision_patch_proj_checkpoint_roundtrip(tmp_path: Path):
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=16, vision=True, seed=4)
    w_before = bot.compute_loss("x", "y")  # touch forward
    del w_before
    path = tmp_path / "vision_patch.mmn"
    ai.export(bot, "safetensors", str(path))
    loaded = ai.import_model("safetensors", [str(path)])
    assert loaded.has_vision_patch_encoder is True
    assert loaded.compute_loss("x", "y") == pytest.approx(bot.compute_loss("x", "y"))


def test_vision_chatbot_trains_patch_encoder(tmp_path: Path):
    ds = ai.DatasetQA(
        file=str(FIXTURES / "qa_valid.json"),
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, vision=True, seed=5)
    loss_before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    loss_after = bot.compute_mean_loss(ds)
    assert loss_after < loss_before


def test_vision_rgb_patch_from_image_file(tmp_path: Path):
    img_path = tmp_path / "red.png"
    img_path.write_bytes(_MINI_PNG)
    patch = ai.vision_rgb_patch_from_image_path(str(img_path))
    assert len(patch) == ai.VISION_RGB_DIM
    assert any(v > 0.0 for v in patch)


def test_qa_dataset_loads_image_path(tmp_path: Path):
    img_path = tmp_path / "scene.png"
    img_path.write_bytes(_MINI_PNG)
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        '[{"input":"describe","output":"red","image":"scene.png"}]',
        encoding="utf-8",
    )
    ds = ai.DatasetQA(str(qa_path), image_row="image")
    assert ds.sample_image_path(0) == "scene.png"
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, vision=True, seed=8)
    loss_file = bot.compute_mean_loss(ds)
    ds_text = ai.DatasetQA(str(qa_path), image_row="")
    loss_text = bot.compute_mean_loss(ds_text)
    assert loss_file != loss_text


def test_multi_patch_cross_attn_changes_loss():
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, vision=True, seed=9)
    p1 = ai.vision_rgb_patch_from_text("tile-a")
    p2 = ai.vision_rgb_patch_from_text("tile-b")
    loss_one = bot.compute_loss("describe", "caption", image_patch=p1)
    loss_two = bot.compute_loss("describe", "caption", image_patches=[p1, p2])
    assert loss_one != loss_two


def test_vision_rgb_patches_grid_from_image(tmp_path: Path):
    img_path = tmp_path / "split.png"
    img_path.write_bytes(_MINI_PNG)
    patches = ai.vision_rgb_patches_from_image_path(str(img_path), grid=2)
    assert len(patches) == 4
    assert all(len(p) == ai.VISION_RGB_DIM for p in patches)


def test_qa_dataset_multi_image_paths(tmp_path: Path):
    qa_path = tmp_path / "qa.json"
    qa_path.write_text(
        '[{"input":"describe","output":"red","image":"a.png,b.png"}]',
        encoding="utf-8",
    )
    ds = ai.DatasetQA(str(qa_path), image_row="image")
    assert ds.sample_image_paths(0) == ["a.png", "b.png"]
