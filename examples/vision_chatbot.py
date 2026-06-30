"""Train a vision-flag chatbot (metadata path) and roundtrip checkpoint."""

import json
import tempfile
from pathlib import Path

import magicmindnet as ai

data = [
    {"input": "What is 2+2?", "output": "Four."},
    {"input": "Say hi.", "output": "Hello!"},
]

with tempfile.TemporaryDirectory() as tmp:
    qa_path = Path(tmp) / "qa.json"
    qa_path.write_text(json.dumps(data), encoding="utf-8")
    ds = ai.DatasetQA(str(qa_path), "input", "output")
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, vision=True, seed=3)
    print("has_vision:", bot.has_vision)
    print("has_vision_patch_encoder:", bot.has_vision_patch_encoder)
    print("has_vision_rgb_conv:", bot.has_vision_rgb_conv)
    print("vision_patch_dim:", bot.vision_patch_dim)
    print("vision_rgb_dim:", bot.vision_rgb_dim)
    patch = ai.vision_rgb_patch_from_text("demo image surrogate")
    print("rgb patch len:", len(patch))
    loss0 = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    loss1 = bot.compute_mean_loss(ds)
    print(f"mean loss: {loss0:.4f} -> {loss1:.4f}")
    ckpt = Path(tmp) / "vision.mmn"
    ai.export(bot, "safetensors", str(ckpt))
    ai.quantize(bot, "int8")
    loaded = ai.import_model("safetensors", [str(ckpt)])
    print("loaded has_vision:", loaded.has_vision)
    assert loaded.has_vision is True
