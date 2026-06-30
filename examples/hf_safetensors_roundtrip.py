"""Export a Chatbot as Hugging Face binary safetensors and reload it."""

from pathlib import Path

import magicmindnet as ai

OUT = Path(__file__).resolve().parent / "_roundtrip_hf.safetensors"


def main() -> None:
    bot = ai.Chatbot(vocab_size=128, n_layer=2, d_model=32, vision=True, seed=5)
    loss_before = bot.compute_loss("hello", "world")
    ai.export(bot, "hf-safetensors", str(OUT))
    raw = OUT.read_bytes()
    assert not raw.startswith(b"{"), "expected binary safetensors, not JSON"
    loaded = ai.import_model("safetensors", [str(OUT)])
    loss_after = loaded.compute_loss("hello", "world")
    assert loaded.has_vision == bot.has_vision
    assert loss_after == loss_before
    print(f"hf safetensors roundtrip ok: {OUT} (loss {loss_before:.4f})")


if __name__ == "__main__":
    main()
