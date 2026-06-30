"""Export a tiny Chatbot checkpoint and reload it (shape + parameter count)."""

from pathlib import Path

import magicmindnet as ai

OUT = Path(__file__).resolve().parent / "_roundtrip_chatbot.mmn"


def main() -> None:
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, seed=7)
    params_before = bot.parameters
    ai.export(bot, "safetensors", str(OUT))
    loaded = ai.import_model("safetensors", [str(OUT)])
    assert loaded.parameters == params_before
    assert loaded.layer_size == bot.layer_size
    assert loaded.init_seed == 7
    print(f"roundtrip ok: {OUT} ({params_before} params, seed={loaded.init_seed})")


if __name__ == "__main__":
    main()
