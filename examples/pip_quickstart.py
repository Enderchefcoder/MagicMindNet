#!/usr/bin/env python3
"""Minimal pip-install quickstart — no repo checkout required beyond this file."""

from __future__ import annotations

import magicmindnet as ai


def main() -> None:
    print(f"magicmindnet {ai.__version__}")
    data = ai.DatasetQA(
        data=[
            {"input": "hi", "output": "hello!"},
            {"input": "bye", "output": "see you!"},
        ]
    )
    bot = ai.Chatbot(vocab_size=256, n_layer=1, d_model=32, seed=7)
    losses = bot.train(data, epochs=3, verbose=False)
    reply = bot.chat("hi", max_new_tokens=8, temperature=0.0)
    print(f"epoch_losses={losses}")
    print(f"chat(hi)={reply!r}")
    path = "_pip_quickstart.mmn"
    bot.save(path)
    loaded = ai.load(path)
    print(f"reloaded={type(loaded).__name__} d_model={loaded.d_model}")
    print("pip_quickstart ok")


if __name__ == "__main__":
    main()
