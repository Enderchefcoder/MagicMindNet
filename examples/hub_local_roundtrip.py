#!/usr/bin/env python3
"""Local hub roundtrip: from_pretrained → generate/train → save (no network)."""

from __future__ import annotations

import tempfile
from pathlib import Path

import magicmindnet as ai


def main() -> None:
    bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, seed=0, max_seq_len=64)
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bot.mmn"
        bot.save(str(path))
        model = ai.from_pretrained(str(path))
        assert model.native is not None
        text = model.generate("hi", max_new_tokens=4)
        losses = model.finetune(
            ai.DatasetCorpus(data=["hello world from hub"] * 4),
            epochs=1,
            learning_rate=1e-2,
            batch_size=1,
        )
        out = Path(tmp) / "trained.mmn"
        model.save(str(out))
        caps = model.capabilities()
        print(f"generate: {text!r}")
        print(f"finetune losses: {losses}")
        print(f"capabilities: {caps}")
        print(f"families: {ai.list_hub_families()[:5]}…")
        print("hub_local_roundtrip: OK")


if __name__ == "__main__":
    main()
