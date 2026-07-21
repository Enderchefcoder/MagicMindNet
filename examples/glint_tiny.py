#!/usr/bin/env python3
"""Tiny Glint-style Chatbot: RMSNorm + SwiGLU + n_loops + tied embeds."""

from __future__ import annotations

import magicmindnet as ai


def main() -> None:
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=64,
        seed=0,
        max_seq_len=64,
        n_loops=2,
        norm="rms",
        ffn="swiglu",
        tie_embeddings=True,
        use_rope=True,
    )
    print(f"norm={bot.norm} ffn={bot.ffn} n_loops={bot.n_loops} params={bot.parameters}")
    data = ai.DatasetCorpus(data=["glint style tiny train " * 4] * 8)
    before = bot.compute_mean_loss(data)
    bot.train(data, epochs=2, learning_rate=1e-2, batch_size=1)
    after = bot.compute_mean_loss(data)
    text = bot.generate("hi", max_new_tokens=8)
    print(f"loss {before:.4f} -> {after:.4f}")
    print(f"generate: {text!r}")
    print("glint_tiny: OK")


if __name__ == "__main__":
    main()
