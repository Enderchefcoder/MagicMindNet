"""Minimal MagicMindNet quickstart — run from repo root after `maturin develop --release`."""

import json
import sys
import tempfile
from pathlib import Path

import magicmindnet as ai


def main() -> None:
    learned_pe = "--learned-pe" in sys.argv[1:]
    with tempfile.TemporaryDirectory() as tmp:
        qa_path = Path(tmp) / "qa.json"
        qa_path.write_text(
            json.dumps(
                [
                    {"input": "What is 2+2?", "output": "4"},
                    {"input": "Say hi", "output": "Hello!"},
                ]
            ),
            encoding="utf-8",
        )

        data = ai.DatasetQA(file=str(qa_path), user_row="input", ai_row="output")
        print(f"Loaded {data.rows} QA rows ({data.format})")

        if learned_pe:
            bot = ai.Chatbot(
                vocab_size=512,
                n_layer=2,
                d_model=64,
                use_learned_pos_embed=True,
                max_seq_len=128,
            )
            print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
        else:
            bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64)
        print(f"Chatbot ~{bot.parameters:,} params, {bot.layer_size} layers")
        cfg = ai.TrainConfig(epochs=2, batch_size=2, cuda=False, optimizer="hybrid")
        ai.Train(bot, data, cfg)
        print("Training finished.")

        out = Path(tmp) / "bot.safetensors"
        ai.export(bot, "safetensors", str(out))
        print(f"Exported to {out}")


if __name__ == "__main__":
    main()
