"""Minimal MagicMindNet quickstart — run from repo root after `maturin develop --release`."""

import json
import sys
import tempfile
from pathlib import Path

import magicmindnet as ai


def main() -> None:
    args = sys.argv[1:]
    learned_pe = "--learned-pe" in args
    use_bpe = "--bpe" in args
    vocab_size = 512

    with tempfile.TemporaryDirectory() as tmp:
        qa_path = Path(tmp) / "qa.json"
        qa_path.write_text(
            json.dumps(
                [
                    {"input": "repeat repeat what is 2+2?", "output": "repeat repeat 4"},
                    {"input": "Say hi", "output": "Hello!"},
                ]
            ),
            encoding="utf-8",
        )

        data = ai.DatasetQA(file=str(qa_path), user_row="input", ai_row="output")
        print(f"Loaded {data.rows} QA rows ({data.format})")

        bpe = None
        if use_bpe:
            bpe = ai.BytePairEncoder.train_from_qa(data, vocab_size=vocab_size, num_merges=16)
            if bpe.merge_count == 0:
                bpe = ai.BytePairEncoder.train(
                    ["repeat repeat token"] * 8,
                    vocab_size=vocab_size,
                    num_merges=16,
                )
            tok_path = Path(tmp) / "tokenizer.mmn"
            bpe.save(str(tok_path))
            print(f"bpe merges: {bpe.merge_count} (saved {tok_path})")

        if learned_pe:
            bot = ai.Chatbot(
                vocab_size=vocab_size,
                n_layer=2,
                d_model=64,
                use_learned_pos_embed=True,
                max_seq_len=128,
            )
            print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
        else:
            bot = ai.Chatbot(vocab_size=vocab_size, n_layer=2, d_model=64)
        print(f"Chatbot ~{bot.parameters:,} params, {bot.layer_size} layers")
        cfg = ai.TrainConfig(epochs=2, batch_size=2, cuda=False, optimizer="hybrid")
        ai.Train(bot, data, cfg, bpe_encoder=bpe)
        print("Training finished.")

        out = Path(tmp) / "bot.safetensors"
        ai.export(bot, "safetensors", str(out))
        print(f"Exported to {out}")


if __name__ == "__main__":
    main()
