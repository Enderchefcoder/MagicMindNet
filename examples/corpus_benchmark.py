"""Train Chatbot on a small corpus fixture (next-token LM)."""

import sys
from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    learned_pe = "--learned-pe" in sys.argv[1:]
    ds = ai.DatasetCorpus(
        use_two_files=True,
        rowfile=str(FIXTURES / "corpus_rows.json"),
        txtfile=str(FIXTURES / "corpus.txt"),
    )
    if learned_pe:
        bot = ai.Chatbot(
            vocab_size=256,
            n_layer=2,
            d_model=32,
            seed=7,
            use_learned_pos_embed=True,
            max_seq_len=128,
        )
        print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
    else:
        bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=7)
    before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=5, batch_size=2, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    print(f"mean corpus loss before: {before:.4f}")
    print(f"mean corpus loss after:  {after:.4f}")
    print("Done.")

if __name__ == "__main__":
    main()
