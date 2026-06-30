"""Train Chatbot on a small corpus fixture (next-token LM)."""

import sys
from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def main() -> None:
    args = sys.argv[1:]
    learned_pe = "--learned-pe" in args
    use_bpe = "--bpe" in args
    ds = ai.DatasetCorpus(
        use_two_files=True,
        rowfile=str(FIXTURES / "corpus_rows.json"),
        txtfile=str(FIXTURES / "corpus.txt"),
    )
    vocab_size = 512 if use_bpe else 256
    if learned_pe:
        bot = ai.Chatbot(
            vocab_size=vocab_size,
            n_layer=2,
            d_model=32,
            seed=7,
            use_learned_pos_embed=True,
            max_seq_len=128,
        )
        print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
    else:
        bot = ai.Chatbot(vocab_size=vocab_size, n_layer=2, d_model=32, seed=7)
    bpe = None
    if use_bpe:
        bpe = ai.BytePairEncoder.train_from_corpus(ds, vocab_size=vocab_size, num_merges=24)
        if bpe.merge_count == 0:
            bpe = ai.BytePairEncoder.train(
                ["repeat repeat corpus"] * 12,
                vocab_size=vocab_size,
                num_merges=24,
            )
        print(f"bpe merges: {bpe.merge_count} (vocab_size={vocab_size})")
    before = bot.compute_mean_loss(ds, bpe_encoder=bpe)
    cfg = ai.TrainConfig(epochs=5, batch_size=2, learning_rate=0.05)
    ai.Train(bot, ds, cfg, bpe_encoder=bpe)
    after = bot.compute_mean_loss(ds, bpe_encoder=bpe)
    print(f"mean corpus loss before: {before:.4f}")
    print(f"mean corpus loss after:  {after:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
