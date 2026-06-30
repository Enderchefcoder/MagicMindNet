"""Quick LM training smoke: print loss before/after a few Train epochs on fixture QA."""

import sys
from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> None:
    args = sys.argv[1:]
    learned_pe = "--learned-pe" in args
    use_rope = "--rope" in args
    use_bpe = "--bpe" in args
    if learned_pe and use_rope:
        raise SystemExit("Use either --learned-pe or --rope, not both")
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    vocab_size = 512 if use_bpe else 256
    # Match tests/test_mean_qa_loss.py (regression-checked in pytest).
    if learned_pe:
        bot = ai.Chatbot(
            vocab_size=vocab_size,
            n_layer=2,
            d_model=32,
            seed=42,
            use_learned_pos_embed=True,
            max_seq_len=128,
        )
        print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
    elif use_rope:
        bot = ai.Chatbot(
            vocab_size=vocab_size,
            n_layer=2,
            d_model=32,
            seed=42,
            use_rope=True,
        )
        print(f"use_rope: {bot.use_rope} rope_theta: {bot.rope_theta}")
    else:
        bot = ai.Chatbot(vocab_size=vocab_size, n_layer=2, d_model=32, seed=42)
    bpe = None
    if use_bpe:
        texts = ["repeat repeat repeat token"] * 12
        bpe = ai.BytePairEncoder.train(texts, vocab_size=vocab_size, num_merges=24)
        print(f"bpe merges: {bpe.merge_count} (vocab_size={vocab_size})")
    before = bot.compute_mean_loss(ds, bpe_encoder=bpe)
    cfg = ai.TrainConfig(epochs=5, batch_size=1, cuda=False, optimizer="adamw", learning_rate=0.05)
    print("Training Chatbot on", path.name, "…")
    print(f"  mean loss before: {before:.4f}")
    ai.Train(bot, ds, cfg, bpe_encoder=bpe)
    after = bot.compute_mean_loss(ds, bpe_encoder=bpe)
    print(f"  mean loss after:  {after:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
