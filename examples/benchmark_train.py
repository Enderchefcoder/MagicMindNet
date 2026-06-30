"""Quick LM training smoke: print loss before/after a few Train epochs on fixture QA."""

import sys
from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> None:
    learned_pe = "--learned-pe" in sys.argv[1:]
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    # Match tests/test_mean_qa_loss.py (regression-checked in pytest).
    if learned_pe:
        bot = ai.Chatbot(
            vocab_size=256,
            n_layer=2,
            d_model=32,
            seed=42,
            use_learned_pos_embed=True,
            max_seq_len=128,
        )
        print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
    else:
        bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=42)
    before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=5, batch_size=1, cuda=False, optimizer="adamw", learning_rate=0.05)
    print("Training Chatbot on", path.name, "…")
    print(f"  mean loss before: {before:.4f}")
    ai.Train(bot, ds, cfg)
    after = bot.compute_mean_loss(ds)
    print(f"  mean loss after:  {after:.4f}")
    print("Done.")

if __name__ == "__main__":
    main()
