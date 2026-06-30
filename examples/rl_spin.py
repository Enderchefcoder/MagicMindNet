"""RL + SPIN on fixture QA — prints mean loss before/after."""

import sys
from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> None:
    use_bpe = "--bpe" in sys.argv[1:]
    ds = ai.DatasetQA(file=str(FIXTURES / "qa_valid.json"), user_row="input", ai_row="output")
    vocab_size = 512 if use_bpe else 256
    bpe = None
    if use_bpe:
        bpe = ai.BytePairEncoder.train(
            ["repeat repeat token"] * 12,
            vocab_size=vocab_size,
            num_merges=16,
        )
        print(f"bpe merges: {bpe.merge_count}")
    bot = ai.Chatbot(vocab_size=vocab_size, n_layer=2, d_model=32, seed=3)
    before = bot.compute_mean_loss(ds, bpe_encoder=bpe)
    cfg = ai.TrainConfig(epochs=1, batch_size=1, learning_rate=0.05)
    print(f"mean QA loss before RL: {before:.4f}")
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy", bpe_encoder=bpe)
    after_rl = bot.compute_mean_loss(ds, bpe_encoder=bpe)
    print(f"mean QA loss after RL:  {after_rl:.4f}")
    ai.SPIN(bot, 2, ds, bpe_encoder=bpe)
    after_spin = bot.compute_mean_loss(ds, bpe_encoder=bpe)
    print(f"mean QA loss after SPIN: {after_spin:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
