"""RL + SPIN on fixture QA — prints mean loss before/after."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).resolve().parent.parent / "tests" / "fixtures"


def main() -> None:
    ds = ai.DatasetQA(file=str(FIXTURES / "qa_valid.json"), user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32, seed=3)
    before = bot.compute_mean_loss(ds)
    cfg = ai.TrainConfig(epochs=1, batch_size=1, learning_rate=0.05)
    print(f"mean QA loss before RL: {before:.4f}")
    ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy")
    after_rl = bot.compute_mean_loss(ds)
    print(f"mean QA loss after RL:  {after_rl:.4f}")
    ai.SPIN(bot, 2, ds)
    after_spin = bot.compute_mean_loss(ds)
    print(f"mean QA loss after SPIN: {after_spin:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
