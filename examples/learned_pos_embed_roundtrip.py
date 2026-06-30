"""Export/import a Chatbot with learned position embeddings (loss + meta parity)."""

import sys
from pathlib import Path

import magicmindnet as ai

OUT = Path(__file__).resolve().parent / "_roundtrip_learned_pos.mmn"
FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "qa_valid.json"


def main() -> None:
    do_train = "--train" in sys.argv[1:]
    ds = ai.DatasetQA(file=str(FIXTURE), user_row="input", ai_row="output")
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=1,
        d_model=32,
        seed=8,
        use_learned_pos_embed=True,
        max_seq_len=64,
    )
    if do_train:
        ai.Train(bot, ds, ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05))
        print("trained before export")
    loss_before = bot.compute_mean_loss(ds)
    ai.export(bot, "safetensors", str(OUT))
    loaded = ai.import_model("safetensors", [str(OUT)])
    assert loaded.use_learned_pos_embed is True
    assert loaded.max_seq_len == 64
    loss_after = loaded.compute_mean_loss(ds)
    assert abs(loss_before - loss_after) < 1e-4
    print(
        f"learned pos_embed roundtrip ok: {OUT} "
        f"(loss {loss_before:.4f} -> {loss_after:.4f}, max_seq_len={loaded.max_seq_len})"
    )

if __name__ == "__main__":
    main()
