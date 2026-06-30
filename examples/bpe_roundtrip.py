"""Train a BPE tokenizer, save/load mmn-bpe-v1, and optionally train Chatbot with it."""

import sys
from pathlib import Path

import magicmindnet as ai

OUT = Path(__file__).resolve().parent / "_roundtrip_bpe.mmn"
FIXTURE = Path(__file__).resolve().parent.parent / "tests" / "fixtures" / "qa_valid.json"


def main() -> None:
    do_train = "--train" in sys.argv[1:]
    sample = "repeat repeat token hello"
    enc = ai.BytePairEncoder.train(
        ["repeat repeat token"] * 12,
        vocab_size=512,
        num_merges=24,
    )
    before_ids = enc.encode(sample)
    enc.save(str(OUT))
    loaded = ai.BytePairEncoder.load(str(OUT))
    after_ids = loaded.encode(sample)
    assert before_ids == after_ids, (before_ids, after_ids)
    if do_train:
        ds = ai.DatasetQA(file=str(FIXTURE), user_row="input", ai_row="output")
        bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=9)
        loss0 = bot.compute_mean_loss(ds, bpe_encoder=loaded)
        ai.Train(
            bot,
            ds,
            ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05),
            bpe_encoder=loaded,
        )
        loss1 = bot.compute_mean_loss(ds, bpe_encoder=loaded)
        print(f"trained with BPE: loss {loss0:.4f} -> {loss1:.4f}")
        ckpt = OUT.with_name("_roundtrip_bot_with_bpe.mmn")
        ai.export(bot, "safetensors", str(ckpt), bpe_encoder=loaded)
        sidecar = ai.load_bpe_sidecar(ckpt)
        assert sidecar is not None
        assert sidecar.encode(sample) == loaded.encode(sample)
        print(f"export sidecar ok: {ckpt.name} + meta.bpe_checkpoint")
    print(
        f"bpe roundtrip ok: {OUT} "
        f"(merges={loaded.merge_count}, vocab_size={loaded.vocab_size}, encode len={len(after_ids)})"
    )


if __name__ == "__main__":
    main()
