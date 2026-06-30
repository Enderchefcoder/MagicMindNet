"""Unigram tokenizer save/load + optional Chatbot export sidecar roundtrip."""

from pathlib import Path

import magicmindnet as ai

OUT = Path(__file__).resolve().parent / "_roundtrip_unigram.mmn"


def main() -> None:
    texts = ["hello hello world", "hello there friend", "repeat repeat token"]
    enc = ai.UnigramEncoder.train(texts, vocab_size=512)
    sample = "hello repeat"
    assert enc.decode(enc.encode(sample)) == sample

    tok_path = OUT.with_suffix(".tokenizer.mmn")
    enc.save(str(tok_path))
    loaded = ai.UnigramEncoder.load(str(tok_path))
    assert loaded.encode(sample) == enc.encode(sample)
    print(
        f"unigram roundtrip ok: {tok_path.name} "
        f"(pieces={loaded.piece_count}, vocab_size={loaded.vocab_size}, encode len={len(loaded.encode(sample))})"
    )

    if "--train" in __import__("sys").argv:
        data = ai.DatasetQA(
            file="tests/fixtures/qa_valid.json",
            user_row="input",
            ai_row="output",
        )
        bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=6)
        ai.Train(
            bot,
            data,
            ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05),
            unigram_encoder=loaded,
        )
        print("trained before export")
        ckpt = OUT
        ai.export(bot, "safetensors", str(ckpt), unigram_encoder=loaded)
        sidecar = ai.load_unigram_sidecar(ckpt)
        assert sidecar is not None
        print(f"export sidecar ok: {ckpt.name} + meta.unigram_checkpoint")


if __name__ == "__main__":
    main()
