"""One model, every ecosystem: GGUF, PyTorch .pt, and NumPy .npz roundtrips.

All three formats are read and written by from-scratch Rust codecs — no
llama.cpp, torch, or numpy needed. `ai.load()` detects each file by content.
"""

from pathlib import Path

import magicmindnet as ai

HERE = Path(__file__).resolve().parent
GGUF = HERE / "_roundtrip_global.gguf"
TORCH = HERE / "_roundtrip_global.pt"
NPZ = HERE / "_roundtrip_global.npz"


def main() -> None:
    data = ai.DatasetQA(data=[{"input": "hi", "output": "hello!"}])
    bot = ai.Chatbot(vocab_size=96, n_layer=1, d_model=16, seed=42)
    reference_loss = bot.compute_mean_loss(data)

    bot.save(str(GGUF), format="gguf")
    bot.save(str(TORCH), format="pt")
    bot.save(str(NPZ), format="npz")

    for path in (GGUF, TORCH, NPZ):
        loaded = ai.load(str(path))
        loss = loaded.compute_mean_loss(data)
        assert abs(loss - reference_loss) < 1e-4, f"{path.name}: {loss} vs {reference_loss}"
        print(f"{path.name}: loaded via ai.load, mean loss {loss:.4f} (matches)")

    # Generic array IO — works with nested lists, numpy arrays, or torch tensors.
    arrays = {"weights": [[1.0, 2.0], [3.0, 4.0]], "bias": [0.5, -0.5]}
    ai.save_npz(str(HERE / "_roundtrip_arrays.npz"), arrays)
    ai.save_pt(str(HERE / "_roundtrip_arrays.pt"), arrays)
    assert ai.load_npz(str(HERE / "_roundtrip_arrays.npz")) == arrays
    assert ai.load_pt(str(HERE / "_roundtrip_arrays.pt")) == arrays
    print("generic .npz / .pt array roundtrips ok")

    # Self-contained GGUF: weights + SentencePiece vocab in one file.
    tok = ai.UnigramEncoder.train(["hello world", "hello there"], vocab_size=300)
    packed = ai.Chatbot(vocab_size=tok.vocab_size, n_layer=1, d_model=16, seed=7)
    sc_path = HERE / "_roundtrip_self_contained.gguf"
    packed.save(str(sc_path), format="gguf-q8_0", unigram_encoder=tok)
    info = ai.gguf_info(str(sc_path))
    tok_back = ai.load_gguf_tokenizer(str(sc_path))
    print(
        f"{sc_path.name}: {info['tensor_count']} tensors, "
        f"vocab {len(info['metadata']['tokenizer.ggml.tokens'])}, "
        f"decode ok: {tok_back.decode(tok_back.encode('hello world')) == 'hello world'}"
    )

    # HDF5 (TensorFlow/Keras bridge) — read without h5py installed.
    h5 = ai.load_h5(str(HERE.parent / "tests" / "fixtures" / "simple.h5"))
    print(f"simple.h5: {sorted(h5)} read via from-scratch HDF5 parser")


if __name__ == "__main__":
    main()
