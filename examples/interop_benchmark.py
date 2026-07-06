"""Benchmark save/load across every checkpoint format (time + file size).

All codecs are from-scratch Rust; GGUF dequant, torch storage decode, and
zip entry inflation run in parallel across cores.
"""

import os
import time
from pathlib import Path

import magicmindnet as ai

HERE = Path(__file__).resolve().parent

FORMATS = [
    ("safetensors", "_bench.mmn"),
    ("hf-safetensors", "_bench.safetensors"),
    ("gguf", "_bench_f32.gguf"),
    ("gguf-f16", "_bench_f16.gguf"),
    ("gguf-q8_0", "_bench_q8.gguf"),
    ("gguf-q6_k", "_bench_q6k.gguf"),
    ("gguf-q4_k", "_bench_q4k.gguf"),
    ("npz", "_bench.npz"),
    ("pt", "_bench.pt"),
]


def main() -> None:
    # d_model 256: rows are k-quant block multiples, so every format quantizes.
    bot = ai.Chatbot(vocab_size=1024, n_layer=1, d_model=256, seed=1)
    print(f"model: vocab=1024 n_layer=1 d_model=256 ({bot.parameters} params)")
    print(f"{'format':<16}{'size KiB':>10}{'save ms':>10}{'load ms':>10}")
    for format, name in FORMATS:
        path = HERE / name
        t0 = time.perf_counter()
        bot.save(str(path), format=format)
        save_ms = (time.perf_counter() - t0) * 1000
        t0 = time.perf_counter()
        loaded = ai.load(str(path))
        load_ms = (time.perf_counter() - t0) * 1000
        assert loaded.d_model == bot.d_model
        size_kib = os.path.getsize(path) / 1024
        print(f"{format:<16}{size_kib:>10.0f}{save_ms:>10.1f}{load_ms:>10.1f}")
        path.unlink(missing_ok=True)
    print("interop benchmark ok")


if __name__ == "__main__":
    main()
