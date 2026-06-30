"""GQA + RoPE Chatbot: train and benchmark KV-cache generation."""

import time

import magicmindnet as ai


def bench_generate(bot: ai.Chatbot, prompt: str, *, use_kv_cache: bool, n_tokens: int) -> float:
    t0 = time.perf_counter()
    bot.generate_tokens(
        prompt,
        max_new_tokens=n_tokens,
        temperature=0.0,
        use_kv_cache=use_kv_cache,
    )
    return time.perf_counter() - t0


def main() -> None:
    data = ai.DatasetQA(
        file="tests/fixtures/qa_valid.json",
        user_row="input",
        ai_row="output",
    )
    bot = ai.Chatbot(
        vocab_size=256,
        n_layer=2,
        d_model=64,
        n_heads=4,
        n_kv_heads=2,
        use_rope=True,
        seed=12,
    )
    ai.Train(bot, data, ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05))
    prompt = "What is"
    n_tokens = 24
    t_full = bench_generate(bot, prompt, use_kv_cache=False, n_tokens=n_tokens)
    t_kv = bench_generate(bot, prompt, use_kv_cache=True, n_tokens=n_tokens)
    ids_full = bot.generate_tokens(
        prompt, max_new_tokens=n_tokens, temperature=0.0, use_kv_cache=False
    )
    ids_kv = bot.generate_tokens(
        prompt, max_new_tokens=n_tokens, temperature=0.0, use_kv_cache=True
    )
    print(f"GQA+RoPE n_heads={bot.n_heads} n_kv_heads={bot.n_kv_heads}")
    print(f"generate {n_tokens} tokens: full={t_full:.4f}s kv_cache={t_kv:.4f}s")
    print(f"parity: {ids_full == ids_kv}")
    print(f"tokens (kv): {ids_kv}")


if __name__ == "__main__":
    main()
