"""Train a Chatbot with unigram tokenization and sample a reply."""

import magicmindnet as ai


def main() -> None:
    data = ai.DatasetQA(
        file="tests/fixtures/qa_valid.json",
        user_row="input",
        ai_row="output",
    )
    enc = ai.UnigramEncoder.train_from_qa(data, vocab_size=512)
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=32, seed=11)
    before = bot.compute_mean_loss(data, unigram_encoder=enc)
    ai.Train(
        bot,
        data,
        ai.TrainConfig(epochs=2, batch_size=1, learning_rate=0.05),
        unigram_encoder=enc,
    )
    after = bot.compute_mean_loss(data, unigram_encoder=enc)
    prompt = "What is"
    reply = bot.generate(
        prompt,
        max_new_tokens=16,
        temperature=0.0,
        unigram_encoder=enc,
    )
    print(f"mean loss: {before:.4f} -> {after:.4f}")
    print(f"prompt: {prompt!r}")
    print(f"reply:  {ascii(reply)}")


if __name__ == "__main__":
    main()
