"""Your first MagicMindNet program — train a chatbot and a classifier in ~30 lines.

Run from the repo root after building (`maturin develop --release`):

    python examples/hello_ai.py
"""

import tempfile
from pathlib import Path

import magicmindnet as ai


def main() -> None:
    # 1. A chatbot: give it a few example conversations (no files needed).
    data = ai.DatasetQA(
        data=[
            {"input": "hi", "output": "hello there!"},
            {"input": "how are you?", "output": "doing great, thanks!"},
            {"input": "bye", "output": "see you soon!"},
        ]
    )
    bot = ai.Chatbot(vocab_size=512, n_layer=2, d_model=64, seed=42)
    print(f"Chatbot with {bot.parameters:,} parameters")

    losses = bot.train(data, epochs=5, verbose=True)
    print(f"Loss went from {losses[0]:.3f} to {losses[-1]:.3f}")

    print("Reply:", repr(bot.chat("hi", max_new_tokens=16)))

    # 2. Save it, load it back with one call.
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "my_first_bot.mmn"
        bot.save(str(path))
        bot2 = ai.load(str(path))
        print(f"Reloaded: {bot2}")

    # 3. A text classifier in four lines.
    labels = ai.DatasetClassification(
        data=[
            {"text": "sunny warm bright", "label": "nice"},
            {"text": "rainy cold windy", "label": "gloomy"},
        ]
    )
    clf = ai.Classifier.from_classification(labels, input_dim=64, seed=42)
    clf.train(labels, epochs=20)
    print("'sunny warm bright' is:", clf.predict_label("sunny warm bright"))


if __name__ == "__main__":
    main()
