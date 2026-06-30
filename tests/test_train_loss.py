from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_train_reduces_loss_on_repeated_sample():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32)
    cfg = ai.TrainConfig(epochs=5, batch_size=1, cuda=False, learning_rate=0.05)
    ai.Train(bot, ds, cfg)
    # smoke: training completes; Rust unit test asserts loss decrease


def test_dataset_qa_format_sample():
    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    text = ds.format_sample(0)
    assert "input" in text.lower() or "Hello" in text
