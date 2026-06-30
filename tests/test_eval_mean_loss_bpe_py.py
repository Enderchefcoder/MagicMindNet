"""eval_mean_loss.py --bpe and --bpe-file integration."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_eval_mean_loss_bpe_file_roundtrip(tmp_path: Path):
    enc = ai.BytePairEncoder.train(
        ["repeat repeat token"] * 10,
        vocab_size=512,
        num_merges=16,
    )
    tok = tmp_path / "tok.mmn"
    enc.save(str(tok))

    path = FIXTURES / "qa_valid.json"
    ds = ai.DatasetQA(file=str(path), user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=512, n_layer=1, d_model=32, seed=1)
    loaded = ai.BytePairEncoder.load(str(tok))
    loss = bot.compute_mean_loss(ds, bpe_encoder=loaded)
    assert loss > 0
    assert loaded.encode("repeat") == enc.encode("repeat")
