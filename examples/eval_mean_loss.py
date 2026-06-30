"""Print mean CE for a Chatbot (QA/corpus) or Classifier (classification) dataset."""

import json
import sys
import tempfile
from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).resolve().parents[1] / "tests" / "fixtures"
MODES = ("qa", "cls", "corpus")
TRAIN_CFG = ai.TrainConfig(epochs=3, batch_size=1, learning_rate=0.05)
CLS_TRAIN_CFG = ai.TrainConfig(epochs=5, batch_size=1, learning_rate=0.08)


def _usage() -> None:
    print("Usage: python examples/eval_mean_loss.py qa|cls|corpus [--train] [--learned-pe]")
    raise SystemExit(2)


def _make_chatbot(seed: int, learned_pe: bool) -> ai.Chatbot:
    kwargs = dict(vocab_size=256, n_layer=2, d_model=32, seed=seed)
    if learned_pe:
        return ai.Chatbot(**kwargs, use_learned_pos_embed=True, max_seq_len=128)
    return ai.Chatbot(**kwargs)


def _print_train_delta(label: str, before: float, after: float) -> None:
    print(f"mean {label} loss before: {before:.4f}")
    print(f"mean {label} loss after:  {after:.4f}")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in MODES:
        _usage()

    mode = args[0]
    flags = set(args[1:])
    do_train = "--train" in flags
    learned_pe = "--learned-pe" in flags

    if mode == "qa":
        ds = ai.DatasetQA(
            file=str(FIXTURES / "qa_valid.json"),
            user_row="input",
            ai_row="output",
        )
        bot = _make_chatbot(seed=1, learned_pe=learned_pe)
        if learned_pe:
            print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
        if do_train:
            before = bot.compute_mean_loss(ds)
            ai.Train(bot, ds, TRAIN_CFG)
            after = bot.compute_mean_loss(ds)
            _print_train_delta("QA", before, after)
        else:
            print("mean QA loss:", bot.compute_mean_loss(ds))
        return

    if mode == "corpus":
        ds = ai.DatasetCorpus(
            use_two_files=True,
            rowfile=str(FIXTURES / "corpus_rows.json"),
            txtfile=str(FIXTURES / "corpus.txt"),
        )
        bot = _make_chatbot(seed=3, learned_pe=learned_pe)
        if learned_pe:
            print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
        if do_train:
            before = bot.compute_mean_loss(ds)
            ai.Train(bot, ds, TRAIN_CFG)
            after = bot.compute_mean_loss(ds)
            _print_train_delta("corpus", before, after)
        else:
            print("mean corpus loss:", bot.compute_mean_loss(ds))
        return

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(
            [
                {"text": "sun", "tag": "Happy"},
                {"text": "rain", "tag": "Sad"},
            ],
            f,
        )
        path = f.name
    ds = ai.DatasetClassification(path, "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=42)
    if do_train:
        before = clf.compute_mean_loss(ds)
        ai.TrainClassifier(clf, ds, CLS_TRAIN_CFG)
        after = clf.compute_mean_loss(ds)
        _print_train_delta("classification", before, after)
    else:
        print("mean classification loss:", clf.compute_mean_loss(ds))


if __name__ == "__main__":
    main()
