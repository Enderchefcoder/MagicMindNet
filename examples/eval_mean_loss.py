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
BPE_VOCAB = 512


def _usage() -> None:
    print(
        "Usage: python examples/eval_mean_loss.py qa|cls|corpus "
        "[--train] [--learned-pe] [--rope] [--bpe] [--bpe-file PATH]"
    )
    raise SystemExit(2)


def _parse_flags(args: list[str]) -> tuple[set[str], str | None]:
    flags: set[str] = set()
    bpe_file: str | None = None
    skip_next = False
    for i, arg in enumerate(args):
        if skip_next:
            skip_next = False
            continue
        if arg == "--bpe-file":
            if i + 1 >= len(args):
                _usage()
            bpe_file = args[i + 1]
            skip_next = True
            continue
        if arg.startswith("--"):
            flags.add(arg)
    return flags, bpe_file


def _make_chatbot(
    seed: int,
    learned_pe: bool,
    use_rope: bool,
    use_bpe: bool,
) -> ai.Chatbot:
    vocab_size = BPE_VOCAB if use_bpe else 256
    kwargs = dict(vocab_size=vocab_size, n_layer=2, d_model=32, seed=seed)
    if learned_pe:
        return ai.Chatbot(**kwargs, use_learned_pos_embed=True, max_seq_len=128)
    if use_rope:
        return ai.Chatbot(**kwargs, use_rope=True)
    return ai.Chatbot(**kwargs)


def _resolve_bpe(
    mode: str,
    ds: ai.DatasetQA | ai.DatasetCorpus,
    use_bpe: bool,
    bpe_file: str | None,
) -> ai.BytePairEncoder | None:
    if bpe_file:
        return ai.BytePairEncoder.load(bpe_file)
    if not use_bpe:
        return None
    if mode == "qa":
        assert isinstance(ds, ai.DatasetQA)
        enc = ai.BytePairEncoder.train_from_qa(ds, vocab_size=BPE_VOCAB, num_merges=24)
    else:
        assert isinstance(ds, ai.DatasetCorpus)
        enc = ai.BytePairEncoder.train_from_corpus(ds, vocab_size=BPE_VOCAB, num_merges=24)
    if enc.merge_count == 0:
        enc = ai.BytePairEncoder.train(
            ["repeat repeat token"] * 12,
            vocab_size=BPE_VOCAB,
            num_merges=24,
        )
    return enc


def _print_train_delta(label: str, before: float, after: float) -> None:
    print(f"mean {label} loss before: {before:.4f}")
    print(f"mean {label} loss after:  {after:.4f}")


def main() -> None:
    args = sys.argv[1:]
    if not args or args[0] not in MODES:
        _usage()

    mode = args[0]
    flags, bpe_file = _parse_flags(args[1:])
    if "--bpe" in flags and bpe_file:
        print("Use either --bpe or --bpe-file, not both.")
        raise SystemExit(2)

    do_train = "--train" in flags
    learned_pe = "--learned-pe" in flags
    use_rope = "--rope" in flags
    use_bpe = "--bpe" in flags or bpe_file is not None

    if learned_pe and use_rope:
        print("Use either --learned-pe or --rope, not both.")
        raise SystemExit(2)
    if use_rope and mode == "cls":
        print("--rope applies only to qa or corpus chatbot modes.")
        raise SystemExit(2)

    if mode == "qa":
        ds = ai.DatasetQA(
            file=str(FIXTURES / "qa_valid.json"),
            user_row="input",
            ai_row="output",
        )
        bpe = _resolve_bpe(mode, ds, use_bpe, bpe_file)
        bot = _make_chatbot(seed=1, learned_pe=learned_pe, use_rope=use_rope, use_bpe=use_bpe)
        if learned_pe:
            print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
        if use_rope:
            print(f"use_rope: {bot.use_rope} rope_theta: {bot.rope_theta}")
        if bpe is not None:
            print(f"bpe merges: {bpe.merge_count} (vocab_size={bpe.vocab_size})")
        if do_train:
            before = bot.compute_mean_loss(ds, bpe_encoder=bpe)
            ai.Train(bot, ds, TRAIN_CFG, bpe_encoder=bpe)
            after = bot.compute_mean_loss(ds, bpe_encoder=bpe)
            _print_train_delta("QA", before, after)
        else:
            print("mean QA loss:", bot.compute_mean_loss(ds, bpe_encoder=bpe))
        return

    if mode == "corpus":
        ds = ai.DatasetCorpus(
            use_two_files=True,
            rowfile=str(FIXTURES / "corpus_rows.json"),
            txtfile=str(FIXTURES / "corpus.txt"),
        )
        bpe = _resolve_bpe(mode, ds, use_bpe, bpe_file)
        bot = _make_chatbot(seed=3, learned_pe=learned_pe, use_rope=use_rope, use_bpe=use_bpe)
        if learned_pe:
            print(f"use_learned_pos_embed: {bot.use_learned_pos_embed}")
        if use_rope:
            print(f"use_rope: {bot.use_rope} rope_theta: {bot.rope_theta}")
        if bpe is not None:
            print(f"bpe merges: {bpe.merge_count} (vocab_size={bpe.vocab_size})")
        if do_train:
            before = bot.compute_mean_loss(ds, bpe_encoder=bpe)
            ai.Train(bot, ds, TRAIN_CFG, bpe_encoder=bpe)
            after = bot.compute_mean_loss(ds, bpe_encoder=bpe)
            _print_train_delta("corpus", before, after)
        else:
            print("mean corpus loss:", bot.compute_mean_loss(ds, bpe_encoder=bpe))
        return

    if use_bpe or bpe_file:
        print("--bpe and --bpe-file apply only to qa or corpus modes.")
        raise SystemExit(2)

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
