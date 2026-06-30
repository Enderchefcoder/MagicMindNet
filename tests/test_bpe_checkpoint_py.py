"""BPE checkpoint save/load roundtrip."""

from pathlib import Path

import magicmindnet as ai

FIXTURES = Path(__file__).parent / "fixtures"


def test_byte_pair_encoder_save_load_roundtrip(tmp_path: Path):
    enc = ai.BytePairEncoder.train(
        ["repeat repeat token"] * 10,
        vocab_size=512,
        num_merges=16,
    )
    assert enc.merge_count > 0
    path = tmp_path / "tokenizer.mmn"
    enc.save(str(path))
    loaded = ai.BytePairEncoder.load(str(path))
    assert loaded.merge_count == enc.merge_count
    assert loaded.vocab_size == enc.vocab_size
    sample = "repeat repeat token test"
    assert loaded.encode(sample) == enc.encode(sample)


def test_byte_pair_encoder_train_from_corpus():
    rowfile = FIXTURES / "corpus_rows.json"
    txtfile = FIXTURES / "corpus.txt"
    ds = ai.DatasetCorpus(use_two_files=True, rowfile=str(rowfile), txtfile=str(txtfile))
    enc = ai.BytePairEncoder.train_from_corpus(ds, vocab_size=512, num_merges=8)
    assert enc.vocab_size == 512
    assert enc.encode("corpus text")
