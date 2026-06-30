import json
from pathlib import Path

import magicmindnet as ai


def test_dataset_corpus_repr_shows_rows(tmp_path: Path):
    rowfile = tmp_path / "corp.json"
    rowfile.write_text(
        json.dumps([{"text": "short"}, {"text": "longer text here"}]),
        encoding="utf-8",
    )
    txtfile = tmp_path / "corpus.txt"
    txtfile.write_text("hello world", encoding="utf-8")
    ds = ai.DatasetCorpus(
        use_two_files=True,
        rowfile=str(rowfile),
        txtfile=str(txtfile),
    )
    text = repr(ds)
    assert "DatasetCorpus" in text
    assert "rows=" in text
    assert str(ds.rows) in text
