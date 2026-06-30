import json
from pathlib import Path

import magicmindnet as ai


def test_dataset_corpus_rows_format_and_type_getters(tmp_path: Path):
    rowfile = tmp_path / "corp.json"
    rowfile.write_text(
        json.dumps([{"text": "one"}, {"text": "two words"}]),
        encoding="utf-8",
    )
    txtfile = tmp_path / "corpus.txt"
    txtfile.write_text("token " * 20, encoding="utf-8")
    ds = ai.DatasetCorpus(
        use_two_files=True,
        rowfile=str(rowfile),
        txtfile=str(txtfile),
    )
    assert ds.rows >= 2
    assert ds.type_ == "corpus"
    assert isinstance(ds.format, str)
    assert len(ds.format) > 0
