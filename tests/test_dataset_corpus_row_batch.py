import magicmindnet as ai


def test_dataset_corpus_row_batch_size_reports_row():
    ds = ai.DatasetCorpus(use_two_files=False, batch_size="row")
    assert ds.corpus_batch_size == "row"
