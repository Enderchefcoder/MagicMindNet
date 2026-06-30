import magicmindnet as ai


def test_dataset_corpus_fixed_batch_size_reports_numeric_string():
    ds = ai.DatasetCorpus(use_two_files=False, batch_size="8")
    assert ds.corpus_batch_size == "8"
