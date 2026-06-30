import magicmindnet as ai


def test_dataset_corpus_invalid_batch_size_falls_back_to_24():
    ds = ai.DatasetCorpus(use_two_files=False, batch_size="not-a-number")
    assert ds.corpus_batch_size == "24"
