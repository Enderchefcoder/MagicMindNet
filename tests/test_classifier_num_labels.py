import magicmindnet as ai


def test_classifier_num_labels_matches_label_count():
    clf = ai.Classifier.with_labels(["A", "B", "C"], input_dim=16, seed=1)
    assert clf.num_labels == 3
    assert len(clf.labels) == 3
