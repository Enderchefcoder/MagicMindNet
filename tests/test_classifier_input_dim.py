import magicmindnet as ai


def test_classifier_exposes_input_dim_getter():
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=24, seed=1)
    assert clf.input_dim == 24
