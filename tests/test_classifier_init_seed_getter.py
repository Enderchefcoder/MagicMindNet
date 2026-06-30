import magicmindnet as ai


def test_classifier_init_seed_getter_matches_constructor():
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=12)
    assert clf.init_seed == 12


def test_classifier_without_seed_has_none_init_seed():
    clf = ai.Classifier(2, input_dim=16)
    assert clf.init_seed is None
