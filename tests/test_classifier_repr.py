import magicmindnet as ai


def test_classifier_repr_lists_labels_and_dims():
    clf = ai.Classifier.with_labels(["Happy", "Sad"], input_dim=32, seed=1)
    text = repr(clf)
    assert "Classifier" in text
    assert "Happy" in text
    assert "input_dim=32" in text
    assert "init_seed=1" in text
