import magicmindnet as ai


def test_merge_classifier_preserves_labels_input_dim_and_num_labels():
    a = ai.Classifier.with_labels(["X", "Y", "Z"], input_dim=24, seed=1)
    b = ai.Classifier.with_labels(["X", "Y", "Z"], input_dim=24, seed=2)
    merged = ai.merge_classifier(a, b)
    assert merged.labels == ["X", "Y", "Z"]
    assert merged.input_dim == 24
    assert merged.num_labels == 3
