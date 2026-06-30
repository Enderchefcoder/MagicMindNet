import magicmindnet as ai


def test_merge_classifier_preserves_init_seed_from_first_model():
    a = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=10)
    b = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=20)
    merged = ai.merge_classifier(a, b)
    assert merged.init_seed == 10
