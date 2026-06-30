import magicmindnet as ai


def test_classifier_predict_probabilities_sum_to_one():
    clf = ai.Classifier.with_labels(["Happy", "Sad", "Neutral"], input_dim=32, seed=1)
    probs = clf.predict("hello world")
    assert set(probs.keys()) == {"Happy", "Sad", "Neutral"}
    total = sum(probs.values())
    assert abs(total - 1.0) < 1e-4
