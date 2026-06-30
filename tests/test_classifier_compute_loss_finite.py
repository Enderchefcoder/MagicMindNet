import math

import magicmindnet as ai


def test_classifier_compute_loss_is_finite_positive():
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=1)
    loss = clf.compute_loss("hello", "A")
    assert math.isfinite(loss)
    assert loss > 0.0
