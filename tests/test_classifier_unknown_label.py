import pytest

import magicmindnet as ai


def test_compute_loss_unknown_label_raises():
    clf = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=1)
    with pytest.raises(ValueError, match="unknown label"):
        clf.compute_loss("hello", "missing")
