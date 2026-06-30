import pytest

import magicmindnet as ai


def test_merge_classifier_rejects_input_dim_mismatch():
    a = ai.Classifier.with_labels(["A", "B"], input_dim=16, seed=1)
    b = ai.Classifier.with_labels(["A", "B"], input_dim=32, seed=2)
    with pytest.raises(ai.ModelMismatchError):
        ai.merge_classifier(a, b)
