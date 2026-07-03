"""`Classifier.predict_label()` returns the argmax label of `predict()`."""

import magicmindnet as ai


def test_predict_label_matches_argmax_of_predict():
    clf = ai.Classifier.with_labels(["happy", "sad", "angry"], input_dim=16, seed=4)
    probs = clf.predict("sunny weather")
    best = max(probs, key=probs.get)
    assert clf.predict_label("sunny weather") == best


def test_predict_label_after_training_learns_labels():
    ds = ai.DatasetClassification(
        data=[
            {"text": "sunny bright warm", "label": "happy"},
            {"text": "rainy cold dark", "label": "sad"},
        ]
    )
    clf = ai.Classifier.from_classification(ds, input_dim=32, seed=4)
    clf.train(ds, epochs=30, learning_rate=0.05)
    assert clf.predict_label("sunny bright warm") == "happy"
    assert clf.predict_label("rainy cold dark") == "sad"


def test_predict_label_is_a_known_label():
    clf = ai.Classifier(num_labels=2, input_dim=16, seed=1)
    assert clf.predict_label("anything") in clf.labels
