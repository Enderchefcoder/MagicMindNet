import magicmindnet as ai


def test_classifier_num_labels_constructor_sets_label_count():
    clf = ai.Classifier(4, input_dim=16, seed=1)
    assert clf.num_labels == 4
    assert clf.labels == ["class_0", "class_1", "class_2", "class_3"]
