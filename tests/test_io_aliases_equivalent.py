import magicmindnet as ai


def test_public_classifier_io_aliases_match_native():
    assert ai.export_classifier is ai.export_classifier_model
    assert ai.import_classifier is ai.import_classifier_model
    assert ai.quantize_classifier is ai.quantize_classifier_model
