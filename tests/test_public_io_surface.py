"""Public IO and utility aliases exported from magicmindnet."""

import magicmindnet as ai


def test_public_io_and_merge_aliases_are_callable():
    assert callable(ai.export)
    assert callable(ai.import_model)
    assert callable(ai.quantize)
    assert callable(ai.import_classifier)
    assert callable(ai.quantize_classifier)
    assert callable(ai.merge)
    assert callable(ai.merge_classifier)
    assert callable(ai.limit)
    assert callable(ai.limit_percent)
