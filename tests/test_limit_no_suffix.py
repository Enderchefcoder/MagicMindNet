import magicmindnet as ai


def test_limit_accepts_numeric_string_without_percent_suffix():
    ai.limit("25")
    assert ai.limit_percent() == 25
