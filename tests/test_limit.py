"""Resource limit API."""

import magicmindnet as ai


def test_limit_accepts_valid_percent():
    ai.limit("50%")
    assert ai.limit_percent() == 50


def test_limit_rejects_invalid():
    import pytest

    with pytest.raises(RuntimeError, match="Limit must be"):
        ai.limit("0%")
