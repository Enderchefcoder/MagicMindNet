"""Every name in magicmindnet.__all__ is defined on the package."""

import magicmindnet as ai


def test_all_exports_are_defined_on_package():
    missing = [name for name in ai.__all__ if not hasattr(ai, name)]
    assert missing == []
