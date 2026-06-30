"""Public exception types are importable Exception subclasses."""

import pytest

import magicmindnet as ai


@pytest.mark.parametrize(
    "name",
    [
        "CPUError",
        "CUDAError",
        "DataMismatchError",
        "DataMissingRowError",
        "ModelMismatchError",
    ],
)
def test_public_exceptions_are_exception_subclasses(name: str):
    cls = getattr(ai, name)
    assert issubclass(cls, Exception)
    err = cls("test message")
    assert "test message" in str(err)
