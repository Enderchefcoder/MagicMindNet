"""Generic pickle array IO (ai.load_pickle_arrays / ai.save_pickle_arrays).

Covers PaddlePaddle-style ``.pdparams`` state dicts, sklearn-style objects
(arrays inside ``__dict__`` state), numpy scalars, Fortran order, and every
pickle protocol 2-5 — cross-validated against CPython pickle + numpy.
"""

import pickle

import numpy as np
import pytest

import magicmindnet as ai


class FakeEstimator:
    """sklearn-style object: arrays live in instance __dict__ state."""

    def __init__(self):
        self.coef_ = np.array([[1.5, -2.5, 0.5]], dtype=np.float64)
        self.intercept_ = np.array([0.25], dtype=np.float64)
        self.n_features_in_ = 3


@pytest.mark.parametrize("protocol", [2, 3, 4, 5])
def test_state_dict_reads_every_protocol(tmp_path, protocol):
    path = str(tmp_path / f"state_p{protocol}.pdparams")
    state = {
        "linear.weight": np.arange(6, dtype=np.float32).reshape(2, 3),
        "linear.bias": np.array([0.5, -0.5], dtype=np.float64),
        "global_step": np.float32(7.0),
    }
    with open(path, "wb") as f:
        pickle.dump(state, f, protocol=protocol)
    loaded = ai.load_pickle_arrays(path)
    assert loaded["linear.weight"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]
    assert loaded["linear.bias"] == [0.5, -0.5]
    assert loaded["global_step"] == 7.0


@pytest.mark.parametrize(
    "dtype",
    [np.float16, np.float32, np.float64, np.int8, np.int16, np.int32, np.int64,
     np.uint8, np.uint16, np.uint32, np.uint64, np.bool_],
)
def test_every_dtype_decodes(tmp_path, dtype):
    path = str(tmp_path / f"dt_{np.dtype(dtype).name}.pkl")
    arr = np.array([0, 1], dtype=dtype)
    with open(path, "wb") as f:
        pickle.dump({"x": arr}, f)
    loaded = ai.load_pickle_arrays(path)
    np.testing.assert_allclose(loaded["x"], arr.astype(np.float32))


def test_big_endian_and_fortran(tmp_path):
    path = str(tmp_path / "be.pkl")
    be = np.array([1.5, -2.0], dtype=">f4")
    fortran = np.asfortranarray(np.arange(6, dtype=np.float32).reshape(2, 3))
    with open(path, "wb") as f:
        pickle.dump({"be": be, "f": fortran}, f)
    loaded = ai.load_pickle_arrays(path)
    assert loaded["be"] == [1.5, -2.0]
    assert loaded["f"] == [[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]]


def test_sklearn_style_object_arrays_surface(tmp_path):
    path = str(tmp_path / "model.pkl")
    with open(path, "wb") as f:
        pickle.dump(FakeEstimator(), f)
    loaded = ai.load_pickle_arrays(path)
    assert loaded["coef_"] == [[1.5, -2.5, 0.5]]
    assert loaded["intercept_"] == [0.25]


def test_nested_lists_and_dicts_get_dotted_paths(tmp_path):
    path = str(tmp_path / "nested.pkl")
    tree = {"layers": [{"w": np.ones((2,), dtype=np.float32)},
                       {"w": np.zeros((2,), dtype=np.float32)}]}
    with open(path, "wb") as f:
        pickle.dump(tree, f)
    loaded = ai.load_pickle_arrays(path)
    assert loaded["layers.0.w"] == [1.0, 1.0]
    assert loaded["layers.1.w"] == [0.0, 0.0]


def test_save_pickle_arrays_loads_with_real_pickle(tmp_path):
    path = str(tmp_path / "ours.pkl")
    ai.save_pickle_arrays(path, {"w": [[1.0, -2.0], [3.5, 0.25]], "b": [9.0]})
    with open(path, "rb") as f:
        back = pickle.load(f)
    assert isinstance(back["w"], np.ndarray)
    assert back["w"].dtype == np.float32
    np.testing.assert_allclose(back["w"], [[1.0, -2.0], [3.5, 0.25]])
    np.testing.assert_allclose(back["b"], [9.0])


def test_load_arrays_detects_pickle(tmp_path):
    path = str(tmp_path / "state.pdparams")
    with open(path, "wb") as f:
        pickle.dump({"w": np.array([1.0], dtype=np.float32)}, f)
    assert ai.detect_arrays_format(path) == "pickle"
    assert ai.load_arrays(path) == {"w": [1.0]}


def test_save_arrays_pdparams_extension(tmp_path):
    path = str(tmp_path / "out.pdparams")
    ai.save_arrays(path, {"w": [[2.0]]})
    with open(path, "rb") as f:
        back = pickle.load(f)
    np.testing.assert_allclose(back["w"], [[2.0]])


def test_pickle_without_arrays_errors(tmp_path):
    path = str(tmp_path / "noarrays.pkl")
    with open(path, "wb") as f:
        pickle.dump({"config": "text", "epochs": 3}, f)
    with pytest.raises((ValueError, RuntimeError), match="no numpy arrays"):
        ai.load_pickle_arrays(path)


def test_object_array_leaf_rejected(tmp_path):
    path = str(tmp_path / "obj.pkl")
    arr = np.array([{"a": 1}, None], dtype=object)
    with open(path, "wb") as f:
        pickle.dump({"o": arr}, f)
    with pytest.raises((ValueError, RuntimeError)):
        ai.load_pickle_arrays(path)
