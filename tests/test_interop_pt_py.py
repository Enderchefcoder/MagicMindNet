"""PyTorch .pt interchange — from-scratch ZIP + pickle VM, no torch required.

The cross-checks below use CPython's own ``pickle`` and ``zipfile`` modules to
(1) load archives MagicMindNet writes exactly the way ``torch.load`` would and
(2) craft genuine torch-style archives for MagicMindNet to read.
"""

import io
import pickle
import struct
import sys
import types
import zipfile
from collections import OrderedDict

import pytest

import magicmindnet as ai


def make_bot(seed=5):
    return ai.Chatbot(vocab_size=48, n_layer=2, d_model=16, seed=seed)


def _install_fake_torch():
    """Register stand-in torch modules so CPython pickle can resolve globals."""
    if "torch" in sys.modules:
        return sys.modules["torch"], sys.modules["torch._utils"]
    torch = types.ModuleType("torch")
    torch_utils = types.ModuleType("torch._utils")

    class FloatStorage:
        pass

    def _rebuild_tensor_v2(storage, offset, size, stride, requires_grad, hooks):
        return {"storage": storage, "offset": offset, "size": size, "stride": stride}

    FloatStorage.__module__ = "torch"
    FloatStorage.__qualname__ = "FloatStorage"
    _rebuild_tensor_v2.__module__ = "torch._utils"
    _rebuild_tensor_v2.__qualname__ = "_rebuild_tensor_v2"
    torch.FloatStorage = FloatStorage
    torch._utils = torch_utils
    torch_utils._rebuild_tensor_v2 = _rebuild_tensor_v2
    sys.modules["torch"] = torch
    sys.modules["torch._utils"] = torch_utils
    return torch, torch_utils


class _TorchStyleUnpickler(pickle.Unpickler):
    """Mimics torch.load's unpickling: resolves globals + persistent ids."""

    def find_class(self, module, name):
        _install_fake_torch()
        return getattr(sys.modules[module], name)

    def persistent_load(self, pid):
        assert pid[0] == "storage"
        return {"key": pid[2], "numel": pid[4]}


def test_pt_roundtrip_dict(tmp_path):
    path = str(tmp_path / "tensors.pt")
    arrays = {"layer.weight": [[1.0, 2.0], [3.0, 4.0]], "layer.bias": [0.5, -0.5]}
    ai.save_pt(path, arrays)
    assert ai.load_pt(path) == arrays


def test_pt_chatbot_roundtrip(tmp_path):
    path = str(tmp_path / "bot.pt")
    bot = make_bot()
    bot.save(path, format="pt")
    loaded = ai.Chatbot.load(path)
    assert loaded.vocab_size == bot.vocab_size
    assert loaded.n_layer == bot.n_layer
    assert loaded.d_model == bot.d_model


def test_pt_universal_load(tmp_path):
    path = str(tmp_path / "bot.pt")
    make_bot().save(path, format="pt")
    assert isinstance(ai.load(path), ai.Chatbot)


def test_pt_roundtrip_preserves_loss(tmp_path):
    data = ai.DatasetQA(data=[{"input": "hey", "output": "yo"}])
    bot = make_bot(seed=13)
    before = bot.compute_mean_loss(data)
    path = str(tmp_path / "bot.pt")
    bot.save(path, format="pytorch")
    after = ai.load(path).compute_mean_loss(data)
    assert abs(before - after) < 1e-4


def test_pt_archive_is_valid_zip_with_torch_layout(tmp_path):
    path = tmp_path / "bot.pt"
    make_bot().save(str(path), format="pt")
    with zipfile.ZipFile(path) as zf:
        names = zf.namelist()
        assert "archive/data.pkl" in names
        assert "archive/version" in names
        assert any(n.startswith("archive/data/") for n in names)
        # CPython's zipfile CRC-checks every member on read.
        assert zf.testzip() is None


def test_python_pickle_loads_our_data_pkl(tmp_path):
    """CPython's pickle module must parse the stream torch.load would see."""
    path = tmp_path / "arrays.pt"
    ai.save_pt(str(path), {"w": [[1.0, 2.0], [3.0, 4.0]]})
    with zipfile.ZipFile(path) as zf:
        with zf.open("archive/data.pkl") as fh:
            state = _TorchStyleUnpickler(fh).load()
    assert set(state.keys()) == {"w"}
    assert state["w"]["size"] == (2, 2)
    assert state["w"]["stride"] == (2, 1)
    assert state["w"]["storage"]["numel"] == 4


def test_import_genuine_python_pickled_archive(tmp_path):
    """A .pt written by CPython's pickle (torch.save layout) must import."""
    torch, _ = _install_fake_torch()

    class StorageRef:
        def __init__(self, key, numel):
            self.key = key
            self.numel = numel

    class TorchStylePickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, StorageRef):
                return ("storage", torch.FloatStorage, obj.key, "cpu", obj.numel)
            return None

    class TensorStub:
        """Pickles exactly like a torch tensor: _rebuild_tensor_v2 + storage pid."""

        def __init__(self, key, numel, size, stride):
            self.key, self.numel, self.size, self.stride = key, numel, size, stride

        def __reduce__(self):
            return (
                sys.modules["torch._utils"]._rebuild_tensor_v2,
                (StorageRef(self.key, self.numel), 0, self.size, self.stride, False, OrderedDict()),
            )

    def contiguous_stride(shape):
        stride, acc = [], 1
        for dim in reversed(shape):
            stride.insert(0, acc)
            acc *= dim
        return tuple(stride)

    d_model, vocab = 8, 16
    tensors = {
        "embed": ([vocab, d_model], 0.5),
        "lm_head": ([vocab, d_model], 0.5),
        "blocks.0.attn.q": ([d_model, d_model], 0.1),
        "blocks.0.attn.k": ([d_model, d_model], 0.1),
        "blocks.0.attn.v": ([d_model, d_model], 0.1),
        "blocks.0.attn.out": ([d_model, d_model], 0.1),
        "blocks.0.ffn": ([d_model * 4, d_model], 0.2),
        "blocks.0.ffn2": ([d_model, d_model * 4], 0.3),
    }
    state = OrderedDict()
    numels = {}
    for i, (tensor_name, (shape, _fill)) in enumerate(tensors.items()):
        numel = 1
        for dim in shape:
            numel *= dim
        numels[str(i)] = numel
        state[tensor_name] = TensorStub(str(i), numel, tuple(shape), contiguous_stride(shape))

    buf = io.BytesIO()
    TorchStylePickler(buf, protocol=2).dump(state)

    path = tmp_path / "external.pt"
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as zf:
        zf.writestr("archive/data.pkl", buf.getvalue())
        zf.writestr("archive/version", "3\n")
        for i, (_name, (shape, fill)) in enumerate(tensors.items()):
            numel = 1
            for dim in shape:
                numel *= dim
            zf.writestr(f"archive/data/{i}", struct.pack(f"<{numel}f", *([fill] * numel)))

    loaded = ai.load(str(path))
    assert isinstance(loaded, ai.Chatbot)
    assert loaded.vocab_size == vocab
    assert loaded.d_model == d_model
    assert loaded.n_layer == 1


def test_pt_missing_file_errors():
    with pytest.raises((RuntimeError, ValueError), match="cannot read"):
        ai.load_pt("/nonexistent/never.pt")
