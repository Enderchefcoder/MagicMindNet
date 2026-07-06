"""Legacy (pre-1.6, non-zip) torch.save format — genuine CPython pickles."""

import io
import pickle
import struct
import sys
from collections import OrderedDict

import magicmindnet as ai
from conftest import install_fake_torch

LEGACY_MAGIC = 0x1950A86A20F9469CFC6C
PROTOCOL_VERSION = 1001


def write_legacy_torch(path, tensors):
    """Replicate torch's legacy _legacy_save layout with CPython's pickle."""
    torch, _ = install_fake_torch()

    class StorageRef:
        def __init__(self, key, numel):
            self.key = key
            self.numel = numel

    class LegacyPickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, StorageRef):
                return ("storage", torch.FloatStorage, obj.key, "cpu", obj.numel, None)
            return None

    class TensorStub:
        def __init__(self, key, numel, size, stride):
            self.key, self.numel, self.size, self.stride = key, numel, size, stride

        def __reduce__(self):
            return (
                sys.modules["torch._utils"]._rebuild_tensor_v2,
                (
                    StorageRef(self.key, self.numel),
                    0,
                    self.size,
                    self.stride,
                    False,
                    OrderedDict(),
                ),
            )

    def contiguous_stride(shape):
        stride, acc = [], 1
        for dim in reversed(shape):
            stride.insert(0, acc)
            acc *= dim
        return tuple(stride)

    state = OrderedDict()
    numels = []
    for i, (name, (shape, _fill)) in enumerate(tensors.items()):
        numel = 1
        for dim in shape:
            numel *= dim
        numels.append(numel)
        state[name] = TensorStub(str(i), numel, tuple(shape), contiguous_stride(shape))

    buf = io.BytesIO()
    LegacyPickler(buf, protocol=2).dump(LEGACY_MAGIC)
    LegacyPickler(buf, protocol=2).dump(PROTOCOL_VERSION)
    LegacyPickler(buf, protocol=2).dump({"little_endian": True})
    LegacyPickler(buf, protocol=2).dump(state)
    LegacyPickler(buf, protocol=2).dump([str(i) for i in range(len(tensors))])
    for i, (_name, (_shape, fill)) in enumerate(tensors.items()):
        numel = numels[i]
        buf.write(struct.pack("<q", numel))
        buf.write(struct.pack(f"<{numel}f", *([fill] * numel)))
    with open(path, "wb") as fh:
        fh.write(buf.getvalue())


def test_legacy_arrays_read(tmp_path):
    path = str(tmp_path / "legacy.pt")
    write_legacy_torch(path, {"w": ([2, 3], 0.25), "b": ([3], -1.5)})
    arrays = ai.load_pt(path)
    assert arrays["w"] == [[0.25, 0.25, 0.25], [0.25, 0.25, 0.25]]
    assert arrays["b"] == [-1.5, -1.5, -1.5]


def test_legacy_chatbot_universal_load(tmp_path):
    d, vocab = 8, 16
    path = str(tmp_path / "legacy_model.pt")
    write_legacy_torch(
        path,
        {
            "embed": ([vocab, d], 0.5),
            "lm_head": ([vocab, d], 0.5),
            "blocks.0.attn.q": ([d, d], 0.1),
            "blocks.0.attn.k": ([d, d], 0.1),
            "blocks.0.attn.v": ([d, d], 0.1),
            "blocks.0.attn.out": ([d, d], 0.1),
            "blocks.0.ffn": ([d * 4, d], 0.2),
            "blocks.0.ffn2": ([d, d * 4], 0.3),
        },
    )
    loaded = ai.load(path)
    assert isinstance(loaded, ai.Chatbot)
    assert loaded.vocab_size == vocab
    assert loaded.n_layer == 1


def test_legacy_and_zip_formats_agree(tmp_path):
    """The same tensors through legacy and zip formats decode identically."""
    legacy = str(tmp_path / "a.pt")
    write_legacy_torch(legacy, {"x": ([2, 2], 1.25)})
    zipped = str(tmp_path / "b.pt")
    ai.save_pt(zipped, {"x": [[1.25, 1.25], [1.25, 1.25]]})
    assert ai.load_pt(legacy) == ai.load_pt(zipped)
