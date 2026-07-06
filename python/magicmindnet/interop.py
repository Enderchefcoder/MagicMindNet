"""Global array/tensor file interop: NumPy ``.npy``/``.npz`` and PyTorch ``.pt``.

Every codec is implemented from scratch in the Rust core — ``numpy`` and
``torch`` do **not** need to be installed. When they are, their arrays and
tensors are accepted directly (anything with ``.tolist()`` works)::

    import magicmindnet as ai

    ai.save_npy("x.npy", [[1.0, 2.0], [3.0, 4.0]])
    x = ai.load_npy("x.npy")            # nested lists

    ai.save_npz("weights.npz", {"w": [[1.0]], "b": [0.5]})
    arrays = ai.load_npz("weights.npz")  # {"w": [[1.0]], "b": [0.5]}

    ai.save_pt("model.pt", {"w": [[1.0]]})  # torch.load-compatible
    tensors = ai.load_pt("model.pt")
"""

from magicmindnet import _native

__all__ = [
    "load_npy",
    "load_npz",
    "load_pt",
    "save_npy",
    "save_npz",
    "save_pt",
]


def _flatten(array):
    """Turn nested lists / numpy arrays / torch tensors into (shape, flat f32)."""
    if hasattr(array, "detach"):  # torch tensor
        array = array.detach()
    if hasattr(array, "tolist"):  # numpy array / torch tensor
        array = array.tolist()
    if isinstance(array, (int, float)):
        return [], [float(array)]
    shape = []
    probe = array
    while isinstance(probe, (list, tuple)):
        shape.append(len(probe))
        if not probe:
            break
        probe = probe[0]
    flat: list[float] = []

    def walk(node, depth):
        if depth == len(shape):
            flat.append(float(node))
            return
        if not isinstance(node, (list, tuple)) or len(node) != shape[depth]:
            raise ValueError("array is ragged: nested lists must be rectangular")
        for item in node:
            walk(item, depth + 1)

    walk(array, 0)
    return shape, flat


def _nest(shape, flat):
    """Rebuild nested lists from a shape and flat data."""
    if not shape:
        return flat[0]

    def build(dims, offset):
        if len(dims) == 1:
            return list(flat[offset : offset + dims[0]]), offset + dims[0]
        out = []
        for _ in range(dims[0]):
            sub, offset = build(dims[1:], offset)
            out.append(sub)
        return out, offset

    nested, _ = build(list(shape), 0)
    return nested


def save_npy(path, array):
    """Write one array (nested lists, numpy, or torch) as a ``.npy`` file."""
    shape, flat = _flatten(array)
    _native.write_npy(path, shape, flat)


def load_npy(path):
    """Read a ``.npy`` file into nested Python lists."""
    shape, flat = _native.read_npy(path)
    return _nest(shape, flat)


def save_npz(path, arrays):
    """Write a dict of named arrays as an ``.npz`` archive (``numpy.load`` compatible)."""
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_npz(path, packed)


def load_npz(path):
    """Read an ``.npz`` archive into ``{name: nested lists}``."""
    return {name: _nest(shape, flat) for name, shape, flat in _native.read_npz(path)}


def save_pt(path, arrays):
    """Write a dict of named arrays as a ``torch.load``-compatible ``.pt`` state dict."""
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_pt(path, packed)


def load_pt(path):
    """Read a PyTorch ``.pt`` state dict into ``{name: nested lists}`` (no torch needed)."""
    return {name: _nest(shape, flat) for name, shape, flat in _native.read_pt(path)}
