"""Global array/tensor file interop: NumPy, PyTorch, TensorFlow/Keras, GGUF.

Every codec is implemented from scratch in the Rust core — ``numpy``,
``torch``, ``tensorflow``, and ``h5py`` do **not** need to be installed. When
they are, their arrays and tensors are accepted directly (anything with
``.tolist()`` works)::

    import magicmindnet as ai

    ai.save_npy("x.npy", [[1.0, 2.0], [3.0, 4.0]])
    x = ai.load_npy("x.npy")            # nested lists

    ai.save_npz("weights.npz", {"w": [[1.0]], "b": [0.5]})
    arrays = ai.load_npz("weights.npz")  # {"w": [[1.0]], "b": [0.5]}

    ai.save_pt("model.pt", {"w": [[1.0]]})  # torch.load-compatible
    tensors = ai.load_pt("model.pt")        # zip or legacy pre-1.6 format

    weights = ai.load_h5("model.h5")        # HDF5 / Keras weights
    weights = ai.load_keras("model.keras")  # Keras v3 archive

    info = ai.gguf_info("model.gguf")       # metadata + tensor summaries
    tok = ai.load_gguf_tokenizer("model.gguf")  # embedded SentencePiece vocab
"""

import json

from magicmindnet import _native

__all__ = [
    "gguf_info",
    "load_gguf_bpe_tokenizer",
    "load_gguf_tokenizer",
    "load_h5",
    "load_keras",
    "load_npy",
    "load_npz",
    "load_onnx",
    "load_pt",
    "load_safetensors",
    "load_tf_checkpoint",
    "save_h5",
    "save_npy",
    "save_npz",
    "save_onnx",
    "save_pt",
    "save_safetensors",
    "save_tf_checkpoint",
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


def save_npz(path, arrays, compress=False):
    """Write a dict of named arrays as an ``.npz`` archive (``numpy.load`` compatible).

    ``compress=True`` DEFLATE-compresses entries like ``np.savez_compressed``
    (from-scratch compressor, no zlib).
    """
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_npz(path, packed, compress)


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


def load_h5(path):
    """Read every dataset in an HDF5 file into ``{path: nested lists}`` (no h5py needed)."""
    return {name: _nest(shape, flat) for name, shape, flat in _native.read_h5(path)}


def load_keras(path):
    """Read Keras weights (``.keras`` archive, ``.weights.h5``, or ``.h5``)."""
    return {name: _nest(shape, flat) for name, shape, flat in _native.read_keras(path)}


def load_onnx(path):
    """Read every graph initializer (weight) in an ``.onnx`` model.

    From-scratch protobuf parsing — no onnx/protobuf install needed. Returns
    ``{name: nested lists}``.
    """
    return {name: _nest(shape, flat) for name, shape, flat in _native.read_onnx(path)}


def save_onnx(path, arrays):
    """Write named arrays as an ``.onnx`` model (``onnx.load``-compatible)."""
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_onnx(path, packed)


def load_safetensors(path):
    """Read every tensor in a ``.safetensors`` file into ``{name: nested lists}``.

    All dtypes in the safetensors spec (BOOL through F64/I64/U64) decode to
    floats — no ``safetensors`` package needed.
    """
    return {
        name: _nest(shape, flat) for name, shape, flat in _native.read_safetensors(path)
    }


def save_safetensors(path, arrays):
    """Write named arrays as a ``.safetensors`` file (F32 tensors)."""
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_safetensors(path, packed)


def save_h5(path, arrays):
    """Write named arrays as an HDF5 file (h5py/Keras-readable, no h5py needed).

    Names with ``/`` create nested groups (e.g. ``"dense/kernel"``).
    """
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_h5(path, packed)


def save_tf_checkpoint(path, arrays):
    """Write a TF checkpoint v2 (``tf.train.load_checkpoint``-compatible).

    Emits ``{path}.index`` and ``{path}.data-00000-of-00001`` from scratch.
    """
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_tf_checkpoint(path, packed)


def load_tf_checkpoint(path):
    """Read a TensorFlow checkpoint v2 (``ckpt`` prefix or ``ckpt.index`` path).

    Parses the LevelDB-table index and raw data shards from scratch — no
    TensorFlow installation needed. Returns ``{name: nested lists}`` with
    ``/.ATTRIBUTES/VARIABLE_VALUE`` suffixes stripped.
    """
    return {
        name: _nest(shape, flat) for name, shape, flat in _native.read_tf_checkpoint(path)
    }


def gguf_info(path):
    """Inspect a GGUF file: version, metadata dict, and tensor summaries.

    Header-only read — multi-gigabyte models are not loaded into memory.
    """
    return json.loads(_native.gguf_info_json(path))


def load_gguf_tokenizer(path):
    """Extract the SentencePiece vocabulary embedded in a GGUF model.

    Returns a :class:`magicmindnet.UnigramEncoder` whose token ids match the
    model's rows, so ``encode``/``decode`` line up with the GGUF weights.
    """
    return _native.load_gguf_tokenizer(path)


def load_gguf_bpe_tokenizer(path):
    """Extract a byte-level BPE ("gpt2") vocabulary embedded in a GGUF model.

    Returns a :class:`magicmindnet.Gpt2BpeEncoder` whose token ids match the
    model's rows (GPT-2 / Llama-3 / Qwen-style vocabularies).
    """
    return _native.load_gguf_bpe_tokenizer(path)
