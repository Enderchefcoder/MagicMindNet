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

# numpy is an optional dependency: only the load_arrays(numpy=True) fast
# path needs it, so the import failure is deferred to that call.
try:
    import numpy as _np
except ImportError:  # pragma: no cover - depends on the environment
    _np = None

__all__ = [
    "detect_arrays_format",
    "gguf_info",
    "load_arrays",
    "load_flax",
    "load_ggml_legacy",
    "load_gguf_arrays",
    "load_gguf_bpe_tokenizer",
    "load_gguf_tokenizer",
    "load_h5",
    "load_keras",
    "load_npy",
    "load_npz",
    "load_onnx",
    "load_pickle_arrays",
    "load_pt",
    "load_safetensors",
    "load_tf_checkpoint",
    "load_tflite",
    "save_arrays",
    "save_flax",
    "save_gguf_arrays",
    "save_h5",
    "save_npy",
    "save_npz",
    "save_onnx",
    "save_pickle_arrays",
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


def load_flax(path):
    """Read a Flax/JAX msgpack checkpoint into ``{"a/b/c": nested lists}``.

    Parses ``flax.serialization.to_bytes`` output from scratch (msgpack map
    tree with ExtType ndarray leaves) — no msgpack, JAX, or Flax install
    needed. All numpy dtypes plus JAX ``bfloat16`` decode to floats.
    """
    return {name: _nest(shape, flat) for name, shape, flat in _native.read_flax(path)}


def save_flax(path, arrays):
    """Write named arrays as a Flax msgpack pytree.

    ``/`` in names creates nested dicts (e.g. ``"params/dense/kernel"``);
    ``flax.serialization.from_bytes`` reads the output.
    """
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_flax(path, packed)


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


def load_pickle_arrays(path):
    """Collect every numpy array in a pickle file into ``{path: nested lists}``.

    Works on PaddlePaddle ``.pdparams`` state dicts, scikit-learn model
    pickles (``coef_``/``intercept_`` fields surface under their attribute
    paths), and plain pickled dicts/lists of ndarrays — no numpy install
    needed. Names are dotted object-graph paths; numpy scalars come back as
    0-d values. Fortran-ordered arrays convert to C order.
    """
    return {
        name: _nest(shape, flat)
        for name, shape, flat in _native.read_pickle_arrays(path)
    }


def save_pickle_arrays(path, arrays):
    """Write named arrays as a pickled ``{name: ndarray}`` dict.

    The output deserializes with plain ``pickle.load`` on any machine with
    numpy installed (F32 C-order arrays).
    """
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_pickle_arrays(path, packed)


def load_tflite(path):
    """Read every weight in a ``.tflite`` model into ``{name: nested lists}``.

    From-scratch flatbuffer parsing — no TensorFlow install needed. INT8/
    UINT8/INT32 tensors with quantization parameters dequantize to
    ``scale * (q - zero_point)`` (per-tensor or per-channel); float16,
    bfloat16, and all int widths decode to floats. Activation tensors
    (empty buffers) are skipped.
    """
    return {name: _nest(shape, flat) for name, shape, flat in _native.read_tflite(path)}


def load_ggml_legacy(path):
    """Read a legacy pre-GGUF llama.cpp file (GGML/GGMF/GGJT v1-v3).

    Returns ``{"container": ..., "hparams": {...}, "vocab": [(bytes, score)],
    "tensors": {name: nested lists}}``. Original f32-scale Q4_0/Q4_1 block
    layouts (pre-GGJT-v2) and the modern layouts (GGJT v2/v3) both decode.
    """
    container, hparams, vocab, arrays = _native.read_ggml_legacy(path)
    names = ["n_vocab", "n_embd", "n_mult", "n_head", "n_layer", "n_rot", "ftype"]
    return {
        "container": container,
        "hparams": dict(zip(names, hparams, strict=True)),
        "vocab": [(bytes(token), score) for token, score in vocab],
        "tensors": {name: _nest(shape, flat) for name, shape, flat in arrays},
    }


def load_gguf_arrays(path):
    """Dequantize every tensor in a GGUF file into ``{name: nested lists}``.

    All GGML tensor types (F32/F16/BF16, classic quants, k-quants, IQ family,
    ternary, MXFP4/NVFP4) decode to floats — no llama.cpp or gguf install
    needed. For header-only inspection use :func:`gguf_info`.
    """
    return {
        name: _nest(shape, flat) for name, shape, flat in _native.read_gguf_arrays(path)
    }


def save_gguf_arrays(path, arrays):
    """Write named arrays as a GGUF file (F32 tensors, gguf-py readable)."""
    packed = []
    for name, array in arrays.items():
        shape, flat = _flatten(array)
        packed.append((str(name), shape, flat))
    _native.write_gguf_arrays(path, packed)


def load_arrays(path, numpy=False):
    """Load *any* supported weights file into ``{name: nested lists}``.

    Detects the container by content: GGUF (v1-v3) and legacy GGML/GGJT,
    PyTorch ``.pt`` (zip + legacy + TorchScript), ``.npy``/``.npz``,
    HDF5/Keras, TFLite, safetensors, Flax msgpack, ONNX, generic numpy
    pickles, and TF checkpoint v2 prefixes. Single-array ``.npy`` files come
    back under the name ``"arr"``. Returns the same shape of dict as the
    per-format ``load_*`` helpers; use :func:`detect_arrays_format` for the
    format tag.

    ``numpy=True`` returns float32 ``numpy.ndarray`` values built straight
    from the decoded bytes (``np.frombuffer``) — roughly an order of
    magnitude faster than nested lists for multi-megabyte models. Requires
    numpy to be installed.
    """
    if numpy:
        if _np is None:
            raise ValueError(
                "load_arrays(numpy=True) requires numpy to be installed"
            )
        _, arrays = _native.read_arrays_auto_bytes(path)
        return {
            name: _np.frombuffer(buf, dtype="<f4").reshape(shape).copy()
            for name, shape, buf in arrays
        }
    _, arrays = _native.read_arrays_auto(path)
    return {name: _nest(shape, flat) for name, shape, flat in arrays}


def detect_arrays_format(path):
    """Name the container ``load_arrays`` would use for *path*.

    One of ``"gguf"``, ``"pt"``, ``"pt-legacy"``, ``"npz"``, ``"npy"``,
    ``"h5"``, ``"keras"``, ``"safetensors"``, ``"flax"``, ``"onnx"``,
    ``"tf-checkpoint"``.
    """
    format_name, _ = _native.read_arrays_auto(path)
    return format_name


def save_arrays(path, arrays, format=None):
    """Write named arrays to any supported container.

    The format comes from the extension (``.npz``, ``.pt``/``.pth``, ``.h5``/
    ``.hdf5``, ``.onnx``, ``.safetensors``, ``.msgpack``, ``.gguf``, ``.pkl``/
    ``.pickle``/``.pdparams``) or an explicit ``format=`` (also accepts
    ``"tf-checkpoint"``, ``"flax"``, and ``"pickle"``).
    """
    inferred = format
    if inferred is None:
        lower = str(path).lower()
        for suffix, name in (
            (".npz", "npz"),
            (".pt", "pt"),
            (".pth", "pt"),
            (".h5", "h5"),
            (".hdf5", "h5"),
            (".onnx", "onnx"),
            (".safetensors", "safetensors"),
            (".msgpack", "flax"),
            (".gguf", "gguf"),
            (".pkl", "pickle"),
            (".pickle", "pickle"),
            (".pdparams", "pickle"),
        ):
            if lower.endswith(suffix):
                inferred = name
                break
    savers = {
        "npz": save_npz,
        "pt": save_pt,
        "h5": save_h5,
        "onnx": save_onnx,
        "safetensors": save_safetensors,
        "flax": save_flax,
        "gguf": save_gguf_arrays,
        "tf-checkpoint": save_tf_checkpoint,
        "pickle": save_pickle_arrays,
    }
    if inferred not in savers:
        raise ValueError(
            f"save_arrays cannot infer a format for {path!r}; pass format= one of "
            + ", ".join(sorted(savers))
        )
    savers[inferred](path, arrays)


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
