"""DistribAI checkpoint interop — Chatbot ⇄ DistribAI ``state_dict`` files.

``export_checkpoint`` writes the exact tensor names
``DistribAIModelWrapper`` produces (``model.embedding.weight``,
``model.layers.N.self_attn.in_proj_weight`` fused Q/K/V or the
``_GroupedQueryAttnBlock`` ``q_proj``/``k_proj``/... flavour) as ``.pt`` or
``.safetensors``; ``import_checkpoint`` reads flat state dicts, safetensors,
and the nested ``torch.save({"model_state": ...})`` wrapper their trainer
saves. Neither direction needs torch installed — see
:mod:`magicmindnet.distribai` for the full bridge.
"""

import json
import math
import os
import struct
import tempfile

from magicmindnet import _native
from magicmindnet.interop import load_arrays, save_pt, save_safetensors

__all__ = [
    "export_checkpoint",
    "import_checkpoint",
    "sinusoidal_position_encoding",
]



def sinusoidal_position_encoding(seq_len, d_model):
    """The fixed PE table MagicMindNet uses, as nested lists.

    Parity with the Rust core (``mmn_nn::sinusoidal_position_encoding``):
    exported so DistribAI's required ``position_embedding.weight`` can be
    synthesized when a Chatbot uses sinusoidal (non-learned) positions.
    """
    rows = []
    for pos in range(seq_len):
        row = []
        for i in range(d_model):
            div = 10000.0 ** (2.0 * (i // 2) / d_model)
            angle = pos / div
            row.append(math.sin(angle) if i % 2 == 0 else math.cos(angle))
        rows.append(row)
    return rows


def _bot_state(bot):
    """Extract ``(meta, {name: (shape, flat float list)})`` from a Chatbot."""
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "bot.mmn")
        bot.save(path)
        with open(path, encoding="utf-8") as handle:
            checkpoint = json.load(handle)
    tensors = {}
    for name, entry in checkpoint["tensors"].items():
        raw = bytes(entry["data"])
        flat = list(struct.unpack(f"<{len(raw) // 4}f", raw))
        tensors[name] = (list(entry["shape"]), flat)
    return checkpoint.get("meta", {}), tensors


def _reshape(shape, flat):
    if not shape:
        return flat[0]
    if len(shape) == 1:
        return list(flat)
    size = len(flat) // shape[0]
    return [_reshape(shape[1:], flat[index * size : (index + 1) * size]) for index in range(shape[0])]


def _flat(nested):
    if not isinstance(nested, list):
        return [float(nested)]
    out = []
    stack = [iter(nested)]
    while stack:
        try:
            item = next(stack[-1])
        except StopIteration:
            stack.pop()
            continue
        if isinstance(item, list):
            stack.append(iter(item))
        else:
            out.append(float(item))
    return out


def _shape_of(nested):
    shape = []
    probe = nested
    while isinstance(probe, list):
        shape.append(len(probe))
        probe = probe[0] if probe else None
    return shape


def export_checkpoint(bot, path, *, prefix="model."):
    """Write a Chatbot as a DistribAI-native ``state_dict`` checkpoint.

    Emits the exact tensor names ``DistribAIModelWrapper`` produces so the
    file loads on a DistribAI node with
    ``wrapper.load_state_dict(torch.load(path))`` (or
    ``safetensors.torch.load_file`` for ``.safetensors`` paths — matching
    DistribAI's own ``export_safetensors``):

    - standard attention (``n_kv_heads == n_heads``) uses torch
      ``TransformerEncoderLayer`` names with Q/K/V fused into
      ``self_attn.in_proj_weight``;
    - grouped-query attention uses DistribAI's ``_GroupedQueryAttnBlock``
      names (``q_proj`` / ``k_proj`` / ``v_proj`` / ``o_proj``, ``ffn.0`` /
      ``ffn.3``);
    - LayerNorm gamma/beta become ``weight`` / ``bias``; missing torch biases
      are written as zeros; sinusoidal positions are materialized into the
      required ``position_embedding.weight`` table.

    ``prefix`` mirrors the wrapper's ``model.`` prefix (set ``prefix=""`` for
    bare inner-module names). Returns the mapped ``{name: nested lists}``
    dict that was written.
    """
    meta, tensors = _bot_state(bot)
    for flag, message in (
        (meta.get("vision"), "vision tensors"),
        (meta.get("ffn_kind") == "swiglu", "SwiGLU gate weights"),
        (meta.get("loop_embed"), "loop_embed tables"),
        (meta.get("lora_rank", 0) > 0, "LoopLoRA adapters"),
        (meta.get("prelude_layers", 0) > 0 or meta.get("coda_layers", 0) > 0, "prelude/coda blocks"),
    ):
        if flag:
            raise ValueError(f"DistribAI checkpoints cannot represent {message}")
    d_model = int(meta["d_model"])
    n_layer = int(meta["n_layer"])
    vocab_size = int(meta["vocab_size"])
    n_heads = int(meta.get("n_heads", 1))
    n_kv_heads = int(meta.get("n_kv_heads", n_heads))
    if meta.get("head_dim") is not None:
        raise ValueError("DistribAI attention requires head_dim == d_model / n_heads")
    grouped = n_kv_heads != n_heads

    def tensor(name):
        shape, flat = tensors[name]
        return _reshape(shape, flat)

    def zeros(*shape):
        total = 1
        for dim in shape:
            total *= dim
        return _reshape(list(shape), [0.0] * total)

    state = {
        "embedding.weight": tensor("embed"),
        "fc_out.weight": tensor("lm_head"),
        "fc_out.bias": zeros(vocab_size),
    }
    if meta.get("use_learned_pos_embed"):
        state["position_embedding.weight"] = tensor("pos_embed")
    else:
        # Checkpoint meta only records max_seq_len for learned positions;
        # the live model always knows its context length.
        state["position_embedding.weight"] = sinusoidal_position_encoding(bot.max_seq_len, d_model)
    if "final_norm.gamma" in tensors:
        state["norm.weight"] = tensor("final_norm.gamma")
        state["norm.bias"] = tensor("final_norm.beta")
    else:
        state["norm.weight"] = _reshape([d_model], [1.0] * d_model)
        state["norm.bias"] = zeros(d_model)
    for i in range(n_layer):
        block = f"blocks.{i}"
        if grouped:
            state[f"layers.{i}.q_proj.weight"] = tensor(f"{block}.attn.q")
            state[f"layers.{i}.k_proj.weight"] = tensor(f"{block}.attn.k")
            state[f"layers.{i}.v_proj.weight"] = tensor(f"{block}.attn.v")
            state[f"layers.{i}.o_proj.weight"] = tensor(f"{block}.attn.out")
            state[f"layers.{i}.ffn.0.weight"] = tensor(f"{block}.ffn")
            state[f"layers.{i}.ffn.0.bias"] = zeros(int(meta["ffn_dim"]))
            state[f"layers.{i}.ffn.3.weight"] = tensor(f"{block}.ffn2")
            state[f"layers.{i}.ffn.3.bias"] = zeros(d_model)
        else:
            fused = tensor(f"{block}.attn.q") + tensor(f"{block}.attn.k") + tensor(f"{block}.attn.v")
            state[f"layers.{i}.self_attn.in_proj_weight"] = fused
            state[f"layers.{i}.self_attn.in_proj_bias"] = zeros(3 * d_model)
            state[f"layers.{i}.self_attn.out_proj.weight"] = tensor(f"{block}.attn.out")
            state[f"layers.{i}.self_attn.out_proj.bias"] = zeros(d_model)
            state[f"layers.{i}.linear1.weight"] = tensor(f"{block}.ffn")
            state[f"layers.{i}.linear1.bias"] = zeros(int(meta["ffn_dim"]))
            state[f"layers.{i}.linear2.weight"] = tensor(f"{block}.ffn2")
            state[f"layers.{i}.linear2.bias"] = zeros(d_model)
        state[f"layers.{i}.norm1.weight"] = tensor(f"{block}.ln1.gamma")
        state[f"layers.{i}.norm1.bias"] = tensor(f"{block}.ln1.beta")
        state[f"layers.{i}.norm2.weight"] = tensor(f"{block}.ln2.gamma")
        state[f"layers.{i}.norm2.bias"] = tensor(f"{block}.ln2.beta")

    named = {f"{prefix}{name}": value for name, value in state.items()}
    if str(path).lower().endswith(".safetensors"):
        save_safetensors(path, named)
    else:
        save_pt(path, named)
    return named


_WRAPPER_PREFIXES = ("model_state.", "state_dict.", "module.", "model.")


def _strip_wrapper_prefixes(arrays):
    names = list(arrays)
    for prefix in _WRAPPER_PREFIXES:
        if names and all(name.startswith(prefix) for name in names):
            arrays = {name[len(prefix) :]: value for name, value in arrays.items()}
            names = list(arrays)
    return arrays


def import_checkpoint(path, *, n_heads=None, n_loops=1, seed=None):
    """Load a DistribAI checkpoint into a :class:`~magicmindnet.Chatbot`.

    Accepts everything DistribAI writes: ``torch.save({"model_state": ...})``
    wrapper checkpoints, plain ``state_dict`` ``.pt`` files, and
    ``export_safetensors`` output — read with MagicMindNet's from-scratch
    codecs, so no torch install is required. Both DistribAI layer flavours
    map: stock ``TransformerEncoderLayer`` names (fused
    ``self_attn.in_proj_weight`` is split into Q/K/V) and
    ``_GroupedQueryAttnBlock`` names.

    Torch checkpoints do not record the head count, so pass ``n_heads`` when
    it differs from DistribAI's default (4, or 1 when 4 does not divide
    ``dim`` — the ``DistribAITinyLanguageModel`` fallback). ``n_loops``
    restores shared-weight looping for jobs submitted with
    ``n_logical_layers > n_unique_layers``. Layer norms, embeddings, the
    final norm, and learned positions all carry over; torch-only biases
    (attention/FFN/``fc_out``) have no MagicMindNet slot and are dropped.
    """
    arrays = _strip_wrapper_prefixes(load_arrays(path))
    if "embedding.weight" not in arrays:
        raise ValueError(
            f"{path} is not a DistribAI checkpoint: missing embedding.weight "
            "(expected DistribAIModelWrapper state dict names)"
        )
    embed = arrays["embedding.weight"]
    vocab_size, d_model = _shape_of(embed)
    layer_ids = set()
    for name in arrays:
        if name.startswith("layers."):
            layer_ids.add(int(name.split(".")[1]))
    if not layer_ids:
        raise ValueError(f"{path} has no layers.N.* tensors")
    n_layer = max(layer_ids) + 1
    if layer_ids != set(range(n_layer)):
        raise ValueError(f"{path} is missing layers: got indices {sorted(layer_ids)}")

    if n_heads is None:
        n_heads = 4 if d_model % 4 == 0 else 1
    if d_model % n_heads:
        raise ValueError(f"n_heads={n_heads} must divide d_model={d_model}")
    head_dim = d_model // n_heads

    grouped = "layers.0.q_proj.weight" in arrays
    if grouped:
        kv_dim = _shape_of(arrays["layers.0.k_proj.weight"])[0]
        if kv_dim % head_dim:
            raise ValueError(
                f"k_proj rows ({kv_dim}) must be a multiple of head_dim ({head_dim}); "
                "pass the n_heads= the job was created with"
            )
        n_kv_heads = kv_dim // head_dim
        ffn_dim = _shape_of(arrays["layers.0.ffn.0.weight"])[0]
    else:
        n_kv_heads = n_heads
        ffn_dim = _shape_of(arrays["layers.0.linear1.weight"])[0]

    meta = {
        "vocab_size": vocab_size,
        "n_layer": n_layer,
        "d_model": d_model,
        "ffn_dim": ffn_dim,
        "n_heads": n_heads,
        "vision": False,
        "final_norm": True,
    }
    if n_kv_heads != n_heads:
        meta["n_kv_heads"] = n_kv_heads
    if n_loops != 1:
        meta["n_loops"] = int(n_loops)
    if seed is not None:
        meta["seed"] = int(seed)

    tensors = {
        "embed": embed,
        "lm_head": arrays["fc_out.weight"],
        "final_norm.gamma": arrays.get("norm.weight", [1.0] * d_model),
        "final_norm.beta": arrays.get("norm.bias", [0.0] * d_model),
    }
    if "position_embedding.weight" in arrays:
        pe = arrays["position_embedding.weight"]
        meta["use_learned_pos_embed"] = True
        meta["max_seq_len"] = _shape_of(pe)[0]
        tensors["pos_embed"] = pe
    for i in range(n_layer):
        layer = f"layers.{i}"
        block = f"blocks.{i}"
        if grouped:
            tensors[f"{block}.attn.q"] = arrays[f"{layer}.q_proj.weight"]
            tensors[f"{block}.attn.k"] = arrays[f"{layer}.k_proj.weight"]
            tensors[f"{block}.attn.v"] = arrays[f"{layer}.v_proj.weight"]
            tensors[f"{block}.attn.out"] = arrays[f"{layer}.o_proj.weight"]
            tensors[f"{block}.ffn"] = arrays[f"{layer}.ffn.0.weight"]
            tensors[f"{block}.ffn2"] = arrays[f"{layer}.ffn.3.weight"]
        else:
            fused = arrays[f"{layer}.self_attn.in_proj_weight"]
            rows = _shape_of(fused)[0]
            if rows != 3 * d_model:
                raise ValueError(
                    f"{layer}.self_attn.in_proj_weight has {rows} rows, expected {3 * d_model}"
                )
            tensors[f"{block}.attn.q"] = fused[:d_model]
            tensors[f"{block}.attn.k"] = fused[d_model : 2 * d_model]
            tensors[f"{block}.attn.v"] = fused[2 * d_model :]
            tensors[f"{block}.attn.out"] = arrays[f"{layer}.self_attn.out_proj.weight"]
            tensors[f"{block}.ffn"] = arrays[f"{layer}.linear1.weight"]
            tensors[f"{block}.ffn2"] = arrays[f"{layer}.linear2.weight"]
        tensors[f"{block}.ln1.gamma"] = arrays[f"{layer}.norm1.weight"]
        tensors[f"{block}.ln1.beta"] = arrays[f"{layer}.norm1.bias"]
        tensors[f"{block}.ln2.gamma"] = arrays[f"{layer}.norm2.weight"]
        tensors[f"{block}.ln2.beta"] = arrays[f"{layer}.norm2.bias"]

    entries = {}
    for name, nested in tensors.items():
        flat = _flat(nested)
        entries[name] = {
            "data": list(struct.pack(f"<{len(flat)}f", *flat)),
            "dtype": "F32",
            "shape": _shape_of(nested),
        }
    checkpoint = {"format": "mmn-safetensors-v1", "meta": meta, "tensors": entries}
    with tempfile.TemporaryDirectory() as tmp:
        target = os.path.join(tmp, "imported.mmn")
        with open(target, "w", encoding="utf-8") as handle:
            json.dump(checkpoint, handle)
        return _native.Chatbot.load(target)
