"""DistribAI checkpoint interop — state-dict names both ways, no torch needed.

``export_checkpoint`` writes the exact names ``DistribAIModelWrapper``
produces; ``import_checkpoint`` accepts flat state dicts, safetensors, and
the nested ``torch.save({"model_state": ...})`` wrapper DistribAI's trainer
saves. The wrapper fixture below is crafted with CPython's own ``pickle`` +
``zipfile`` so it is byte-level identical to what torch.save would emit.
"""

import io
import pickle
import struct
import sys
import zipfile
from collections import OrderedDict

import pytest

import magicmindnet as ai
from conftest import install_fake_torch as _install_fake_torch
from magicmindnet import distribai as bridge

DATA = ai.DatasetQA(data=[{"input": "hi", "output": "yo"}, {"input": "a", "output": "b"}])


def make_bot(**kwargs):
    defaults = {"vocab_size": 48, "n_layer": 2, "d_model": 16, "seed": 7, "final_norm": True}
    defaults.update(kwargs)
    return ai.Chatbot(**defaults)


# ---------------------------------------------------------------------------
# Export: DistribAI state-dict naming
# ---------------------------------------------------------------------------


def test_export_uses_stock_transformer_encoder_names(tmp_path):
    bot = make_bot()
    named = bridge.export_checkpoint(bot, str(tmp_path / "ck.pt"))
    for key in (
        "model.embedding.weight",
        "model.position_embedding.weight",
        "model.layers.0.self_attn.in_proj_weight",
        "model.layers.0.self_attn.in_proj_bias",
        "model.layers.0.self_attn.out_proj.weight",
        "model.layers.1.linear1.weight",
        "model.layers.1.linear2.bias",
        "model.layers.1.norm1.weight",
        "model.norm.weight",
        "model.norm.bias",
        "model.fc_out.weight",
        "model.fc_out.bias",
    ):
        assert key in named, key
    fused = named["model.layers.0.self_attn.in_proj_weight"]
    assert len(fused) == 3 * bot.d_model and len(fused[0]) == bot.d_model


def test_export_uses_gqa_block_names_when_kv_heads_differ(tmp_path):
    bot = make_bot(n_layer=1, n_heads=4, n_kv_heads=2)
    named = bridge.export_checkpoint(bot, str(tmp_path / "gqa.pt"))
    assert "model.layers.0.q_proj.weight" in named
    assert "model.layers.0.k_proj.weight" in named
    assert "model.layers.0.ffn.0.weight" in named
    assert "model.layers.0.ffn.3.bias" in named
    assert "model.layers.0.self_attn.in_proj_weight" not in named
    # GQA K projection carries n_kv_heads * head_dim rows.
    assert len(named["model.layers.0.k_proj.weight"]) == bot.d_model // 2


def test_export_prefix_can_be_disabled(tmp_path):
    named = bridge.export_checkpoint(make_bot(n_layer=1), str(tmp_path / "bare.pt"), prefix="")
    assert "embedding.weight" in named


def test_export_synthesizes_sinusoidal_position_table(tmp_path):
    bot = make_bot(n_layer=1, max_seq_len=32)
    named = bridge.export_checkpoint(bot, str(tmp_path / "pe.pt"))
    table = named["model.position_embedding.weight"]
    assert len(table) == 32
    expected = bridge.sinusoidal_position_encoding(32, bot.d_model)
    assert table[3][:4] == pytest.approx(expected[3][:4])


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"ffn": "swiglu"}, "SwiGLU"),
        ({"n_loops": 2, "loop_embed": True}, "loop_embed"),
        ({"n_loops": 2, "lora_rank": 2}, "LoopLoRA"),
        ({"prelude_layers": 1}, "prelude/coda"),
        ({"vision": True}, "vision"),
    ],
)
def test_export_rejects_features_distribai_cannot_store(tmp_path, kwargs, fragment):
    bot = make_bot(**kwargs)
    with pytest.raises(ValueError, match=fragment):
        bridge.export_checkpoint(bot, str(tmp_path / "no.pt"))


# ---------------------------------------------------------------------------
# Roundtrips: exact loss parity
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("extension", ["pt", "safetensors"])
def test_roundtrip_preserves_loss(tmp_path, extension):
    bot = make_bot()
    bot.train(DATA, epochs=2)
    path = str(tmp_path / f"ck.{extension}")
    bridge.export_checkpoint(bot, path)
    back = bridge.import_checkpoint(path)
    assert back.vocab_size == bot.vocab_size
    assert back.n_layer == bot.n_layer
    assert back.d_model == bot.d_model
    assert back.compute_mean_loss(DATA) == pytest.approx(bot.compute_mean_loss(DATA), abs=1e-5)


def test_gqa_roundtrip_preserves_loss_and_shape(tmp_path):
    bot = make_bot(n_layer=1, n_heads=4, n_kv_heads=2)
    path = str(tmp_path / "gqa.pt")
    bridge.export_checkpoint(bot, path)
    back = bridge.import_checkpoint(path)
    assert back.n_heads == 4
    assert back.n_kv_heads == 2
    assert back.compute_mean_loss(DATA) == pytest.approx(bot.compute_mean_loss(DATA), abs=1e-5)


def test_learned_pos_embed_roundtrip(tmp_path):
    bot = make_bot(n_layer=1, use_learned_pos_embed=True, max_seq_len=64)
    path = str(tmp_path / "lpe.pt")
    bridge.export_checkpoint(bot, path)
    back = bridge.import_checkpoint(path)
    assert back.use_learned_pos_embed is True
    assert back.max_seq_len == 64
    assert back.compute_mean_loss(DATA) == pytest.approx(bot.compute_mean_loss(DATA), abs=1e-5)


def test_loop_roundtrip_via_n_loops_argument(tmp_path):
    bot = make_bot(n_layer=1, n_loops=3)
    path = str(tmp_path / "loops.pt")
    bridge.export_checkpoint(bot, path)
    back = bridge.import_checkpoint(path, n_loops=3)
    assert back.n_loops == 3
    assert back.compute_mean_loss(DATA) == pytest.approx(bot.compute_mean_loss(DATA), abs=1e-5)


def test_import_seed_is_recorded(tmp_path):
    path = str(tmp_path / "seed.pt")
    bridge.export_checkpoint(make_bot(n_layer=1), path)
    assert bridge.import_checkpoint(path, seed=11).init_seed == 11


# ---------------------------------------------------------------------------
# Genuine DistribAI wrapper checkpoints (torch.save({"model_state": ...}))
# ---------------------------------------------------------------------------


class _StorageRef:
    def __init__(self, key, numel):
        self.key = key
        self.numel = numel


class _TensorStub:
    """Pickles exactly like a torch tensor (_rebuild_tensor_v2 + storage pid)."""

    def __init__(self, key, shape):
        self.key = key
        self.shape = tuple(shape)

    def __reduce__(self):
        numel = 1
        for dim in self.shape:
            numel *= dim
        stride, acc = [], 1
        for dim in reversed(self.shape):
            stride.insert(0, acc)
            acc *= dim
        return (
            sys.modules["torch._utils"]._rebuild_tensor_v2,
            (_StorageRef(self.key, numel), 0, self.shape, tuple(stride), False, OrderedDict()),
        )


class _FakeModelConfig:
    """Stands in for DistribAI's ModelConfig dataclass inside the pickle."""


def _write_distribai_wrapper_checkpoint(path, tensors, extras):
    """Craft the exact archive DistribAIModelWrapper.save_checkpoint writes."""
    torch_module, _ = _install_fake_torch()

    class TorchStylePickler(pickle.Pickler):
        def persistent_id(self, obj):
            if isinstance(obj, _StorageRef):
                return ("storage", torch_module.FloatStorage, obj.key, "cpu", obj.numel)
            return None

    state = OrderedDict()
    payloads = {}
    for index, (name, (shape, fill)) in enumerate(tensors.items()):
        key = str(index)
        state[name] = _TensorStub(key, shape)
        numel = 1
        for dim in shape:
            numel *= dim
        payloads[key] = struct.pack(f"<{numel}f", *([fill] * numel))
    wrapper = {"model_state": state, **extras}

    buffer = io.BytesIO()
    TorchStylePickler(buffer, protocol=2).dump(wrapper)
    with zipfile.ZipFile(path, "w", zipfile.ZIP_STORED) as archive:
        archive.writestr("archive/data.pkl", buffer.getvalue())
        archive.writestr("archive/version", "3\n")
        for key, payload in payloads.items():
            archive.writestr(f"archive/data/{key}", payload)


def _distribai_state_tensors(d_model=8, vocab=16, seq_len=32, ffn=32):
    tensors = OrderedDict()
    tensors["model.embedding.weight"] = ([vocab, d_model], 0.5)
    tensors["model.position_embedding.weight"] = ([seq_len, d_model], 0.01)
    layer = "model.layers.0"
    tensors[f"{layer}.self_attn.in_proj_weight"] = ([3 * d_model, d_model], 0.1)
    tensors[f"{layer}.self_attn.in_proj_bias"] = ([3 * d_model], 0.0)
    tensors[f"{layer}.self_attn.out_proj.weight"] = ([d_model, d_model], 0.1)
    tensors[f"{layer}.self_attn.out_proj.bias"] = ([d_model], 0.0)
    tensors[f"{layer}.linear1.weight"] = ([ffn, d_model], 0.2)
    tensors[f"{layer}.linear1.bias"] = ([ffn], 0.0)
    tensors[f"{layer}.linear2.weight"] = ([d_model, ffn], 0.2)
    tensors[f"{layer}.linear2.bias"] = ([d_model], 0.0)
    tensors[f"{layer}.norm1.weight"] = ([d_model], 1.0)
    tensors[f"{layer}.norm1.bias"] = ([d_model], 0.0)
    tensors[f"{layer}.norm2.weight"] = ([d_model], 1.0)
    tensors[f"{layer}.norm2.bias"] = ([d_model], 0.0)
    tensors["model.norm.weight"] = ([d_model], 1.0)
    tensors["model.norm.bias"] = ([d_model], 0.0)
    tensors["model.fc_out.weight"] = ([vocab, d_model], 0.3)
    tensors["model.fc_out.bias"] = ([vocab], 0.0)
    return tensors


def test_import_genuine_distribai_wrapper_checkpoint(tmp_path):
    """A save_checkpoint-style archive (nested model_state + config/step/loss)."""
    path = tmp_path / "wrapper.pt"
    _write_distribai_wrapper_checkpoint(
        path,
        _distribai_state_tensors(),
        extras={"config": _FakeModelConfig(), "step": 25, "loss": 2.5},
    )
    bot = bridge.import_checkpoint(str(path))
    assert bot.vocab_size == 16
    assert bot.d_model == 8
    assert bot.n_layer == 1
    assert bot.n_heads == 4  # DistribAI's ModelConfig default
    assert bot.use_learned_pos_embed is True
    assert bot.max_seq_len == 32
    loss = bot.compute_mean_loss(ai.DatasetQA(data=[{"input": "a", "output": "b"}]))
    assert loss == loss  # finite


def test_load_pt_flattens_nested_wrapper_with_dotted_names(tmp_path):
    """`ai.load_pt` itself surfaces nested state dicts (Rust reader change)."""
    path = tmp_path / "wrapper.pt"
    _write_distribai_wrapper_checkpoint(
        path,
        OrderedDict({"model.embedding.weight": ([2, 2], 1.5)}),
        extras={"step": 3},
    )
    arrays = ai.load_pt(str(path))
    assert arrays == {"model_state.model.embedding.weight": [[1.5, 1.5], [1.5, 1.5]]}


def test_import_checkpoint_derives_odd_head_fallback(tmp_path):
    """dim not divisible by 4 falls back to a single head, like DistribAI."""
    path = tmp_path / "odd.pt"
    _write_distribai_wrapper_checkpoint(
        path,
        _distribai_state_tensors(d_model=6, ffn=24),
        extras={"step": 1},
    )
    assert bridge.import_checkpoint(str(path)).n_heads == 1


def test_import_checkpoint_accepts_explicit_heads(tmp_path):
    path = tmp_path / "heads.pt"
    _write_distribai_wrapper_checkpoint(path, _distribai_state_tensors(), extras={})
    assert bridge.import_checkpoint(str(path), n_heads=2).n_heads == 2


# ---------------------------------------------------------------------------
# Import error paths
# ---------------------------------------------------------------------------


def test_import_rejects_non_distribai_checkpoints(tmp_path):
    path = str(tmp_path / "other.pt")
    ai.save_pt(path, {"weights.w": [[1.0]]})
    with pytest.raises(ValueError, match="missing embedding.weight"):
        bridge.import_checkpoint(path)


def test_import_rejects_missing_layers(tmp_path):
    path = str(tmp_path / "nolayers.pt")
    ai.save_pt(path, {"embedding.weight": [[1.0, 2.0], [3.0, 4.0]]})
    with pytest.raises(ValueError, match="no layers"):
        bridge.import_checkpoint(path)


def test_import_rejects_bad_head_count(tmp_path):
    bot = make_bot(n_layer=1)
    path = str(tmp_path / "heads.pt")
    bridge.export_checkpoint(bot, path)
    with pytest.raises(ValueError, match="must divide d_model"):
        bridge.import_checkpoint(path, n_heads=3)


def test_import_rejects_malformed_fused_qkv(tmp_path):
    path = str(tmp_path / "fused.pt")
    ai.save_pt(
        path,
        {
            "embedding.weight": [[0.0] * 8] * 16,
            "layers.0.self_attn.in_proj_weight": [[0.0] * 8] * 8,  # not 3*d rows
            "layers.0.self_attn.out_proj.weight": [[0.0] * 8] * 8,
            "layers.0.linear1.weight": [[0.0] * 8] * 32,
            "layers.0.linear2.weight": [[0.0] * 32] * 8,
            "layers.0.norm1.weight": [1.0] * 8,
            "layers.0.norm1.bias": [0.0] * 8,
            "layers.0.norm2.weight": [1.0] * 8,
            "layers.0.norm2.bias": [0.0] * 8,
            "norm.weight": [1.0] * 8,
            "norm.bias": [0.0] * 8,
            "fc_out.weight": [[0.0] * 8] * 16,
        },
    )
    with pytest.raises(ValueError, match="in_proj_weight has 8 rows"):
        bridge.import_checkpoint(path)
