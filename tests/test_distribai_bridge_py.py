"""DistribAI bridge — architecture configs, grid profiles, install detection.

Mirrors the DistribAI 0.9.0 contract (``services_python/architecture_config.py``
and ``services_python/mytrainer_sync.py``): every check here matches what
their orchestrator enforces before queuing a job.
"""

import json
from pathlib import Path

import pytest

import magicmindnet as ai
from magicmindnet import distribai as bridge

ROOT = Path(__file__).resolve().parents[1]


def make_bot(**kwargs):
    defaults = {"vocab_size": 48, "n_layer": 2, "d_model": 16, "seed": 7}
    defaults.update(kwargs)
    return ai.Chatbot(**defaults)


# ---------------------------------------------------------------------------
# architecture_config: Chatbot -> DistribAI
# ---------------------------------------------------------------------------


def test_architecture_config_maps_chatbot_shape():
    bot = make_bot()
    config = bridge.architecture_config(bot)
    assert config["family"] == "decoder_transformer"
    assert config["architecture"] == "decoder_transformer"
    assert config["version"] == 1
    assert config["dim"] == bot.d_model
    assert config["n_unique_layers"] == bot.n_layer
    assert config["n_logical_layers"] == bot.n_layer
    assert config["n_heads"] == bot.n_heads
    assert config["n_kv_heads"] == bot.n_kv_heads
    assert config["ffn_dim"] == bot.ffn_dim
    assert config["seq_len"] == bot.max_seq_len


def test_architecture_config_maps_loops_to_logical_layers():
    bot = make_bot(n_layer=2, n_loops=4)
    config = bridge.architecture_config(bot)
    assert config["n_unique_layers"] == 2
    assert config["n_logical_layers"] == 8


def test_architecture_config_maps_attention_window_to_sliding_window():
    bot = make_bot(attention_window=128)
    assert bridge.architecture_config(bot)["sliding_window"] == 128


def test_architecture_config_accepts_overrides():
    config = bridge.architecture_config(make_bot(), seq_len=2048)
    assert config["seq_len"] == 2048


def test_architecture_config_without_bot_uses_overrides_only():
    config = bridge.architecture_config(dim=64, n_heads=4, ffn_dim=256)
    assert config["dim"] == 64


@pytest.mark.parametrize(
    "kwargs,fragment",
    [
        ({"norm": "rms"}, "RMSNorm"),
        ({"ffn": "swiglu"}, "SwiGLU"),
        ({"n_loops": 2, "loop_embed": True}, "loop_embed"),
        ({"n_loops": 2, "lora_rank": 2}, "LoopLoRA"),
        ({"prelude_layers": 1}, "prelude/coda"),
        ({"coda_layers": 1}, "prelude/coda"),
        ({"vision": True}, "vision"),
    ],
)
def test_architecture_config_rejects_unmappable_features(kwargs, fragment):
    bot = make_bot(**kwargs)
    with pytest.raises(ValueError, match=fragment):
        bridge.architecture_config(bot)


def test_architecture_config_rejects_custom_head_dim():
    bot = make_bot(d_model=16, n_heads=2, head_dim=16)
    with pytest.raises(ValueError, match="head_dim"):
        bridge.architecture_config(bot)


# ---------------------------------------------------------------------------
# validate_architecture_config: DistribAI bounds mirror
# ---------------------------------------------------------------------------


def test_validator_normalizes_family_and_version():
    config = bridge.validate_architecture_config({"family": "Decoder_Transformer ", "dim": 64, "n_heads": 4})
    assert config["family"] == "decoder_transformer"
    assert config["architecture"] == "decoder_transformer"
    assert config["version"] == 1


@pytest.mark.parametrize(
    "config,fragment",
    [
        ([], "must be an object"),
        ({"family": "decoder_transformer", "warp_drive": 1}, "unsupported architecture_config keys"),
        ({"family": "nonsense"}, "family must be one of"),
        ({"family": "decoder_transformer", "version": 2}, "unsupported architecture_config version"),
        ({"family": "decoder_transformer", "dim": 8}, "dim must be between 16 and 4096"),
        ({"family": "decoder_transformer", "dim": 65, "n_heads": 4}, "divisible by n_heads"),
        ({"family": "decoder_transformer", "n_heads": 4, "n_kv_heads": 3}, "divisible by n_kv_heads"),
        ({"family": "decoder_transformer", "seq_len": 16384}, "at most 8192 for transformer"),
        (
            {"family": "decoder_transformer", "n_unique_layers": 8, "n_logical_layers": 4},
            "n_logical_layers cannot be less",
        ),
        ({"family": "moe_decoder", "num_experts": 2, "top_k": 4}, "top_k cannot exceed"),
        ({"family": "decoder_transformer", "dropout": 0.9}, "dropout must be between"),
        ({"family": "decoder_transformer", "qk_norm": 1}, "must be a boolean"),
        ({"family": "gru", "architecture": "lstm"}, "family and architecture must agree"),
        (
            {"family": "decoder_transformer", "dim": 4096, "n_unique_layers": 64, "ffn_dim": 16384},
            "estimated parameter count exceeds",
        ),
    ],
)
def test_validator_rejects_out_of_contract_configs(config, fragment):
    with pytest.raises(ValueError, match=fragment):
        bridge.validate_architecture_config(config)


def test_validator_matches_distribai_default_bounds():
    # The defaults DistribAI documents in docs/api/endpoints.md stay valid.
    config = bridge.validate_architecture_config(
        {
            "version": 1,
            "family": "moe_decoder",
            "dim": 256,
            "ffn_dim": 1024,
            "n_logical_layers": 8,
            "num_experts": 4,
            "top_k": 2,
            "seq_len": 2048,
        }
    )
    assert config["family"] == "moe_decoder"


# ---------------------------------------------------------------------------
# chatbot_from_architecture: DistribAI -> Chatbot
# ---------------------------------------------------------------------------


def test_chatbot_from_architecture_roundtrips_shape():
    original = make_bot(n_layer=2, n_loops=3)
    config = bridge.architecture_config(original)
    rebuilt = bridge.chatbot_from_architecture(config, vocab_size=original.vocab_size, seed=7)
    assert rebuilt.n_layer == original.n_layer
    assert rebuilt.d_model == original.d_model
    assert rebuilt.n_heads == original.n_heads
    assert rebuilt.n_kv_heads == original.n_kv_heads
    assert rebuilt.ffn_dim == original.ffn_dim
    assert rebuilt.max_seq_len == original.max_seq_len
    assert rebuilt.n_loops == original.n_loops


def test_chatbot_from_architecture_maps_gqa_and_window():
    config = {
        "family": "decoder_transformer",
        "dim": 32,
        "n_heads": 4,
        "n_kv_heads": 2,
        "n_unique_layers": 1,
        "n_logical_layers": 1,
        "ffn_dim": 64,
        "seq_len": 256,
        "sliding_window": 64,
    }
    bot = bridge.chatbot_from_architecture(config, vocab_size=32, seed=1)
    assert bot.n_kv_heads == 2
    assert bot.attention_window == 64
    assert bot.max_seq_len == 256


@pytest.mark.parametrize(
    "config,fragment",
    [
        ({"family": "gru", "dim": 64}, "decoder_transformer family"),
        ({"family": "decoder_transformer", "dropout": 0.1}, "no dropout"),
        ({"family": "decoder_transformer", "qk_norm": True}, "qk_norm"),
        ({"family": "decoder_transformer", "use_head_gating": True}, "use_head_gating"),
        ({"family": "decoder_transformer", "embedding_scale": True}, "embedding_scale"),
        ({"family": "decoder_transformer", "attn_res_block_size": 4}, "attn_res_block_size"),
        (
            {"family": "decoder_transformer", "n_unique_layers": 2, "n_logical_layers": 5},
            "whole multiple",
        ),
    ],
)
def test_chatbot_from_architecture_rejects_unmappable(config, fragment):
    with pytest.raises(ValueError, match=fragment):
        bridge.chatbot_from_architecture(config)


# ---------------------------------------------------------------------------
# Grid profiles + MyTrainer tree contract
# ---------------------------------------------------------------------------


def test_grid_profiles_all_validate_and_construct():
    configs = bridge.grid_architectures()
    assert set(bridge.GRID_PROFILES) <= set(configs)
    for name, config in configs.items():
        assert config == bridge.validate_architecture_config(config), name
        # Small vocab keeps construction cheap; shape mapping is what matters.
        bot = bridge.chatbot_from_architecture(config, vocab_size=32, seed=0)
        assert bot.d_model == config["dim"], name


def test_grid_profiles_include_loop_and_gqa_showcases():
    configs = bridge.grid_architectures()
    glint = configs["mmn-glint2-mini"]
    assert glint["n_logical_layers"] == 8 * glint["n_unique_layers"]
    gqa = configs["mmn-gqa-small"]
    assert gqa["n_kv_heads"] < gqa["n_heads"]


def test_grid_architectures_accepts_chatbot_and_dict_extras():
    bot = make_bot()
    configs = bridge.grid_architectures(
        extra={"my-bot": bot, "my-dict": {"family": "decoder_transformer", "dim": 64, "n_heads": 4}}
    )
    assert configs["my-bot"]["dim"] == bot.d_model
    assert configs["my-dict"]["dim"] == 64


def test_committed_grid_architectures_matches_generator():
    """configs/grid_architectures.json (the MyTrainer sync contract file) is generated."""
    committed = json.loads((ROOT / "configs" / "grid_architectures.json").read_text(encoding="utf-8"))
    assert committed == bridge.grid_architectures()


def test_committed_mytrainer_tree_has_distribai_markers():
    # DistribAI's verify_mytrainer_submodule.py requires this exact file.
    assert (ROOT / "configs" / "grid_architectures.json").is_file()
    # MyTrainerSync.check_for_updates also watches models/model.py.
    assert (ROOT / "models" / "model.py").is_file()


def test_export_mytrainer_tree_writes_contract_files(tmp_path):
    written = bridge.export_mytrainer_tree(tmp_path)
    config_file = tmp_path / "configs" / "grid_architectures.json"
    model_py = tmp_path / "models" / "model.py"
    assert set(written) == {str(config_file), str(model_py)}
    configs = json.loads(config_file.read_text(encoding="utf-8"))
    assert configs == bridge.grid_architectures()
    assert "grid_architectures" in model_py.read_text(encoding="utf-8")


def test_repo_models_model_py_builds_profiles():
    """The committed models/model.py builder instantiates grid profiles."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("mmn_mytrainer_model", ROOT / "models" / "model.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    bot = module.build_model("mmn-tiny", vocab_size=32, seed=0)
    assert bot.d_model == bridge.GRID_PROFILES["mmn-tiny"]["dim"]
    with pytest.raises(ValueError, match="unknown grid architecture"):
        module.build_model("mmn-nonexistent")


def test_write_grid_architectures_roundtrips(tmp_path):
    path = bridge.write_grid_architectures(tmp_path / "nested" / "grid.json")
    assert json.loads(Path(path).read_text(encoding="utf-8")) == bridge.grid_architectures()


# ---------------------------------------------------------------------------
# Install detection + compatibility report
# ---------------------------------------------------------------------------


def test_detect_install_finds_checkout_markers(tmp_path):
    (tmp_path / "proto").mkdir()
    (tmp_path / "proto" / "distribai.proto").write_text("syntax = \"proto3\";\n", encoding="utf-8")
    install = bridge.detect_install(tmp_path)
    assert install is not None
    assert install.kind == "checkout"
    assert install.path == str(tmp_path)


def test_detect_install_returns_none_for_plain_dir(tmp_path):
    assert bridge.detect_install(tmp_path) is None


def test_detect_install_honours_env_home(tmp_path, monkeypatch):
    (tmp_path / "services_python").mkdir()
    (tmp_path / "services_python" / "orchestrator_grpc.py").write_text("", encoding="utf-8")
    monkeypatch.setenv("DISTRIBAI_HOME", str(tmp_path))
    install = bridge.detect_install()
    assert install is not None and install.kind == "checkout"


def test_detect_install_admin_url_fallback(tmp_path, monkeypatch):
    monkeypatch.setenv("DISTRIBAI_HOME", "")
    monkeypatch.setenv("ORCHESTRATOR_ADMIN_URL", "http://10.0.0.5:8766")
    monkeypatch.setattr(bridge.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    install = bridge.detect_install()
    assert install is not None
    assert install.kind == "admin-url"
    assert install.admin_url == "http://10.0.0.5:8766"


def test_check_compatibility_reports_python_and_no_conflicts(monkeypatch, tmp_path):
    monkeypatch.setenv("DISTRIBAI_HOME", "")
    monkeypatch.delenv("ORCHESTRATOR_ADMIN_URL", raising=False)
    monkeypatch.setattr(bridge.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    report = bridge.check_compatibility()
    assert report["python_ok"] is True
    assert report["dependency_conflicts"] == []
    assert report["admin_url"] == bridge.DEFAULT_ADMIN_URL


def test_check_compatibility_uses_detected_admin_url(monkeypatch, tmp_path):
    monkeypatch.setenv("DISTRIBAI_HOME", "")
    monkeypatch.setenv("ORCHESTRATOR_ADMIN_URL", "http://grid.example:8766")
    monkeypatch.setattr(bridge.Path, "home", staticmethod(lambda: tmp_path / "nohome"))
    assert bridge.check_compatibility()["admin_url"] == "http://grid.example:8766"
