"""DistribAI compatibility bridge — run MagicMindNet on DistribAI grids.

`DistribAI <https://github.com/naxium-oss/DistribAI>`_ pools contributor GPUs
into distributed training jobs: an orchestrator schedules micro-tasks over
gRPC and worker nodes execute them. This module makes a MagicMindNet install
a first-class citizen on any device that also runs DistribAI:

- **Coexistence** — ``magicmindnet`` has zero required Python dependencies, so
  it installs cleanly into a DistribAI node/orchestrator venv
  (:func:`check_compatibility` reports the details).
- **Architecture bridge** — :func:`architecture_config` maps a
  :class:`~magicmindnet.Chatbot` onto DistribAI's declarative
  ``architecture_config`` (family ``decoder_transformer``), validated with a
  faithful mirror of DistribAI's own bounds. :func:`chatbot_from_architecture`
  goes the other way. MagicMindNet's ``n_loops`` shared-weight looping maps
  exactly onto DistribAI's ``n_unique_layers`` / ``n_logical_layers``.
- **MyTrainer contract** — :func:`export_mytrainer_tree` (and the committed
  ``configs/grid_architectures.json`` at the repo root) satisfy DistribAI's
  ``external/mytrainer`` sync: ``git clone`` MagicMindNet at
  ``external/mytrainer`` and ``POST /api/admin/mytrainer/sync`` registers the
  MagicMindNet grid profiles.
- **Script jobs** — :func:`build_script_package` builds the gzip tarball
  DistribAI workers execute (``run.py`` + ``requirements.txt`` +
  ``config.json``), pre-validated against DistribAI's preflight and static
  script rules. :func:`default_run_py` is a ready-made MagicMindNet training
  entrypoint honouring the worker contract (``hyperparams.json`` in,
  ``results.json`` / ``metrics.json`` / ``checkpoint.pt`` out).
- **Job submission** — :class:`DistribAIClient` talks to a local or remote
  orchestrator admin API (``POST /admin/jobs`` etc.) with stdlib HTTP only.
- **Checkpoint interop** — :func:`export_checkpoint` writes DistribAI-native
  ``state_dict`` names (``model.embedding.weight``,
  ``model.layers.N.self_attn.in_proj_weight``, ...) as ``.pt`` or
  ``.safetensors``; :func:`import_checkpoint` reads DistribAI checkpoints
  (including the nested ``{"model_state": ...}`` wrapper their trainer saves)
  back into a :class:`~magicmindnet.Chatbot`. No torch install needed for
  either direction.

Contract constants mirror DistribAI 0.9.0 (``services_python`` /
``worker/src/daemon/script_runner.py``).
"""

import ast
import base64
import hashlib
import io
import json
import math
import os
import re
import sys
import tarfile
import urllib.error
import urllib.request
from dataclasses import dataclass
from importlib import metadata as _importlib_metadata
from pathlib import Path

from magicmindnet import _native
from magicmindnet.distribai.checkpoints import (
    export_checkpoint,
    import_checkpoint,
    sinusoidal_position_encoding,
)

__all__ = [
    "ARCHITECTURE_CONFIG_VERSION",
    "DEFAULT_ADMIN_URL",
    "GRID_PROFILES",
    "MAX_ESTIMATED_PARAMETERS",
    "MAX_SCRIPT_PACKAGE_BYTES",
    "MAX_TRANSFORMER_SEQ_LEN",
    "SUPPORTED_ARCHITECTURE_FAMILIES",
    "DistribAIClient",
    "DistribAIError",
    "DistribAIInstall",
    "architecture_config",
    "build_script_package",
    "chatbot_from_architecture",
    "check_compatibility",
    "default_run_py",
    "detect_install",
    "export_checkpoint",
    "export_mytrainer_tree",
    "grid_architectures",
    "import_checkpoint",
    "package_sha256",
    "sinusoidal_position_encoding",
    "validate_architecture_config",
    "validate_script_package",
    "validate_script_source",
    "write_grid_architectures",
]


class DistribAIError(RuntimeError):
    """Raised when the DistribAI orchestrator rejects or fails a request."""


# ---------------------------------------------------------------------------
# DistribAI contract constants (mirrors services_python/architecture_config.py,
# services_python/preflight.py, and services_python/script_validation.py)
# ---------------------------------------------------------------------------

DEFAULT_ADMIN_URL = "http://127.0.0.1:8766"
ARCHITECTURE_CONFIG_VERSION = 1
SUPPORTED_ARCHITECTURE_FAMILIES = frozenset(
    {
        "decoder_transformer",
        "gru",
        "gated_conv",
        "moe_decoder",
        "lstm",
        "resnet_lm",
        "hybrid_attn_rnn",
        "dense_ffn",
    }
)
MAX_ARCHITECTURE_CONFIG_BYTES = 64 * 1024
MAX_ARCHITECTURE_CONFIG_DEPTH = 6
MAX_ESTIMATED_PARAMETERS = 512_000_000
MAX_TRANSFORMER_SEQ_LEN = 8192
MAX_SCRIPT_PACKAGE_BYTES = 5_000_000

_INT_LIMITS = {
    "dim": (16, 4096),
    "n_unique_layers": (1, 64),
    "n_logical_layers": (1, 128),
    "n_heads": (1, 64),
    "n_kv_heads": (1, 64),
    "ffn_dim": (16, 16384),
    "seq_len": (8, 32768),
    "sliding_window": (0, 32768),
    "engram_dim": (0, 1024),
    "mhc_expansion": (1, 16),
    "num_experts": (1, 64),
    "top_k": (1, 16),
    "conv_kernel": (2, 31),
    "gru_layers": (1, 16),
    "attn_res_block_size": (0, 64),
}
_FLOAT_LIMITS = {"dropout": (0.0, 0.5)}
_BOOL_KEYS = frozenset({"qk_norm", "use_head_gating", "embedding_scale"})
_ALLOWED_KEYS = {"version", "family", "architecture", *_INT_LIMITS, *_FLOAT_LIMITS, *_BOOL_KEYS}

_FORBIDDEN_BUNDLE_PATTERNS = (
    re.compile(r"(^|/)\.env($|/)"),
    re.compile(r"(^|/)id_rsa"),
    re.compile(r"(^|/)\.ssh/"),
    re.compile(r"\.pem$"),
    re.compile(r"\.key$"),
)
_DISALLOWED_CALLS = frozenset({"exec", "eval", "compile", "__import__", "input", "breakpoint"})
_DISALLOWED_IMPORTS = frozenset({"subprocess", "ctypes", "multiprocessing"})


# ---------------------------------------------------------------------------
# architecture_config validation (faithful mirror of DistribAI's validator)
# ---------------------------------------------------------------------------


def _container_nesting(value):
    deepest = 0
    pending = [(value, 0)]
    while pending:
        node, level = pending.pop()
        if not isinstance(node, (dict, list)):
            continue
        deepest = max(deepest, level)
        kids = node.values() if isinstance(node, dict) else node
        pending.extend((kid, level + 1) for kid in kids)
    return deepest


def _rough_parameter_count(config):
    """DistribAI's pre-queue native size estimate (vocab fixed at 256)."""
    dim = int(config.get("dim", 256))
    ffn_dim = int(config.get("ffn_dim", 4 * dim))
    layers = int(config.get("n_unique_layers", config.get("n_logical_layers", 8)))
    vocab = 256
    family = config["family"]
    if family == "decoder_transformer":
        return layers * (4 * dim * dim + 2 * dim * ffn_dim) + 2 * vocab * dim
    if family == "gru":
        gru_layers = int(config.get("gru_layers", 2))
        return gru_layers * 3 * (2 * dim * dim + 2 * dim) + 2 * vocab * dim
    if family == "lstm":
        lstm_layers = int(config.get("gru_layers", 2))
        return lstm_layers * 4 * (2 * dim * dim + 2 * dim) + 2 * vocab * dim
    if family == "gated_conv":
        conv_layers = int(config.get("n_logical_layers", 6))
        return conv_layers * 2 * dim * dim * int(config.get("conv_kernel", 5)) + 2 * vocab * dim
    if family == "resnet_lm":
        res_layers = int(config.get("n_logical_layers", 6))
        kernel = int(config.get("conv_kernel", 5))
        return res_layers * (dim * dim * kernel + dim * dim) + 2 * vocab * dim
    if family == "hybrid_attn_rnn":
        hybrid_layers = int(config.get("n_logical_layers", 8))
        attn_layers = (hybrid_layers + 1) // 2
        rnn_layers = hybrid_layers // 2
        return (
            attn_layers * (4 * dim * dim + 2 * dim * ffn_dim)
            + rnn_layers * 3 * (2 * dim * dim + 2 * dim)
            + 2 * vocab * dim
        )
    if family == "dense_ffn":
        ffn_layers = int(config.get("n_logical_layers", 8))
        return ffn_layers * (2 * dim * ffn_dim) + 2 * vocab * dim
    experts = int(config.get("num_experts", 4))
    moe_layers = int(config.get("n_logical_layers", 8))
    return moe_layers * (experts * 2 * dim * ffn_dim + dim * experts) + 2 * vocab * dim


def validate_architecture_config(value):
    """Normalize a DistribAI ``architecture_config`` dict, mirroring their bounds.

    Runs the exact checks DistribAI's orchestrator applies before queuing a
    job, so invalid configs fail locally with the same messages instead of a
    400 from the admin API. Returns the normalized dict; raises ``ValueError``
    on any violation.
    """
    if not isinstance(value, dict):
        raise ValueError("architecture_config must be an object")
    if _container_nesting(value) > MAX_ARCHITECTURE_CONFIG_DEPTH:
        raise ValueError("architecture_config is nested too deeply")
    try:
        encoded = len(json.dumps(value, separators=(",", ":"), allow_nan=False).encode())
    except (RecursionError, TypeError, ValueError) as exc:
        raise ValueError("architecture_config must contain JSON-finite values") from exc
    if encoded > MAX_ARCHITECTURE_CONFIG_BYTES:
        raise ValueError("architecture_config exceeds 64 KiB")
    unknown = set(value) - _ALLOWED_KEYS
    if unknown:
        raise ValueError(f"unsupported architecture_config keys: {sorted(unknown)}")

    normalized = dict(value)
    if "family" in normalized and "architecture" in normalized:
        raw_family, raw_arch = normalized["family"], normalized["architecture"]
        if (
            not isinstance(raw_family, str)
            or not isinstance(raw_arch, str)
            or raw_family.strip().lower() != raw_arch.strip().lower()
        ):
            raise ValueError("architecture_config.family and architecture must agree")
    raw_version = normalized.get("version", ARCHITECTURE_CONFIG_VERSION)
    if isinstance(raw_version, bool) or not isinstance(raw_version, int):
        raise ValueError("architecture_config.version must be an integer")
    normalized["version"] = raw_version
    if normalized["version"] != ARCHITECTURE_CONFIG_VERSION:
        raise ValueError(f"unsupported architecture_config version: {normalized['version']}")
    family = normalized.get("family", normalized.get("architecture"))
    if not isinstance(family, str) or family.strip().lower() not in SUPPORTED_ARCHITECTURE_FAMILIES:
        raise ValueError(
            "architecture_config.family must be one of: "
            + ", ".join(sorted(SUPPORTED_ARCHITECTURE_FAMILIES))
        )
    normalized["family"] = family.strip().lower()
    normalized["architecture"] = normalized["family"]

    for key, (minimum, maximum) in _INT_LIMITS.items():
        if key not in normalized:
            continue
        item = normalized[key]
        if isinstance(item, bool) or not isinstance(item, int):
            raise ValueError(f"architecture_config.{key} must be an integer")
        if not minimum <= item <= maximum:
            raise ValueError(f"architecture_config.{key} must be between {minimum} and {maximum}")
    for key, (minimum, maximum) in _FLOAT_LIMITS.items():
        if key not in normalized:
            continue
        item = normalized[key]
        if isinstance(item, bool) or not isinstance(item, (int, float)):
            raise ValueError(f"architecture_config.{key} must be numeric")
        if not math.isfinite(float(item)) or not minimum <= float(item) <= maximum:
            raise ValueError(f"architecture_config.{key} must be between {minimum} and {maximum}")
        normalized[key] = float(item)
    for key in _BOOL_KEYS:
        if key in normalized and not isinstance(normalized[key], bool):
            raise ValueError(f"architecture_config.{key} must be a boolean")

    if normalized["family"] in {"decoder_transformer", "hybrid_attn_rnn"}:
        dim = int(normalized.get("dim", 256))
        heads = int(normalized.get("n_heads", 8))
        if dim % heads:
            raise ValueError("architecture_config.dim must be divisible by n_heads")
        kv_heads = int(normalized.get("n_kv_heads", heads))
        if heads % kv_heads:
            raise ValueError("architecture_config.n_heads must be divisible by n_kv_heads")
        if int(normalized.get("seq_len", 512)) > MAX_TRANSFORMER_SEQ_LEN:
            raise ValueError(
                "architecture_config.seq_len must be at most "
                f"{MAX_TRANSFORMER_SEQ_LEN} for transformer attention"
            )
    if normalized["family"] == "decoder_transformer" and int(
        normalized.get("n_logical_layers", normalized.get("n_unique_layers", 8))
    ) < int(normalized.get("n_unique_layers", 8)):
        raise ValueError("n_logical_layers cannot be less than n_unique_layers")
    if normalized["family"] == "moe_decoder" and int(normalized.get("top_k", 2)) > int(
        normalized.get("num_experts", 4)
    ):
        raise ValueError("architecture_config.top_k cannot exceed num_experts")
    if _rough_parameter_count(normalized) > MAX_ESTIMATED_PARAMETERS:
        raise ValueError(
            f"architecture_config estimated parameter count exceeds {MAX_ESTIMATED_PARAMETERS:,}"
        )
    return normalized


# ---------------------------------------------------------------------------
# Chatbot <-> architecture_config mapping
# ---------------------------------------------------------------------------


def architecture_config(bot=None, **overrides):
    """Map a :class:`~magicmindnet.Chatbot` to a DistribAI ``architecture_config``.

    The result is a validated ``decoder_transformer`` config ready for
    ``POST /admin/jobs`` or a ``grid_architectures.json`` profile. Shared-weight
    looping maps as ``n_unique_layers = n_layer`` and
    ``n_logical_layers = n_layer * n_loops`` (DistribAI cycles its unique
    layers in the same order MagicMindNet loops its stack).

    Chatbot features with no DistribAI equivalent raise ``ValueError``:
    custom ``head_dim``, ``prelude_layers`` / ``coda_layers``, ``loop_embed``,
    ``lora_rank``, RMSNorm, SwiGLU, and vision prefixes.

    ``overrides`` are applied on top of the derived config (e.g.
    ``seq_len=2048``) before validation. Call with no ``bot`` to build a
    config from keyword arguments alone.
    """
    config = {"version": ARCHITECTURE_CONFIG_VERSION, "family": "decoder_transformer"}
    if bot is not None:
        for feature, message in (
            (bot.has_vision, "vision-enabled Chatbots"),
            (bot.norm == "rms", "RMSNorm (norm='rms')"),
            (bot.ffn == "swiglu", "SwiGLU FFNs (ffn='swiglu')"),
            (bot.loop_embed, "loop_embed"),
            (bot.lora_rank > 0, "LoopLoRA adapters (lora_rank > 0)"),
            (bot.prelude_layers > 0 or bot.coda_layers > 0, "prelude/coda layers"),
        ):
            if feature:
                raise ValueError(
                    f"DistribAI has no equivalent for {message}; "
                    "train this architecture with a MagicMindNet script job instead "
                    "(build_script_package)"
                )
        if bot.head_dim * bot.n_heads != bot.d_model:
            raise ValueError(
                "DistribAI attention requires head_dim == d_model / n_heads; "
                f"got head_dim={bot.head_dim} with d_model={bot.d_model}, n_heads={bot.n_heads}"
            )
        config.update(
            {
                "dim": bot.d_model,
                "n_unique_layers": bot.n_layer,
                "n_logical_layers": bot.n_layer * bot.n_loops,
                "n_heads": bot.n_heads,
                "n_kv_heads": bot.n_kv_heads,
                "ffn_dim": bot.ffn_dim,
                "seq_len": bot.max_seq_len,
            }
        )
        if bot.attention_window is not None:
            config["sliding_window"] = bot.attention_window
    config.update(overrides)
    return validate_architecture_config(config)


def chatbot_from_architecture(config, *, vocab_size=256, seed=None, **chatbot_kwargs):
    """Build a :class:`~magicmindnet.Chatbot` from a DistribAI ``architecture_config``.

    Only the ``decoder_transformer`` family maps onto a Chatbot; other
    families (gru, moe_decoder, ...) raise ``ValueError``. ``vocab_size``
    defaults to 256 — DistribAI's byte-level default. ``n_logical_layers``
    must be a whole multiple of ``n_unique_layers`` (the multiple becomes
    ``n_loops``); nonzero ``dropout`` and torch-only knobs (``qk_norm``,
    ``use_head_gating``, ``embedding_scale``, ``attn_res_block_size``) raise
    because MagicMindNet has no equivalent.
    """
    normalized = validate_architecture_config(config)
    family = normalized["family"]
    if family != "decoder_transformer":
        raise ValueError(
            f"only the decoder_transformer family maps onto a MagicMindNet Chatbot; got {family!r}"
        )
    if float(normalized.get("dropout", 0.0)) != 0.0:
        raise ValueError("MagicMindNet has no dropout; set architecture_config.dropout to 0")
    for knob in ("qk_norm", "use_head_gating", "embedding_scale"):
        if normalized.get(knob):
            raise ValueError(f"MagicMindNet has no equivalent for architecture_config.{knob}")
    if int(normalized.get("attn_res_block_size", 0)) != 0:
        raise ValueError("MagicMindNet has no equivalent for attn_res_block_size")
    dim = int(normalized.get("dim", 256))
    unique = int(normalized.get("n_unique_layers", normalized.get("n_logical_layers", 8)))
    logical = int(normalized.get("n_logical_layers", unique))
    if logical % unique:
        raise ValueError(
            "n_logical_layers must be a whole multiple of n_unique_layers to map onto "
            f"MagicMindNet n_loops; got {logical} logical / {unique} unique"
        )
    heads = int(normalized.get("n_heads", 8))
    kwargs = {
        "vocab_size": vocab_size,
        "n_layer": unique,
        "d_model": dim,
        "n_heads": heads,
        "n_kv_heads": int(normalized.get("n_kv_heads", heads)),
        "ffn_dim": int(normalized.get("ffn_dim", 4 * dim)),
        "max_seq_len": int(normalized.get("seq_len", 512)),
        "n_loops": logical // unique,
    }
    if int(normalized.get("sliding_window", 0)) > 0:
        kwargs["attention_window"] = int(normalized["sliding_window"])
    if seed is not None:
        kwargs["seed"] = seed
    kwargs.update(chatbot_kwargs)
    return _native.Chatbot(**kwargs)


# ---------------------------------------------------------------------------
# Grid profiles + MyTrainer tree (external/mytrainer contract)
# ---------------------------------------------------------------------------

# MagicMindNet profiles registered by DistribAI's MyTrainer sync. Every entry
# is a complete architecture_config passing both validators: DistribAI's
# architecture bounds and chatbot_from_architecture (regression-tested).
_PROFILE_SHAPES = {
    "mmn-tiny": {"dim": 64, "n_unique_layers": 2, "n_logical_layers": 2, "n_heads": 4, "n_kv_heads": 4, "ffn_dim": 256, "seq_len": 512},
    "mmn-small": {"dim": 128, "n_unique_layers": 4, "n_logical_layers": 4, "n_heads": 4, "n_kv_heads": 4, "ffn_dim": 512, "seq_len": 1024},
    "mmn-base": {"dim": 256, "n_unique_layers": 6, "n_logical_layers": 6, "n_heads": 8, "n_kv_heads": 8, "ffn_dim": 1024, "seq_len": 2048},
    "mmn-medium": {"dim": 512, "n_unique_layers": 8, "n_logical_layers": 8, "n_heads": 8, "n_kv_heads": 8, "ffn_dim": 2048, "seq_len": 2048},
    "mmn-large": {"dim": 768, "n_unique_layers": 12, "n_logical_layers": 12, "n_heads": 12, "n_kv_heads": 12, "ffn_dim": 3072, "seq_len": 4096},
    "mmn-gqa-small": {"dim": 256, "n_unique_layers": 6, "n_logical_layers": 6, "n_heads": 8, "n_kv_heads": 2, "ffn_dim": 1024, "seq_len": 2048},
    "mmn-glint2-mini": {"dim": 256, "n_unique_layers": 4, "n_logical_layers": 32, "n_heads": 8, "n_kv_heads": 8, "ffn_dim": 1024, "seq_len": 2048},
}
GRID_PROFILES = {
    name: {"version": ARCHITECTURE_CONFIG_VERSION, "family": "decoder_transformer", **shape}
    for name, shape in _PROFILE_SHAPES.items()
}


def grid_architectures(extra=None):
    """Return the MagicMindNet grid profiles as ``{name: architecture_config}``.

    This is the payload DistribAI's MyTrainer sync
    (``external/mytrainer/configs/grid_architectures.json``) registers with
    ``DistribAIModelWrapper.register_model_config``. ``extra`` adds or
    overrides entries — values may be :class:`~magicmindnet.Chatbot` instances
    (mapped via :func:`architecture_config`) or plain config dicts.
    """
    configs = {name: validate_architecture_config(profile) for name, profile in GRID_PROFILES.items()}
    for name, value in (extra or {}).items():
        if isinstance(value, dict):
            configs[str(name)] = validate_architecture_config(value)
        else:
            configs[str(name)] = architecture_config(value)
    return configs


def write_grid_architectures(path, architectures=None):
    """Write ``grid_architectures.json`` (the MyTrainer sync config file)."""
    configs = architectures if architectures is not None else grid_architectures()
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(configs, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return str(target)


def export_mytrainer_tree(directory, architectures=None):
    """Materialize a DistribAI ``external/mytrainer``-shaped tree.

    Writes ``configs/grid_architectures.json`` (required by DistribAI's
    ``verify_mytrainer_submodule.py --require`` and ``MyTrainerSync``) plus a
    ``models/model.py`` builder module. Point ``MYTRAINER_PATH`` at the
    directory — or clone the MagicMindNet repo itself at
    ``external/mytrainer``, which ships the same files. Returns the list of
    written paths.
    """
    root = Path(directory)
    written = [write_grid_architectures(root / "configs" / "grid_architectures.json", architectures)]
    model_py = root / "models" / "model.py"
    model_py.parent.mkdir(parents=True, exist_ok=True)
    model_py.write_text(_MYTRAINER_MODEL_PY, encoding="utf-8")
    written.append(str(model_py))
    return written


_MYTRAINER_MODEL_PY = '''"""MagicMindNet grid architectures for DistribAI's MyTrainer sync.

DistribAI reads ``configs/grid_architectures.json`` from this tree
(``services_python/mytrainer_sync.py``); this module is the optional
Python-side builder for the same profiles. Requires ``pip install
magicmindnet`` — DistribAI itself never imports this file.
"""

from magicmindnet.distribai import chatbot_from_architecture, grid_architectures


def build_model(name, *, vocab_size=256, seed=None, **overrides):
    """Instantiate the named grid profile as a magicmindnet.Chatbot."""
    configs = grid_architectures()
    if name not in configs:
        raise ValueError(f"unknown grid architecture {name!r}; known: {sorted(configs)}")
    config = dict(configs[name])
    config.update(overrides)
    return chatbot_from_architecture(config, vocab_size=vocab_size, seed=seed)
'''


# ---------------------------------------------------------------------------
# Install detection / compatibility report
# ---------------------------------------------------------------------------

_CHECKOUT_MARKERS = (
    "services_python/orchestrator_grpc.py",
    "worker/src/daemon/run.py",
    "proto/distribai.proto",
)


@dataclass
class DistribAIInstall:
    """One detected DistribAI installation on this device."""

    kind: str  # "checkout" | "package" | "node-data" | "admin-url"
    path: str | None = None
    version: str | None = None
    admin_url: str | None = None


def _checkout_install(path):
    root = Path(path)
    if any((root / marker).exists() for marker in _CHECKOUT_MARKERS):
        return DistribAIInstall(kind="checkout", path=str(root))
    return None


def detect_install(path=None):
    """Find a DistribAI install on this device, or ``None``.

    Checks, in order: an explicit ``path`` (or ``DISTRIBAI_HOME``) repo
    checkout, the installed ``distribai`` pip package, a worker node data
    directory (``~/.distribai``), and a configured ``ORCHESTRATOR_ADMIN_URL``.
    """
    if path is not None:
        return _checkout_install(path)
    env_home = os.getenv("DISTRIBAI_HOME", "").strip()
    if env_home:
        found = _checkout_install(env_home)
        if found is not None:
            return found
    try:
        dist = _importlib_metadata.distribution("distribai")
    except _importlib_metadata.PackageNotFoundError:
        dist = None
    if dist is not None:
        return DistribAIInstall(kind="package", version=dist.version)
    node_dir = Path.home() / ".distribai"
    if node_dir.is_dir():
        return DistribAIInstall(kind="node-data", path=str(node_dir))
    admin_url = os.getenv("ORCHESTRATOR_ADMIN_URL", "").strip()
    if admin_url:
        return DistribAIInstall(kind="admin-url", admin_url=admin_url)
    return None


def check_compatibility(path=None):
    """Report how this MagicMindNet install coexists with DistribAI.

    Returns a dict: ``python_ok`` (DistribAI needs Python 3.11+, MagicMindNet
    3.12+ — any interpreter running this code satisfies both),
    ``dependency_conflicts`` (always ``[]``: magicmindnet declares zero
    required dependencies, so DistribAI's torch/grpcio/protobuf pins are
    untouched), ``install`` (a :class:`DistribAIInstall` or ``None``), and
    ``admin_url`` (the orchestrator endpoint a client would use).
    """
    install = detect_install(path)
    return {
        "python_ok": sys.version_info >= (3, 12),
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "dependency_conflicts": [],
        "install": install,
        "admin_url": (install.admin_url if install and install.admin_url else None)
        or os.getenv("ORCHESTRATOR_ADMIN_URL", "").strip()
        or DEFAULT_ADMIN_URL,
    }


# ---------------------------------------------------------------------------
# Script jobs: validation + packaging (worker ScriptRunner contract)
# ---------------------------------------------------------------------------


def validate_script_source(source):
    """Static-check Python source with DistribAI's submitted-script rules.

    Returns the list of error codes their orchestrator would produce
    (``disallowed_call:eval``, ``disallowed_import:subprocess``,
    ``syntax_error:...``); empty means the script passes.
    """
    errors = []
    stripped = (source or "").strip()
    if not stripped:
        return ["empty_script"]
    try:
        tree = ast.parse(source)
    except SyntaxError as exc:
        return [f"syntax_error:{exc.lineno}:{exc.msg}"]
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name) and func.id in _DISALLOWED_CALLS:
                errors.append(f"disallowed_call:{func.id}")
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".", 1)[0] in _DISALLOWED_IMPORTS:
                    errors.append(f"disallowed_import:{alias.name}")
        if isinstance(node, ast.ImportFrom) and node.module:
            if node.module.split(".", 1)[0] in _DISALLOWED_IMPORTS:
                errors.append(f"disallowed_import_from:{node.module}")
    return errors


def validate_script_package(data):
    """Check a gzip tarball against DistribAI's submission preflight.

    Mirrors ``services_python/preflight.py`` (size cap, ``run.py`` entry,
    forbidden secret paths) plus the worker's tar-member safety rules (no
    ``..`` traversal, absolute paths, symlinks, or device nodes). Raises
    ``ValueError`` with DistribAI's message on the first violation; returns
    ``{"member_count": ..., "size_bytes": ...}`` when the package is clean.
    """
    if not data:
        raise ValueError("empty script package")
    if len(data) > MAX_SCRIPT_PACKAGE_BYTES:
        raise ValueError("script package too large")
    try:
        with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as archive:
            members = archive.getmembers()
    except tarfile.TarError as exc:
        raise ValueError(f"invalid tarball: {exc}") from exc
    file_names = [member.name for member in members if member.isfile()]
    if not file_names:
        raise ValueError("tarball contains no files")
    for member in members:
        parts = Path(str(member.name).replace("\\", "/")).parts
        if not parts or ".." in parts or Path(member.name).is_absolute():
            raise ValueError(f"Invalid tar member rejected: {member.name}")
        if member.issym() or member.islnk() or member.isdev():
            raise ValueError(f"Invalid tar member rejected: {member.name}")
    if not any(name == "run.py" or name.endswith("/run.py") or name.endswith(".ipynb") for name in file_names):
        raise ValueError("entry script run.py (or run.ipynb) missing from package")
    for name in file_names:
        for blocked in _FORBIDDEN_BUNDLE_PATTERNS:
            if blocked.search(name):
                raise ValueError(f"forbidden path in bundle: {name}")
    return {"member_count": len(file_names), "size_bytes": len(data)}


def package_sha256(data):
    """SHA-256 hex digest workers verify against ``hyperparams.package_sha256``."""
    return hashlib.sha256(data).hexdigest()


def default_run_py():
    """Return the default MagicMindNet training entrypoint for a script job.

    The generated ``run.py`` follows the DistribAI worker contract
    (``worker/src/daemon/script_runner.py``): it reads ``hyperparams.json``
    and ``config.json`` from the task directory, honours
    ``DISTRIBAI_TASK_ID`` / ``DISTRIBAI_JOB_ID`` env vars, trains a Chatbot
    described by ``architecture_config``, and writes ``results.json``,
    ``metrics.json``, and ``checkpoint.pt`` (a DistribAI-native state dict)
    for the worker to collect. It passes DistribAI's static script validation
    (no exec/eval, no subprocess/ctypes/multiprocessing imports).
    """
    return _DEFAULT_RUN_PY


_DEFAULT_RUN_PY = '''"""MagicMindNet training task for a DistribAI worker node."""

import json
import os

import magicmindnet as ai
from magicmindnet import distribai as bridge


def read_json(path, fallback):
    if not os.path.exists(path):
        return fallback
    with open(path, encoding="utf-8") as handle:
        return json.load(handle)


def main():
    hyperparams = read_json("hyperparams.json", {})
    config = read_json("config.json", {})
    arch = (
        hyperparams.get("architecture_config")
        or config.get("architecture_config")
        or bridge.GRID_PROFILES["mmn-tiny"]
    )
    vocab_size = int(hyperparams.get("vocab_size", config.get("vocab_size", 256)))
    seed = hyperparams.get("seed", config.get("seed"))
    bot = bridge.chatbot_from_architecture(arch, vocab_size=vocab_size, seed=seed)

    rows = read_json("dataset.json", [{"input": "ping", "output": "pong"}])
    if isinstance(rows, dict):
        rows = rows.get("data", [])
    if hyperparams.get("dataset_kind", "qa") == "corpus":
        dataset = ai.DatasetCorpus(data=[row["text"] for row in rows])
    else:
        dataset = ai.DatasetQA(data=rows)

    losses = bot.train(
        dataset,
        epochs=int(hyperparams.get("epochs", 1)),
        batch_size=int(hyperparams.get("batch_size", 8)),
        learning_rate=float(hyperparams.get("lr", hyperparams.get("learning_rate", 3e-4))),
        optimizer=str(hyperparams.get("optimizer", "hybrid")),
    )

    bridge.export_checkpoint(bot, "checkpoint.pt")
    final_loss = losses[-1] if losses else None
    with open("results.json", "w", encoding="utf-8") as handle:
        json.dump(
            {
                "task_id": os.getenv("DISTRIBAI_TASK_ID", ""),
                "job_id": os.getenv("DISTRIBAI_JOB_ID", ""),
                "framework": "magicmindnet",
                "rows": dataset.rows,
                "losses": losses,
                "final_loss": final_loss,
            },
            handle,
        )
    with open("metrics.json", "w", encoding="utf-8") as handle:
        json.dump({"loss": final_loss, "epochs": int(hyperparams.get("epochs", 1))}, handle)
    print(f"magicmindnet task complete: final_loss={final_loss}")


if __name__ == "__main__":
    main()
'''


def build_script_package(
    run_py=None,
    *,
    files=None,
    requirements=("magicmindnet>=0.2.1",),
    config=None,
    dataset=None,
    out=None,
):
    """Build the gzip tar script package a DistribAI worker executes.

    ``run_py`` is the entry script source (defaults to
    :func:`default_run_py`); ``files`` maps extra archive paths to ``str`` or
    ``bytes`` contents; ``requirements`` becomes ``requirements.txt``
    (installed by the worker into an isolated ``.site-packages``); ``config``
    becomes ``config.json``; ``dataset`` (a list of ``{"input", "output"}``
    rows, ``{"text": ...}`` rows, or any JSON-serializable payload) becomes
    ``dataset.json``. The result is validated with
    :func:`validate_script_source` and :func:`validate_script_package` before
    it is returned (and optionally written to ``out``).
    """
    source = run_py if run_py is not None else default_run_py()
    errors = validate_script_source(source)
    if errors:
        raise ValueError(f"submitted script failed validation: {', '.join(errors)}")
    entries = {"run.py": source.encode("utf-8")}
    if requirements:
        entries["requirements.txt"] = ("\n".join(requirements) + "\n").encode("utf-8")
    if config is not None:
        entries["config.json"] = json.dumps(config).encode("utf-8")
    if dataset is not None:
        entries["dataset.json"] = json.dumps(dataset).encode("utf-8")
    for name, content in (files or {}).items():
        entries[str(name)] = content.encode("utf-8") if isinstance(content, str) else bytes(content)

    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as tar:
        for name in sorted(entries):
            info = tarfile.TarInfo(name=name)
            info.size = len(entries[name])
            info.mtime = 0
            tar.addfile(info, io.BytesIO(entries[name]))
    data = buffer.getvalue()
    validate_script_package(data)
    if out is not None:
        Path(out).parent.mkdir(parents=True, exist_ok=True)
        Path(out).write_bytes(data)
    return data


# ---------------------------------------------------------------------------
# Orchestrator admin API client (stdlib HTTP)
# ---------------------------------------------------------------------------


class DistribAIClient:
    """Talk to a DistribAI orchestrator admin API (``/admin/*`` routes).

    ``admin_url`` defaults to ``ORCHESTRATOR_ADMIN_URL`` (or DistribAI's
    loopback default ``http://127.0.0.1:8766``); ``admin_secret`` defaults to
    ``DISTRIBAI_ADMIN_SECRET`` and is sent as a Bearer token when set — the
    same conventions as the ``distribai`` CLI.
    """

    def __init__(self, admin_url=None, admin_secret=None, timeout=10.0):
        self.admin_url = (
            admin_url or os.getenv("ORCHESTRATOR_ADMIN_URL", "").strip() or DEFAULT_ADMIN_URL
        ).rstrip("/")
        self.admin_secret = (
            admin_secret if admin_secret is not None else os.getenv("DISTRIBAI_ADMIN_SECRET", "")
        )
        self.timeout = timeout

    def _request(self, method, route, body=None):
        url = f"{self.admin_url}{route}"
        payload = None if body is None else json.dumps(body).encode("utf-8")
        request = urllib.request.Request(url, data=payload, method=method)
        request.add_header("Accept", "application/json")
        if payload is not None:
            request.add_header("Content-Type", "application/json")
        if self.admin_secret:
            request.add_header("Authorization", f"Bearer {self.admin_secret}")
        try:
            with urllib.request.urlopen(request, timeout=self.timeout) as response:
                text = response.read().decode("utf-8")
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            try:
                detail = json.loads(detail).get("error", detail)
            except (json.JSONDecodeError, AttributeError):
                pass
            raise DistribAIError(f"{method} {route} failed with HTTP {exc.code}: {detail}") from exc
        except urllib.error.URLError as exc:
            raise DistribAIError(
                f"cannot reach DistribAI orchestrator at {self.admin_url}: {exc.reason}"
            ) from exc
        try:
            return json.loads(text) if text else {}
        except json.JSONDecodeError as exc:
            raise DistribAIError(f"{method} {route} returned invalid JSON") from exc

    def health(self):
        """``GET /admin/health`` — orchestrator liveness and queue counters."""
        return self._request("GET", "/admin/health")

    def stats(self):
        """``GET /admin/stats`` — operator dashboard counters."""
        return self._request("GET", "/admin/stats")

    def nodes(self):
        """``GET /admin/nodes`` — the registered worker fleet."""
        return self._request("GET", "/admin/nodes")

    def jobs(self):
        """``GET /admin/jobs`` — queued and running jobs."""
        return self._request("GET", "/admin/jobs")

    def job(self, job_id):
        """``GET /admin/jobs/{job_id}`` — one job's detail."""
        return self._request("GET", f"/admin/jobs/{job_id}")

    def cancel_job(self, job_id):
        """``DELETE /admin/jobs/{job_id}`` — cancel a queued/running job."""
        return self._request("DELETE", f"/admin/jobs/{job_id}")

    def retry_job(self, job_id):
        """``POST /admin/jobs/{job_id}/retry`` — requeue a terminal job."""
        return self._request("POST", f"/admin/jobs/{job_id}/retry")

    def submit_job(
        self,
        *,
        job_type="train",
        base_model="magicmindnet",
        steps=100,
        dataset_ref="",
        architecture_config=None,
        script_package=None,
        hyperparams=None,
        **extra,
    ):
        """``POST /admin/jobs`` — create a training job.

        ``architecture_config`` is validated locally with DistribAI's own
        bounds before submission. ``script_package`` (bytes from
        :func:`build_script_package`) rides along base64-encoded with its
        SHA-256 pinned in ``hparams.package_sha256``, exactly like
        ``distribai submit``.
        """
        hparams = dict(hyperparams or {})
        body = {
            "job_type": job_type,
            "base_model": base_model,
            "model_name": base_model,
            "steps": int(steps),
            "hparams": hparams,
        }
        if dataset_ref:
            body["dataset_ref"] = dataset_ref
        if architecture_config is not None:
            body["architecture_config"] = validate_architecture_config(architecture_config)
        if script_package is not None:
            validate_script_package(script_package)
            body["script_package_b64"] = base64.b64encode(script_package).decode("ascii")
            hparams["execution_paradigm"] = "script"
            hparams["package_sha256"] = package_sha256(script_package)
        body.update(extra)
        return self._request("POST", "/admin/jobs", body)

    def submit_training_job(self, bot_or_config, dataset=None, *, steps=100, job_type="train", hyperparams=None, **extra):
        """Submit a MagicMindNet training run as a DistribAI script job.

        ``bot_or_config`` is a :class:`~magicmindnet.Chatbot` or an
        ``architecture_config`` dict; ``dataset`` is a list of
        ``{"input", "output"}`` rows shipped as ``dataset.json`` inside the
        package. The worker executes :func:`default_run_py`, which trains the
        model with MagicMindNet and reports ``results.json`` /
        ``metrics.json`` / ``checkpoint.pt``.
        """
        if isinstance(bot_or_config, dict):
            arch = validate_architecture_config(bot_or_config)
        else:
            arch = architecture_config(bot_or_config)
        hparams = dict(hyperparams or {})
        hparams["architecture_config"] = arch
        package = build_script_package(config={"architecture_config": arch}, dataset=dataset)
        return self.submit_job(
            job_type=job_type,
            steps=steps,
            architecture_config=arch,
            script_package=package,
            hyperparams=hparams,
            **extra,
        )
