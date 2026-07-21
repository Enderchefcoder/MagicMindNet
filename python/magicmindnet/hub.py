"""Universal model hub: load / train / finetune from HF, ModelScope, Ollama, or local paths.

Design goals
------------
* One entry point: ``ai.from_pretrained(source)``.
* Do **not** special-case individual repos — route by pipeline / architecture /
  file layout so any Hub model is runnable.
* Prefer MagicMindNet native ``Chatbot`` / ``Classifier`` / ``Diffusion`` when
  weights adapt cleanly; otherwise wrap a foreign backend (transformers /
  diffusers / sentence-transformers) behind a uniform ``HubModel`` API.
* ``finetune`` / ``train`` / ``generate`` / ``predict`` work on every family
  that the backend supports.
"""

from __future__ import annotations

import json
import os
import re
import urllib.error
import urllib.request
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import magicmindnet as ai

PathLike = str | Path

__all__ = [
    "HubModel",
    "ModelCard",
    "from_pretrained",
    "inspect_source",
    "list_hub_families",
    "resolve_source",
]


# ---------------------------------------------------------------------------
# Cards / inspection
# ---------------------------------------------------------------------------


@dataclass
class ModelCard:
    """Lightweight description of a resolved model source."""

    family: str
    pipeline_tag: str | None = None
    architectures: list[str] = field(default_factory=list)
    files: list[str] = field(default_factory=list)
    source: str | None = None
    local_path: str | None = None
    backend: str = "native"  # native | transformers | diffusers | ollama | arrays
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "family": self.family,
            "pipeline_tag": self.pipeline_tag,
            "architectures": list(self.architectures),
            "files": list(self.files),
            "source": self.source,
            "local_path": self.local_path,
            "backend": self.backend,
            "notes": list(self.notes),
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelCard:
        return cls(
            family=str(data.get("family") or "unknown"),
            pipeline_tag=data.get("pipeline_tag"),
            architectures=list(data.get("architectures") or []),
            files=list(data.get("files") or []),
            source=data.get("source"),
            local_path=data.get("local_path"),
            backend=str(data.get("backend") or "native"),
            notes=list(data.get("notes") or []),
        )


def list_hub_families() -> list[str]:
    """Known HubModel family names (routing targets)."""
    return [
        "causal-lm",
        "classifier",
        "reranker",
        "seq2seq",
        "diffusion",
        "video",
        "tts",
        "asr",
        "embedding",
        "zero-shot",
        "fill-mask",
        "question-answering",
        "vlm",
        "gguf",
        "arrays",
        "unknown",
    ]


_CAUSAL_ARCH = re.compile(
    r"(CausalLM|ForCausalLM|GPT|Llama|Mistral|Qwen|Gemma|Phi|Mpt|Falcon|Bloom|Olmo)",
    re.I,
)
_SEQ_CLS_ARCH = re.compile(
    r"(ForSequenceClassification|ForTokenClassification|Classification)",
    re.I,
)
_SEQ2SEQ_ARCH = re.compile(
    r"(ForConditionalGeneration|EncoderDecoder|T5|Bart|M2M100|Marian|Nllb|Pegasus)",
    re.I,
)


def inspect_source(meta: dict[str, Any]) -> ModelCard:
    """Classify a model from config-like metadata (no I/O)."""
    pipeline = meta.get("pipeline_tag") or meta.get("pipeline")
    arches = meta.get("architectures") or []
    if isinstance(arches, str):
        arches = [arches]
    files = [str(f) for f in (meta.get("files") or [])]
    model_index = meta.get("model_index") or {}
    class_name = (
        model_index.get("_class_name")
        or meta.get("_class_name")
        or meta.get("model_type")
    )

    notes: list[str] = []
    lower_files = " ".join(files).lower()

    # File-layout first (most reliable for GGUF / diffusers / MLX).
    if any(f.lower().endswith(".gguf") for f in files) or "gguf" in (pipeline or ""):
        return ModelCard(
            family="gguf",
            pipeline_tag=pipeline or "text-generation",
            architectures=list(arches),
            files=files,
            backend="native",
            notes=["GGUF → native Chatbot via ai.load"],
        )
    if (
        "model_index.json" in files
        or (class_name and "Pipeline" in str(class_name))
        or "unet/" in lower_files
        or "vae/" in lower_files
    ):
        family = "diffusion"
        if (
            pipeline in {"text-to-video", "image-to-video"}
            or "Wan" in str(class_name)
            or "t2v" in lower_files
        ):
            family = "video"
        elif pipeline == "text-to-speech" or "TTS" in str(class_name):
            family = "tts"
        return ModelCard(
            family=family,
            pipeline_tag=pipeline,
            architectures=[str(class_name)] if class_name else list(arches),
            files=files,
            backend="diffusers",
            notes=["Diffusers / pipeline layout"],
        )
    if any("mlx" in f.lower() or f.endswith("-4bit") for f in files) or (
        pipeline == "text-generation" and any("mlx" in str(t).lower() for t in meta.get("tags", []))
    ):
        notes.append("MLX quant — prefer transformers fp fallback or mlx backend")

    # Architecture / pipeline routing.
    arch_blob = " ".join(str(a) for a in arches) + " " + str(class_name or "")
    if pipeline in {"text-classification", "token-classification"} or _SEQ_CLS_ARCH.search(
        arch_blob
    ):
        family = "classifier"
        if "rerank" in " ".join(meta.get("tags", [])).lower() or "rerank" in (
            meta.get("source") or ""
        ).lower():
            family = "reranker"
        return ModelCard(
            family=family,
            pipeline_tag=pipeline or "text-classification",
            architectures=list(arches),
            files=files,
            backend="transformers",
            notes=notes,
        )
    if pipeline in {"translation", "text2text-generation", "summarization"} or _SEQ2SEQ_ARCH.search(
        arch_blob
    ):
        return ModelCard(
            family="seq2seq",
            pipeline_tag=pipeline or "text2text-generation",
            architectures=list(arches),
            files=files,
            backend="transformers",
            notes=notes,
        )
    if pipeline in {"text-to-speech", "automatic-speech-recognition"}:
        return ModelCard(
            family="tts" if pipeline == "text-to-speech" else "asr",
            pipeline_tag=pipeline,
            architectures=list(arches),
            files=files,
            backend="transformers",
            notes=notes or ["speech model — foreign backend"],
        )
    if pipeline in {
        "feature-extraction",
        "sentence-similarity",
    } or "Embedding" in arch_blob:
        return ModelCard(
            family="embedding",
            pipeline_tag=pipeline or "feature-extraction",
            architectures=list(arches),
            files=files,
            backend="transformers",
            notes=notes or ["embedding / sentence-transformers style"],
        )
    if pipeline == "zero-shot-classification":
        return ModelCard(
            family="zero-shot",
            pipeline_tag=pipeline,
            architectures=list(arches),
            files=files,
            backend="transformers",
            notes=notes,
        )
    if pipeline == "fill-mask":
        return ModelCard(
            family="fill-mask",
            pipeline_tag=pipeline,
            architectures=list(arches),
            files=files,
            backend="transformers",
            notes=notes,
        )
    if pipeline == "question-answering":
        return ModelCard(
            family="question-answering",
            pipeline_tag=pipeline,
            architectures=list(arches),
            files=files,
            backend="transformers",
            notes=notes,
        )
    if pipeline in {"image-text-to-text", "any-to-any", "image-to-text"}:
        return ModelCard(
            family="vlm",
            pipeline_tag=pipeline,
            architectures=list(arches),
            files=files,
            backend="transformers",
            notes=notes or ["vision-language model — foreign backend"],
        )
    if pipeline in {"text-to-image", "image-to-image", "image-to-video", "text-to-video"}:
        fam = "video" if "video" in (pipeline or "") else "diffusion"
        return ModelCard(
            family=fam,
            pipeline_tag=pipeline,
            architectures=list(arches),
            files=files,
            backend="diffusers",
            notes=notes,
        )
    if pipeline in {"text-generation", "conversational"} or _CAUSAL_ARCH.search(arch_blob):
        backend = "native"
        if any(f.endswith(".safetensors") for f in files) and not any(
            f.endswith(".gguf") for f in files
        ):
            # External HF causal LMs often need transformers; native import is best-effort.
            backend = "transformers"
            notes.append("causal LM — transformers backend with native Chatbot adapt when possible")
        return ModelCard(
            family="causal-lm",
            pipeline_tag=pipeline or "text-generation",
            architectures=list(arches),
            files=files,
            backend=backend,
            notes=notes,
        )

    # Fallbacks from files alone.
    if any(f.endswith((".safetensors", ".bin", ".pt", ".pth")) for f in files):
        return ModelCard(
            family="arrays",
            pipeline_tag=pipeline,
            architectures=list(arches),
            files=files,
            backend="arrays",
            notes=["generic weight archive — ai.load_arrays / HubModel"],
        )
    return ModelCard(
        family="unknown",
        pipeline_tag=pipeline,
        architectures=list(arches),
        files=files,
        backend="transformers",
        notes=notes or ["unclassified — try transformers AutoModel"],
    )


# ---------------------------------------------------------------------------
# Source resolution (HF / ModelScope / Ollama / local)
# ---------------------------------------------------------------------------


@dataclass
class ResolvedSource:
    kind: str  # local | hf | modelscope | ollama
    spec: str
    local_path: Path | None = None
    repo_id: str | None = None
    filename: str | None = None
    revision: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)


def _parse_spec(source: str) -> tuple[str, str]:
    s = source.strip()
    # ModelScope web URLs (with optional www / models/ path).
    ms_url = re.match(
        r"^https?://(?:www\.)?modelscope\.(?:cn|com)/(?:models/)?(.+?)(?:/summary)?/?$",
        s,
        re.I,
    )
    if ms_url:
        return "modelscope", ms_url.group(1).strip("/")
    for prefix, kind in (
        ("hf://", "hf"),
        ("huggingface://", "hf"),
        ("huggingface.co/", "hf"),
        ("https://huggingface.co/", "hf"),
        ("http://huggingface.co/", "hf"),
        ("ms://", "modelscope"),
        ("modelscope://", "modelscope"),
        ("ollama://", "ollama"),
        ("ollama:", "ollama"),
    ):
        if s.lower().startswith(prefix):
            return kind, s[len(prefix) :].strip("/")
    if os.path.exists(s) or s.endswith((".mmn", ".gguf", ".safetensors", ".bin", ".pt", ".npz")):
        return "local", s
    # org/name → HF by default
    if re.match(r"^[\w.-]+/[\w.-]+$", s):
        return "hf", s
    return "local", s


def _hf_list_files(repo_id: str, revision: str | None = None) -> list[str]:
    from huggingface_hub import list_repo_files

    return list(list_repo_files(repo_id, revision=revision))


def _hf_model_info(repo_id: str, revision: str | None = None) -> dict[str, Any]:
    from huggingface_hub import hf_hub_download, model_info

    info = model_info(repo_id, revision=revision)
    files = _hf_list_files(repo_id, revision=revision)
    meta: dict[str, Any] = {
        "pipeline_tag": getattr(info, "pipeline_tag", None),
        "tags": list(getattr(info, "tags", None) or []),
        "files": files,
        "source": repo_id,
    }
    for name in ("config.json", "model_index.json"):
        if name in files:
            path = hf_hub_download(repo_id, name, revision=revision)
            with open(path, encoding="utf-8") as f:
                cfg = json.load(f)
            if name == "model_index.json":
                meta["model_index"] = cfg
                meta["_class_name"] = cfg.get("_class_name")
            else:
                meta["architectures"] = cfg.get("architectures") or []
                meta["model_type"] = cfg.get("model_type")
                meta["config"] = cfg
            break
    return meta


def _snapshot_hf(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    filename: str | None = None,
    allow_patterns: Sequence[str] | None = None,
) -> Path:
    from huggingface_hub import hf_hub_download, snapshot_download

    if filename:
        path = hf_hub_download(
            repo_id, filename, revision=revision, cache_dir=cache_dir
        )
        return Path(path)

    patterns = list(allow_patterns) if allow_patterns else None
    path = snapshot_download(
        repo_id,
        revision=revision,
        cache_dir=cache_dir,
        allow_patterns=patterns,
    )
    return Path(path)


def _snapshot_modelscope(
    repo_id: str,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
) -> Path:
    try:
        from modelscope.hub.snapshot_download import snapshot_download as ms_download
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "ModelScope support requires the optional 'modelscope' package. "
            "Install with: pip install modelscope"
        ) from e
    kwargs: dict[str, Any] = {"model_id": repo_id}
    if revision:
        kwargs["revision"] = revision
    if cache_dir:
        kwargs["cache_dir"] = cache_dir
    return Path(ms_download(**kwargs))


def _ollama_pull(model: str, host: str | None = None) -> dict[str, Any]:
    """Pull an Ollama model via the local HTTP API (or raise a clear error)."""
    base = (host or os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip(
        "/"
    )
    req = urllib.request.Request(
        f"{base}/api/pull",
        data=json.dumps({"name": model, "stream": False}).encode(),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=600) as resp:
            return json.loads(resp.read().decode())
    except urllib.error.URLError as e:
        raise RuntimeError(
            f"Ollama is not reachable at {base} ({e}). "
            "Start Ollama or use an HF/ModelScope id instead."
        ) from e


def resolve_source(
    source: PathLike,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    filename: str | None = None,
    download: bool = True,
    allow_patterns: Sequence[str] | None = None,
) -> ResolvedSource:
    """Resolve a hub id / URL / local path into a local directory or file."""
    kind, spec = _parse_spec(str(source))
    if kind == "local":
        path = Path(spec).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Local model path not found: {path}")
        meta: dict[str, Any] = {"files": []}
        if path.is_file():
            meta["files"] = [path.name]
        else:
            meta["files"] = [p.name for p in path.rglob("*") if p.is_file()][:200]
            cfg = path / "config.json"
            if cfg.exists():
                with open(cfg, encoding="utf-8") as f:
                    data = json.load(f)
                meta["architectures"] = data.get("architectures") or []
                meta["model_type"] = data.get("model_type")
                meta["config"] = data
            idx = path / "model_index.json"
            if idx.exists():
                with open(idx, encoding="utf-8") as f:
                    meta["model_index"] = json.load(f)
        return ResolvedSource(
            kind="local", spec=spec, local_path=path, meta=meta, filename=filename
        )

    if kind == "ollama":
        info = _ollama_pull(spec) if download else {}
        return ResolvedSource(
            kind="ollama",
            spec=spec,
            repo_id=spec,
            meta={
                "pipeline_tag": "text-generation",
                "architectures": ["OllamaModel"],
                "files": [],
                "ollama": info,
                "tags": ["ollama"],
            },
        )

    if kind == "modelscope":
        meta = {"source": spec, "files": [], "tags": ["modelscope"]}
        local = _snapshot_modelscope(spec, revision=revision, cache_dir=cache_dir) if download else None
        if local and local.exists():
            meta["files"] = [p.name for p in local.rglob("*") if p.is_file()][:200]
            cfg = local / "config.json"
            if cfg.exists():
                with open(cfg, encoding="utf-8") as f:
                    data = json.load(f)
                meta["architectures"] = data.get("architectures") or []
                meta["config"] = data
        return ResolvedSource(
            kind="modelscope",
            spec=spec,
            repo_id=spec,
            local_path=local,
            revision=revision,
            meta=meta,
            filename=filename,
        )

    # Hugging Face Hub
    meta = _hf_model_info(spec, revision=revision)
    local: Path | None = None
    if download:
        # Smart allow_patterns: avoid multi-GB downloads when a single file is enough.
        patterns = allow_patterns
        if filename:
            local = _snapshot_hf(
                spec, revision=revision, cache_dir=cache_dir, filename=filename
            )
        else:
            files = meta.get("files") or []
            if any(f.lower().endswith(".gguf") for f in files) and not patterns:
                # Prefer smallest GGUF when several exist.
                ggufs = sorted(
                    (f for f in files if f.lower().endswith(".gguf")),
                    key=lambda n: (
                        0 if "Q4" in n.upper() or "IQ4" in n.upper() else 1,
                        0 if "Q8" in n.upper() else 2,
                        len(n),
                    ),
                )
                local = _snapshot_hf(
                    spec,
                    revision=revision,
                    cache_dir=cache_dir,
                    filename=ggufs[0],
                )
                meta["chosen_gguf"] = ggufs[0]
            elif "model_index.json" in files and not patterns:
                # Diffusers: download configs first; weights on demand by backend.
                patterns = [
                    "model_index.json",
                    "*/config.json",
                    "*/tokenizer*",
                    "*/vocab*",
                    "*/merges.txt",
                    "scheduler/*",
                    "feature_extractor/*",
                    "*.json",
                ]
                local = _snapshot_hf(
                    spec,
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_patterns=patterns,
                )
                meta["notes"] = ["diffusers configs downloaded; weights lazy-loaded by backend"]
            else:
                # Transformers: prefer safetensors + tokenizer, skip tf/flax/onnx.
                if not patterns:
                    patterns = [
                        "config.json",
                        "generation_config.json",
                        "tokenizer*",
                        "vocab*",
                        "merges.txt",
                        "special_tokens_map.json",
                        "added_tokens.json",
                        "*.safetensors",
                        "*.safetensors.index.json",
                        "model.safetensors",
                        "pytorch_model.bin",
                        "sentencepiece*",
                        "spiece.model",
                        "*.json",
                    ]
                local = _snapshot_hf(
                    spec,
                    revision=revision,
                    cache_dir=cache_dir,
                    allow_patterns=patterns,
                )
    return ResolvedSource(
        kind="hf",
        spec=spec,
        repo_id=spec,
        local_path=local,
        revision=revision,
        meta=meta,
        filename=filename,
    )


# ---------------------------------------------------------------------------
# HubModel — uniform train / finetune / generate / predict
# ---------------------------------------------------------------------------


def _score_from_pipeline_out(out: Any) -> float:
    """Best-effort relevance score from classifier / reranker pipeline output."""
    while isinstance(out, (list, tuple)) and len(out) == 1:
        out = out[0]
    if isinstance(out, (int, float)):
        return float(out)
    if isinstance(out, dict):
        if "score" in out:
            return float(out["score"])
        # logit dict label→score
        vals = [float(v) for v in out.values() if isinstance(v, (int, float))]
        return max(vals) if vals else 0.0
    if isinstance(out, (list, tuple)) and out:
        best = 0.0
        for item in out:
            if isinstance(item, dict) and "score" in item:
                best = max(best, float(item["score"]))
        return best
    return 0.0


class HubModel:
    """Runnable wrapper around a native or foreign model."""

    def __init__(
        self,
        *,
        family: str,
        card: ModelCard,
        native: Any = None,
        foreign: Any = None,
        tokenizer: Any = None,
        source: str | None = None,
        labels: Sequence[str] | None = None,
    ) -> None:
        self.family = family
        self.card = card
        self.native = native
        self.foreign = foreign
        self.tokenizer = tokenizer
        self.source = source or card.source
        self.labels = list(labels or [])

    def __getattr__(self, name: str) -> Any:
        # Proxy native Chatbot/Classifier/Diffusion attributes (vocab_size, …).
        if name.startswith("_"):
            raise AttributeError(name)
        native = object.__getattribute__(self, "native")
        if native is not None and hasattr(native, name):
            return getattr(native, name)
        raise AttributeError(f"{type(self).__name__!r} object has no attribute {name!r}")

    # --- factories for offline tests ---

    @classmethod
    def synthetic_classifier(cls, labels: Sequence[str]) -> HubModel:
        labels = list(labels)
        clf = ai.Classifier.with_labels(labels, 64, seed=0)
        card = ModelCard(family="classifier", pipeline_tag="text-classification")
        return cls(family="classifier", card=card, native=clf, labels=labels, source="synthetic")

    @classmethod
    def synthetic_causal_lm(
        cls, vocab_size: int = 256, max_seq_len: int = 64
    ) -> HubModel:
        bot = ai.Chatbot(
            vocab_size=vocab_size,
            n_layer=1,
            d_model=32,
            seed=0,
            max_seq_len=max_seq_len,
            use_rope=True,
        )
        card = ModelCard(family="causal-lm", pipeline_tag="text-generation")
        return cls(family="causal-lm", card=card, native=bot, source="synthetic")

    @classmethod
    def synthetic_diffusion(cls) -> HubModel:
        diff = ai.Diffusion()
        card = ModelCard(family="diffusion", pipeline_tag="text-to-image", backend="native")
        return cls(family="diffusion", card=card, native=diff, source="synthetic")

    @classmethod
    def synthetic_reranker(cls) -> HubModel:
        """Offline reranker stub: scores by simple lexical overlap (not neural)."""

        class _LexRerank:
            def score(self, query: str, doc: str) -> float:
                q = set(query.lower().split())
                d = set(doc.lower().split())
                if not q:
                    return 0.0
                return float(len(q & d) / len(q))

            def __call__(self, text: str, **kwargs: Any) -> list[dict[str, Any]]:
                # Single-string pipeline shape for predict().
                return [{"label": "LABEL_0", "score": float(len(text))}]

        card = ModelCard(family="reranker", pipeline_tag="text-classification")
        return cls(
            family="reranker",
            card=card,
            foreign=_LexRerank(),
            source="synthetic",
        )

    @classmethod
    def synthetic_seq2seq(cls) -> HubModel:
        bot = ai.Chatbot(vocab_size=128, n_layer=1, d_model=32, seed=0, max_seq_len=64)
        card = ModelCard(family="seq2seq", pipeline_tag="translation", backend="native")
        return cls(family="seq2seq", card=card, native=bot, source="synthetic")

    def capabilities(self) -> dict[str, bool]:
        """What this HubModel can do without probing the backend."""
        has_native = self.native is not None
        has_foreign = self.foreign is not None
        ollama = self.card.backend == "ollama"
        gen = False
        if has_native and (
            hasattr(self.native, "generate")
            or hasattr(self.native, "chat")
            or hasattr(self.native, "sample_rgb_patch")
        ):
            gen = True
        if has_foreign or ollama:
            gen = True
        if self.family == "video" and not has_foreign:
            gen = False
        if self.family == "gguf" and not has_native and not has_foreign:
            gen = False
        pred = has_native and (
            hasattr(self.native, "predict") or hasattr(self.native, "predict_label")
        )
        if has_foreign or self.family in {"classifier", "reranker", "asr", "embedding"}:
            pred = pred or has_foreign
        ft = has_native and hasattr(self.native, "train")
        if has_foreign and self.family in {
            "classifier",
            "reranker",
            "causal-lm",
            "seq-cls",
            "text-classification",
        }:
            ft = True
        return {
            "generate": gen,
            "predict": bool(pred),
            "finetune": bool(ft),
            "chat": gen or ollama,
            "native": has_native,
            "foreign": has_foreign,
        }

    # --- inference ---

    def generate(self, prompt: str, **kwargs: Any) -> Any:
        if self.family == "video" and self.foreign is None:
            raise RuntimeError(
                "video generate requires a loaded Diffusers/Wan pipeline "
                "(multi-GB weights). Route-only HubModel shells cannot sample frames."
            )
        if self.family == "gguf" and self.native is None and self.foreign is None:
            raise RuntimeError(
                "GGUF native load failed and no foreign backend is available. "
                f"notes={self.card.notes}"
            )
        if self.native is not None and hasattr(self.native, "generate"):
            return self.native.generate(prompt, **kwargs)
        if self.native is not None and hasattr(self.native, "chat"):
            return self.native.chat(prompt, **kwargs)
        if self.native is not None and hasattr(self.native, "sample_rgb_patch"):
            steps = int(kwargs.get("steps", kwargs.get("num_inference_steps", 1)))
            return self.native.sample_rgb_patch(steps=steps)
        if self.foreign is not None:
            return self._foreign_generate(prompt, **kwargs)
        if self.card.backend == "ollama":
            return self._ollama_generate(prompt, **kwargs)
        raise RuntimeError(f"generate() not available for family={self.family}")

    def chat(self, prompt: str, **kwargs: Any) -> str:
        if self.card.backend == "ollama":
            return self._ollama_chat(prompt, **kwargs)
        if self.native is not None and hasattr(self.native, "chat"):
            return self.native.chat(prompt, **kwargs)
        return str(self.generate(prompt, **kwargs))

    def predict(self, text: str, **kwargs: Any) -> Any:
        if self.native is not None and hasattr(self.native, "predict"):
            return self.native.predict(text, **kwargs)
        if self.native is not None and hasattr(self.native, "predict_label"):
            return self.native.predict_label(text)
        if self.foreign is not None:
            return self._foreign_predict(text, **kwargs)
        raise RuntimeError(f"predict() not available for family={self.family}")

    def score_pairs(self, query: str, documents: Sequence[str], **kwargs: Any) -> list[float]:
        """Score (query, document) pairs for rerankers / cross-encoders."""
        docs = [str(d) for d in documents]
        foreign = self.foreign
        if foreign is not None and hasattr(foreign, "score"):
            return [float(foreign.score(query, d)) for d in docs]
        if foreign is not None and callable(foreign):
            scores: list[float] = []
            for doc in docs:
                out = foreign({"text": query, "text_pair": doc}, **kwargs)
                # Also try tuple / string pair conventions.
                if out is None:
                    out = foreign(f"query: {query} document: {doc}", **kwargs)
                scores.append(_score_from_pipeline_out(out))
            return scores
        if self.native is not None and hasattr(self.native, "predict"):
            # Native classifier fallback: score label confidence on concatenated pair.
            scores = []
            for doc in docs:
                out = self.native.predict(f"{query} [SEP] {doc}")
                scores.append(_score_from_pipeline_out(out))
            return scores
        raise RuntimeError(f"score_pairs() not available for family={self.family}")

    def rerank(
        self, query: str, documents: Sequence[str], *, top_k: int | None = None, **kwargs: Any
    ) -> list[tuple[str, float]]:
        docs = list(documents)
        scores = self.score_pairs(query, docs, **kwargs)
        ranked = sorted(zip(docs, scores, strict=True), key=lambda x: x[1], reverse=True)
        if top_k is not None:
            ranked = ranked[: max(0, int(top_k))]
        return ranked

    def embed(self, texts: str | Sequence[str], **kwargs: Any) -> Any:
        """Embedding / feature-extraction models."""
        batch = [texts] if isinstance(texts, str) else list(texts)
        foreign = self.foreign
        if foreign is not None and callable(foreign):
            return foreign(batch, **kwargs)
        if self.tokenizer is not None and hasattr(foreign, "forward"):
            import torch

            enc = self.tokenizer(
                batch, padding=True, truncation=True, return_tensors="pt"
            )
            with torch.no_grad():
                out = foreign(**enc)
            if hasattr(out, "last_hidden_state"):
                return out.last_hidden_state.mean(dim=1).cpu().tolist()
            return out
        raise RuntimeError(f"embed() not available for family={self.family}")

    def predict_label(self, text: str) -> str:
        out = self.predict(text)
        # Unwrap pipeline quirks: [[{label, score}, ...]] or [{label, score}, ...]
        while isinstance(out, (list, tuple)) and len(out) == 1:
            out = out[0]
        if isinstance(out, str):
            return out
        if isinstance(out, dict):
            if "label" in out:
                return str(out["label"])
            return max(out, key=out.get)
        if isinstance(out, (list, tuple)) and out:
            # score-sorted list of {label, score}
            best = None
            best_score = float("-inf")
            for item in out:
                if isinstance(item, dict) and "label" in item:
                    score = float(item.get("score", 0.0))
                    if score >= best_score:
                        best_score = score
                        best = str(item["label"])
            if best is not None:
                return best
        return str(out)

    # --- training ---

    def train(self, dataset: Any, **kwargs: Any) -> list[float]:
        return self.finetune(dataset, **kwargs)

    def finetune(self, dataset: Any, **kwargs: Any) -> list[float]:
        epochs = int(kwargs.pop("epochs", 1))
        lr = float(kwargs.pop("learning_rate", kwargs.pop("lr", 3e-4)))
        batch_size = int(kwargs.pop("batch_size", 8))
        optimizer = kwargs.pop("optimizer", "adamw")
        verbose = bool(kwargs.pop("verbose", False))

        if self.native is not None and hasattr(self.native, "train"):
            return list(
                self.native.train(
                    dataset,
                    epochs=epochs,
                    learning_rate=lr,
                    batch_size=batch_size,
                    optimizer=optimizer,
                    verbose=verbose,
                    **kwargs,
                )
            )
        if self.foreign is not None:
            return self._foreign_finetune(
                dataset,
                epochs=epochs,
                learning_rate=lr,
                batch_size=batch_size,
                verbose=verbose,
                **kwargs,
            )
        raise RuntimeError(
            f"finetune() not available for family={self.family} backend={self.card.backend}"
        )

    def save(self, path: PathLike, **kwargs: Any) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if self.native is not None and hasattr(self.native, "save"):
            self.native.save(str(path), **kwargs)
            return
        if self.foreign is not None and hasattr(self.foreign, "save_pretrained"):
            self.foreign.save_pretrained(str(path))
            if self.tokenizer is not None and hasattr(self.tokenizer, "save_pretrained"):
                self.tokenizer.save_pretrained(str(path))
            return
        raise RuntimeError("save() not available for this HubModel")

    def to_native(self) -> Any:
        """Best-effort unwrap to Chatbot / Classifier / Diffusion."""
        if self.native is not None:
            return self.native
        raise RuntimeError("No native MagicMindNet model available for this hub source")

    # --- foreign backends ---

    def _foreign_generate(self, prompt: str, **kwargs: Any) -> Any:
        max_new = int(kwargs.get("max_new_tokens", 32))
        foreign = self.foreign
        # Diffusers text-to-image / inpaint / video pipelines.
        if self.family in {"diffusion", "video"} or (
            hasattr(foreign, "__class__")
            and "Pipeline" in foreign.__class__.__name__
            and self.family != "classifier"
        ):
            call_kwargs = dict(kwargs)
            call_kwargs.pop("max_new_tokens", None)
            call_kwargs.pop("temperature", None)
            if "num_inference_steps" not in call_kwargs and "steps" in call_kwargs:
                call_kwargs["num_inference_steps"] = call_kwargs.pop("steps")
            try:
                out = foreign(prompt, **call_kwargs)
            except TypeError:
                out = foreign(prompt)
            if hasattr(out, "images"):
                imgs = out.images
                return imgs[0] if imgs else out
            if hasattr(out, "frames"):
                return out.frames
            if isinstance(out, dict) and "images" in out:
                imgs = out["images"]
                return imgs[0] if imgs else out
            return out
        # transformers text-generation pipeline or model+tokenizer
        if callable(foreign) and foreign.__class__.__name__.endswith("Pipeline"):
            pipe_kwargs: dict[str, Any] = {}
            # TTS / ASR / text2text pipelines take different kwargs.
            name = foreign.__class__.__name__.lower()
            if "textgeneration" in name.replace("_", "") or "text2text" in name:
                pipe_kwargs["max_new_tokens"] = max_new
                pipe_kwargs["do_sample"] = kwargs.get("temperature", 0) > 0
            try:
                out = foreign(prompt, **pipe_kwargs)
            except TypeError:
                out = foreign(prompt)
            if isinstance(out, list) and out:
                item = out[0]
                if isinstance(item, dict):
                    return str(
                        item.get("generated_text")
                        or item.get("translation_text")
                        or item.get("text")
                        or item.get("audio")
                        or item
                    )
            return out if not isinstance(out, str) else out
        if self.tokenizer is not None and hasattr(foreign, "generate"):
            import torch

            tok_kwargs: dict[str, Any] = {"return_tensors": "pt"}
            # NLLB / M2M style forced BOS language id when provided.
            src_lang = kwargs.get("src_lang")
            tgt_lang = kwargs.get("tgt_lang")
            if src_lang and hasattr(self.tokenizer, "src_lang"):
                self.tokenizer.src_lang = src_lang
            toks = self.tokenizer(prompt, **tok_kwargs)
            gen_kwargs: dict[str, Any] = {"max_new_tokens": max_new}
            # Avoid transformers warning when both max_length and max_new_tokens set.
            if hasattr(foreign, "generation_config") and getattr(
                foreign.generation_config, "max_length", None
            ):
                foreign.generation_config.max_length = None
            if tgt_lang and hasattr(self.tokenizer, "lang_code_to_id"):
                lang_id = self.tokenizer.lang_code_to_id.get(tgt_lang)
                if lang_id is not None:
                    gen_kwargs["forced_bos_token_id"] = lang_id
            with torch.no_grad():
                ids = foreign.generate(**toks, **gen_kwargs)
            return self.tokenizer.decode(ids[0], skip_special_tokens=True)
        raise RuntimeError("foreign generate backend not recognized")

    def _foreign_predict(self, text: str, **kwargs: Any) -> Any:
        foreign = self.foreign
        if callable(foreign):
            return foreign(text, **kwargs)
        raise RuntimeError("foreign predict backend not recognized")

    def _foreign_finetune(
        self,
        dataset: Any,
        *,
        epochs: int,
        learning_rate: float,
        batch_size: int,
        verbose: bool = False,
        **kwargs: Any,
    ) -> list[float]:
        """Minimal transformers Trainer finetune for seq-cls / causal LM."""
        try:
            import torch
            from torch.utils.data import Dataset as TorchDataset
            from transformers import Trainer, TrainingArguments
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "Foreign finetune requires transformers + torch. "
                "Install with: pip install transformers torch"
            ) from e

        model = self.foreign
        tokenizer = self.tokenizer
        pipe = None
        # Unwrap HF pipelines to the underlying nn.Module + tokenizer.
        if hasattr(model, "model") and hasattr(model, "tokenizer"):
            pipe = model
            if tokenizer is None:
                tokenizer = model.tokenizer
            model = model.model
        if tokenizer is None:
            raise RuntimeError("foreign finetune needs a tokenizer")
        if self.tokenizer is None:
            self.tokenizer = tokenizer

        class _Wrap(TorchDataset):
            def __init__(self, rows: list[dict[str, Any]]):
                self.rows = rows

            def __len__(self) -> int:
                return len(self.rows)

            def __getitem__(self, idx: int) -> dict[str, Any]:
                return self.rows[idx]

        rows: list[dict[str, Any]] = []
        pairs: list[tuple[str, str]] = []
        texts: list[str] = []

        if hasattr(dataset, "as_pairs"):
            pairs = list(dataset.as_pairs())
        elif isinstance(dataset, (list, tuple)):
            for item in dataset:
                if isinstance(item, dict):
                    text = item.get("text") or item.get("input") or ""
                    label = item.get("label") or item.get("tag") or item.get("output")
                    if label is not None:
                        pairs.append((str(text), str(label)))
                    else:
                        texts.append(str(text))
                else:
                    texts.append(str(item))

        if hasattr(dataset, "as_texts"):
            texts = list(dataset.as_texts())
        # DatasetQA.as_pairs → causal LM prompt+completion rows.
        if (
            not texts
            and pairs
            and self.family
            in {"causal-lm", "seq2seq", "gguf", "chatbot", "llm"}
        ):
            for inp, out in pairs:
                texts.append(f"{inp}\n{out}")
            pairs = []
        if pairs and self.family in {
            "classifier",
            "reranker",
            "seq-cls",
            "text-classification",
        }:
            label2id: dict[str, int] = {}
            if self.labels:
                label2id = {str(lab): i for i, lab in enumerate(self.labels)}
            elif hasattr(model, "config"):
                raw = getattr(model.config, "label2id", None) or {}
                label2id = {str(k): int(v) for k, v in raw.items()}
            if not label2id and hasattr(model, "config"):
                id2label = getattr(model.config, "id2label", None) or {}
                if id2label:
                    label2id = {str(v): int(k) for k, v in id2label.items()}
                    self.labels = [
                        str(id2label[i]) for i in sorted(id2label, key=lambda x: int(x))
                    ]
            for text, tag in pairs:
                tag_s = str(tag)
                if tag_s not in label2id:
                    lower_map = {k.lower(): v for k, v in label2id.items()}
                    if tag_s.lower() not in lower_map:
                        continue
                    lab = int(lower_map[tag_s.lower()])
                else:
                    lab = int(label2id[tag_s])
                enc = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=int(kwargs.get("max_length", 64)),
                    return_tensors="pt",
                )
                rows.append(
                    {
                        "input_ids": enc["input_ids"].squeeze(0),
                        "attention_mask": enc["attention_mask"].squeeze(0),
                        "labels": torch.tensor(lab),
                    }
                )
        else:
            if hasattr(dataset, "as_texts"):
                texts = list(dataset.as_texts())
            for text in texts:
                enc = tokenizer(
                    text,
                    truncation=True,
                    padding="max_length",
                    max_length=int(kwargs.get("max_length", 64)),
                    return_tensors="pt",
                )
                ids = enc["input_ids"].squeeze(0)
                rows.append(
                    {
                        "input_ids": ids,
                        "attention_mask": enc["attention_mask"].squeeze(0),
                        "labels": ids.clone(),
                    }
                )
        if not rows:
            raise ValueError("No trainable rows after label/tokenization filtering")

        args = TrainingArguments(
            output_dir=str(kwargs.get("output_dir", "/tmp/mmn_hub_ft")),
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            learning_rate=learning_rate,
            logging_steps=1 if verbose else max(1, len(rows) // max(batch_size, 1)),
            disable_tqdm=not verbose,
            report_to=[],
            save_strategy="no",
            remove_unused_columns=False,
        )
        trainer = Trainer(model=model, args=args, train_dataset=_Wrap(rows))
        result = trainer.train()
        # Keep pipeline wrapper callable for predict/generate when we unwrapped one.
        if pipe is not None:
            pipe.model = model
            self.foreign = pipe
        else:
            self.foreign = model
        loss = float(getattr(result, "training_loss", 0.0) or 0.0)
        return [loss]

    def _ollama_generate(self, prompt: str, **kwargs: Any) -> str:
        base = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        model_name = (self.source or self.card.source or "").removeprefix("ollama://").removeprefix(
            "ollama:"
        )
        payload = {
            "model": model_name,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": float(kwargs.get("temperature", 0.7)),
                "num_predict": int(kwargs.get("max_new_tokens", 64)),
            },
        }
        if kwargs.get("system"):
            payload["system"] = str(kwargs["system"])
        req = urllib.request.Request(
            f"{base}/api/generate",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode())
        return str(data.get("response", ""))

    def _ollama_chat(self, prompt: str, **kwargs: Any) -> str:
        base = (os.environ.get("OLLAMA_HOST") or "http://127.0.0.1:11434").rstrip("/")
        model_name = (self.source or self.card.source or "").removeprefix("ollama://").removeprefix(
            "ollama:"
        )
        messages: list[dict[str, str]] = []
        if kwargs.get("system"):
            messages.append({"role": "system", "content": str(kwargs["system"])})
        if kwargs.get("messages"):
            messages.extend(list(kwargs["messages"]))
        else:
            messages.append({"role": "user", "content": prompt})
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": float(kwargs.get("temperature", 0.7)),
                "num_predict": int(kwargs.get("max_new_tokens", 64)),
            },
        }
        req = urllib.request.Request(
            f"{base}/api/chat",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=600) as resp:
            data = json.loads(resp.read().decode())
        msg = data.get("message") or {}
        if isinstance(msg, dict) and "content" in msg:
            return str(msg["content"])
        return str(data.get("response", msg))

    def __repr__(self) -> str:
        return (
            f"HubModel(family={self.family!r}, source={self.source!r}, "
            f"backend={self.card.backend!r}, native={self.native is not None})"
        )


# ---------------------------------------------------------------------------
# from_pretrained
# ---------------------------------------------------------------------------


def _try_native_load(path: Path) -> Any:
    """Load a MagicMindNet-native checkpoint (any supported format)."""
    try:
        return ai.load(str(path))
    except Exception:
        pass
    if path.is_dir():
        # Prefer GGUF / safetensors / mmn inside the directory.
        for pattern in ("*.gguf", "*.mmn", "*.safetensors", "*.bin", "*.pt", "*.npz"):
            hits = sorted(path.rglob(pattern))
            for hit in hits:
                try:
                    return ai.load(str(hit))
                except Exception:
                    continue
    return None


def _load_transformers_model(
    local_or_id: str, card: ModelCard
) -> tuple[Any, Any, list[str] | None]:
    from transformers import (
        AutoModel,
        AutoModelForCausalLM,
        AutoModelForQuestionAnswering,
        AutoModelForSeq2SeqLM,
        AutoModelForSequenceClassification,
        AutoTokenizer,
        pipeline,
    )

    tokenizer = AutoTokenizer.from_pretrained(local_or_id, trust_remote_code=True)
    labels: list[str] | None = None
    if card.family in {"classifier", "reranker", "seq-cls"}:
        model = AutoModelForSequenceClassification.from_pretrained(
            local_or_id, trust_remote_code=True
        )
        id2label = getattr(model.config, "id2label", None) or {}
        if id2label:
            labels = [id2label[i] for i in sorted(id2label, key=lambda x: int(x))]
        pipe = pipeline(
            "text-classification",
            model=model,
            tokenizer=tokenizer,
            top_k=None,
        )
        return pipe, tokenizer, labels
    if card.family == "seq2seq":
        model = AutoModelForSeq2SeqLM.from_pretrained(local_or_id, trust_remote_code=True)
        return model, tokenizer, None
    if card.family == "tts":
        try:
            pipe = pipeline("text-to-speech", model=local_or_id)
            return pipe, getattr(pipe, "tokenizer", tokenizer), None
        except Exception:
            # Kokoro and friends may need trust_remote_code / custom code.
            pipe = pipeline(
                "text-to-speech", model=local_or_id, trust_remote_code=True
            )
            return pipe, getattr(pipe, "tokenizer", tokenizer), None
    if card.family == "asr":
        pipe = pipeline(
            "automatic-speech-recognition",
            model=local_or_id,
            trust_remote_code=True,
        )
        return pipe, getattr(pipe, "tokenizer", tokenizer), None
    if card.family == "embedding":
        model = AutoModel.from_pretrained(local_or_id, trust_remote_code=True)
        pipe = pipeline(
            "feature-extraction",
            model=model,
            tokenizer=tokenizer,
        )
        return pipe, tokenizer, None
    if card.family == "zero-shot":
        pipe = pipeline(
            "zero-shot-classification",
            model=local_or_id,
            trust_remote_code=True,
        )
        return pipe, getattr(pipe, "tokenizer", tokenizer), None
    if card.family == "fill-mask":
        pipe = pipeline("fill-mask", model=local_or_id, trust_remote_code=True)
        return pipe, getattr(pipe, "tokenizer", tokenizer), None
    if card.family == "question-answering":
        model = AutoModelForQuestionAnswering.from_pretrained(
            local_or_id, trust_remote_code=True
        )
        pipe = pipeline(
            "question-answering", model=model, tokenizer=tokenizer
        )
        return pipe, tokenizer, None
    # causal LM / vlm default
    model = AutoModelForCausalLM.from_pretrained(
        local_or_id, trust_remote_code=True, torch_dtype="auto"
    )
    return model, tokenizer, None


def _load_diffusers_pipeline(local_or_id: str, card: ModelCard) -> Any:
    import diffusers

    cls_name = None
    if card.architectures:
        cls_name = card.architectures[0]
    # Map common pipelines.
    mapping = {
        "StableDiffusionPipeline": "StableDiffusionPipeline",
        "StableDiffusionInpaintPipeline": "StableDiffusionInpaintPipeline",
        "WanPipeline": "DiffusionPipeline",
    }
    attr = mapping.get(cls_name or "", "DiffusionPipeline")
    pipe_cls = getattr(diffusers, attr, None) or diffusers.DiffusionPipeline
    return pipe_cls.from_pretrained(local_or_id, trust_remote_code=True)


def from_pretrained(
    source: PathLike,
    *,
    revision: str | None = None,
    cache_dir: str | None = None,
    filename: str | None = None,
    download: bool = True,
    trust_remote_code: bool = True,
    prefer_native: bool = True,
    allow_patterns: Sequence[str] | None = None,
    **kwargs: Any,
) -> HubModel | Any:
    """Load any model from Hugging Face Hub, ModelScope, Ollama, or a local path.

    Returns a native MagicMindNet model when possible, otherwise a :class:`HubModel`
    with ``generate`` / ``predict`` / ``finetune`` / ``train`` / ``save``.
    """
    del trust_remote_code  # reserved for future kwargs plumbing
    resolved = resolve_source(
        source,
        revision=revision,
        cache_dir=cache_dir,
        filename=filename,
        download=download,
        allow_patterns=allow_patterns,
    )
    meta = dict(resolved.meta)
    meta.setdefault("source", resolved.spec)
    card = inspect_source(meta)
    card.source = resolved.spec
    card.local_path = str(resolved.local_path) if resolved.local_path else None

    # Ollama: no local weights — HubModel talks to the daemon.
    if resolved.kind == "ollama":
        card.backend = "ollama"
        card.family = "causal-lm"
        return HubModel(family="causal-lm", card=card, source=resolved.spec)

    local = resolved.local_path
    if local is None:
        raise RuntimeError(f"Failed to resolve local path for {source!r}")

    # 1) Native MagicMindNet checkpoint / GGUF / interop.
    if prefer_native:
        native = _try_native_load(local)
        if native is not None:
            fam = card.family
            if native.__class__.__name__ == "Chatbot":
                fam = "causal-lm"
            elif native.__class__.__name__ == "Classifier":
                fam = "classifier"
            elif native.__class__.__name__ == "Diffusion":
                fam = "diffusion"
            card.backend = "native"
            return HubModel(
                family=fam,
                card=card,
                native=native,
                source=resolved.spec,
                labels=list(getattr(native, "labels", []) or [])
                if hasattr(native, "labels")
                else None,
            )

    # 2) Foreign backends by family.
    target = str(local)
    # If we downloaded a single file (GGUF), prefer its parent for transformers.
    if local.is_file() and local.suffix.lower() == ".gguf":
        raise RuntimeError(
            f"GGUF native Chatbot load failed for {local}. "
            f"notes={card.notes}. Ensure the file is a supported GGUF "
            "(Llama/Qwen-style) or pass prefer_native=False with a foreign backend."
        )

    if card.backend == "diffusers" or card.family in {"diffusion", "video"}:
        try:
            pipe = _load_diffusers_pipeline(resolved.repo_id or target, card)
            return HubModel(
                family=card.family,
                card=card,
                foreign=pipe,
                source=resolved.spec,
            )
        except Exception as e:
            card.notes.append(f"diffusers load deferred: {e}")
            # Return a card-only HubModel that can still finetune native Diffusion toys
            # or be retried after `pip install diffusers`.
            return HubModel(family=card.family, card=card, source=resolved.spec)

    if card.family in {
        "causal-lm",
        "classifier",
        "reranker",
        "seq2seq",
        "seq-cls",
        "tts",
        "asr",
        "embedding",
        "zero-shot",
        "fill-mask",
        "question-answering",
        "vlm",
        "unknown",
        "arrays",
    }:
        try:
            foreign, tok, labels = _load_transformers_model(
                resolved.repo_id or target, card
            )
            return HubModel(
                family=card.family if card.family != "unknown" else "causal-lm",
                card=card,
                foreign=foreign,
                tokenizer=tok,
                source=resolved.spec,
                labels=labels,
            )
        except Exception as e:
            card.notes.append(f"transformers load failed: {e}")
            # Last resort: arrays
            try:
                if local.is_file():
                    arrays = ai.load_arrays(str(local))
                else:
                    arrays = None
                    for hit in sorted(local.rglob("*.safetensors"))[:1]:
                        arrays = ai.load_arrays(str(hit))
                        break
                if arrays is not None:
                    card.backend = "arrays"
                    return HubModel(
                        family="arrays",
                        card=card,
                        foreign=arrays,
                        source=resolved.spec,
                    )
            except Exception as e2:
                card.notes.append(f"arrays load failed: {e2}")
            raise RuntimeError(
                f"Could not load {source!r} as native or foreign model. "
                f"family={card.family} notes={card.notes}"
            ) from e

    raise RuntimeError(f"Unhandled model family {card.family!r} for {source!r}")
