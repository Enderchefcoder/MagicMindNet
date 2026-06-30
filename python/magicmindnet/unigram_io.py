"""Load unigram tokenizer sidecar referenced from chatbot checkpoint meta."""

from __future__ import annotations

import json
from pathlib import Path

from magicmindnet._native import UnigramEncoder


def load_unigram_sidecar(checkpoint_path: str | Path) -> UnigramEncoder | None:
    """Load `mmn-unigram-v1` sidecar when `meta.unigram_checkpoint` is set on a chatbot export."""
    path = Path(checkpoint_path)
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    rel = wrapper.get("meta", {}).get("unigram_checkpoint")
    if not rel:
        return None
    sidecar = path.parent / str(rel)
    return UnigramEncoder.load(str(sidecar))
