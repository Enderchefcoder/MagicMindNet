"""Load BPE tokenizer sidecar referenced from chatbot checkpoint meta."""

from __future__ import annotations

import json
from pathlib import Path

from magicmindnet._native import BytePairEncoder


def load_bpe_sidecar(checkpoint_path: str | Path) -> BytePairEncoder | None:
    """Load `mmn-bpe-v1` sidecar when `meta.bpe_checkpoint` is set on a chatbot export."""
    path = Path(checkpoint_path)
    wrapper = json.loads(path.read_text(encoding="utf-8"))
    rel = wrapper.get("meta", {}).get("bpe_checkpoint")
    if not rel:
        return None
    sidecar = path.parent / str(rel)
    return BytePairEncoder.load(str(sidecar))
