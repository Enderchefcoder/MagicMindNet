"""Vision patch helpers for multimodal Chatbot models."""

VISION_PATCH_DIM = 64


def vision_patch_from_text(text: str) -> list[float]:
    """Build a normalized 8×8 grayscale patch from UTF-8 bytes (pads with zeros)."""
    raw = text.encode("utf-8")[:VISION_PATCH_DIM]
    return [b / 255.0 for b in raw] + [0.0] * (VISION_PATCH_DIM - len(raw))
