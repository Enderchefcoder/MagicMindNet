"""Vision patch helpers for multimodal Chatbot models."""

VISION_PATCH_DIM = 64
VISION_RGB_DIM = VISION_PATCH_DIM * 3
VISION_RGB_SPATIAL = 8
VISION_RGB_CHANNELS = 3


def vision_patch_from_text(text: str) -> list[float]:
    """Build a normalized 8×8 grayscale patch from UTF-8 bytes (pads with zeros)."""
    raw = text.encode("utf-8")[:VISION_PATCH_DIM]
    return [b / 255.0 for b in raw] + [0.0] * (VISION_PATCH_DIM - len(raw))


def vision_rgb_patch_from_text(text: str) -> list[float]:
    """Build a normalized 8×8×3 RGB patch (NCHW planes) from UTF-8 bytes."""
    raw = text.encode("utf-8")
    if not raw:
        return [0.0] * VISION_RGB_DIM
    out = [0.0] * VISION_RGB_DIM
    for idx in range(VISION_PATCH_DIM):
        b0 = raw[idx % len(raw)]
        b1 = raw[(idx + 1) % len(raw)]
        b2 = raw[(idx + 2) % len(raw)]
        out[idx] = b0 / 255.0
        out[VISION_PATCH_DIM + idx] = b1 / 255.0
        out[2 * VISION_PATCH_DIM + idx] = b2 / 255.0
    return out
