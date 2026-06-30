"""Inpaint sample demo using image_edit fixtures."""

from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures" / "samples"
OUT = ROOT / "examples" / "_inpaint_sample.png"


def main() -> None:
    d = ai.Diffusion()
    patch = d.sample_inpaint_rgb_patch(
        str(FIXTURES / "photo.png"),
        str(FIXTURES / "mask.png"),
        steps=6,
        seed=3,
    )
    d.sample_inpaint_rgb_patch_to_png(
        str(OUT),
        str(FIXTURES / "photo.png"),
        str(FIXTURES / "mask.png"),
        steps=6,
        seed=3,
    )
    print(
        f"inpaint patch len={len(patch)} min={min(patch):.4f} max={max(patch):.4f} png={OUT.name}"
    )


if __name__ == "__main__":
    main()
