"""Inpaint sample demo using DatasetImageEdit manifest path resolvers."""

from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "fixtures" / "image_edit.json"
OUT = ROOT / "examples" / "_inpaint_sample.png"


def main() -> None:
    ds = ai.DatasetImageEdit(file=str(MANIFEST))
    image_path = ds.image_path_at(0)
    mask_path = ds.mask_path_at(0)
    print(f"prompt: {ds.prompt_at(0)!r}")
    d = ai.Diffusion()
    patch = d.sample_inpaint_rgb_patch(image_path, mask_path, steps=6, seed=3)
    d.sample_inpaint_rgb_patch_to_png(str(OUT), image_path, mask_path, steps=6, seed=3)
    print(
        f"inpaint patch len={len(patch)} min={min(patch):.4f} max={max(patch):.4f} png={OUT.name}"
    )


if __name__ == "__main__":
    main()
