"""Train inpainting diffusion on an image_edit manifest."""

from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "fixtures" / "image_edit.json"


def main() -> None:
    ds = ai.DatasetImageEdit(file=str(MANIFEST))
    d = ai.Diffusion()
    cfg = ai.TrainConfig(epochs=4, batch_size=1, learning_rate=0.05, cuda=False)
    ai.TrainDiffusion(d, ds, cfg)
    patch = d.sample_rgb_patch(steps=6, seed=1)
    print(f"sample_rgb_patch len={len(patch)} min={min(patch):.4f} max={max(patch):.4f}")


if __name__ == "__main__":
    main()
