"""Train Diffusion UNet on an image_gen manifest (foundation demo)."""

from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "fixtures" / "image_gen.json"


def main() -> None:
    ds = ai.DatasetImageGen(file=str(MANIFEST))
    img = ROOT / "tests" / "fixtures" / "samples" / "cat.png"
    d = ai.Diffusion()
    before = d.denoise_loss_on_image(str(img), 5)
    cfg = ai.TrainConfig(epochs=6, learning_rate=0.05, cuda=False)
    ai.TrainDiffusion(d, ds, cfg)
    after = d.denoise_loss_on_image(str(img), 5)
    print(f"denoise loss: {before:.6f} -> {after:.6f}")
    assert after <= before * 1.1 + 1e-4
    print("Done.")


if __name__ == "__main__":
    main()
