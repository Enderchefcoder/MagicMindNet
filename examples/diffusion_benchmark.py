"""Mean denoise loss before/after TrainDiffusion on fixture image manifests."""

import sys
from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "tests" / "fixtures"


def main() -> None:
    use_edit = "--edit" in sys.argv[1:]
    if use_edit:
        ds = ai.DatasetImageEdit(file=str(FIXTURES / "image_edit.json"))
        label = "image_edit (inpainting)"
        t = 5
    else:
        ds = ai.DatasetImageGen(file=str(FIXTURES / "image_gen.json"))
        label = "image_gen"
        t = 7
    d = ai.Diffusion()
    before = d.compute_mean_denoise_loss(ds, t=t)
    cfg = ai.TrainConfig(epochs=8, batch_size=1, learning_rate=0.05, cuda=False)
    print(f"Training Diffusion on {label} …")
    print(f"  mean denoise loss before: {before:.4f}")
    ai.TrainDiffusion(d, ds, cfg)
    after = d.compute_mean_denoise_loss(ds, t=t)
    print(f"  mean denoise loss after:  {after:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
