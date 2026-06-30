"""Train diffusion, export checkpoint, reload and sample."""

from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "fixtures" / "image_gen.json"
OUT = ROOT / "examples" / "_roundtrip_diffusion.mmn"


def main() -> None:
    ds = ai.DatasetImageGen(file=str(MANIFEST))
    d = ai.Diffusion()
    cfg = ai.TrainConfig(epochs=4, learning_rate=0.05, cuda=False)
    ai.TrainDiffusion(d, ds, cfg)
    before = d.sample_rgb_patch(steps=6, seed=1)
    ai.export_diffusion(d, "safetensors", str(OUT))
    loaded = ai.import_diffusion("safetensors", [str(OUT)])
    after = loaded.sample_rgb_patch(steps=6, seed=1)
    assert before == after
    print(f"diffusion roundtrip ok: {OUT.name} patch[0]={after[0]:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
