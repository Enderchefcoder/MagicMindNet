"""Train inpainting diffusion, export checkpoint, reload and inpaint sample."""

from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parent.parent
MANIFEST = ROOT / "tests" / "fixtures" / "image_edit.json"
FIXTURES = ROOT / "tests" / "fixtures" / "samples"
OUT = ROOT / "examples" / "_roundtrip_diffusion_edit.mmn"


def main() -> None:
    ds = ai.DatasetImageEdit(file=str(MANIFEST))
    d = ai.Diffusion()
    cfg = ai.TrainConfig(epochs=4, learning_rate=0.05, cuda=False)
    ai.TrainDiffusion(d, ds, cfg)
    before = d.sample_inpaint_rgb_patch(
        str(FIXTURES / "photo.png"),
        str(FIXTURES / "mask.png"),
        steps=5,
        seed=2,
    )
    ai.export_diffusion(d, "safetensors", str(OUT))
    loaded = ai.import_diffusion("safetensors", [str(OUT)])
    after = loaded.sample_inpaint_rgb_patch(
        str(FIXTURES / "photo.png"),
        str(FIXTURES / "mask.png"),
        steps=5,
        seed=2,
    )
    assert before == after
    print(f"diffusion edit roundtrip ok: {OUT.name} patch[0]={after[0]:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
