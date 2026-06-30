"""Quantize diffusion weights, export, reload, and verify sampling parity."""

from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "examples" / "_roundtrip_diffusion_quant.mmn"


def main() -> None:
    d = ai.Diffusion()
    print(f"parameters: {d.parameters}")
    ai.quantize_diffusion(d, "int8")
    patch_before = d.sample_rgb_patch(steps=5, seed=11)
    ai.export_diffusion(d, "safetensors", str(OUT))
    loaded = ai.import_diffusion("safetensors", [str(OUT)])
    assert loaded.parameters == d.parameters
    patch_after = loaded.sample_rgb_patch(steps=5, seed=11)
    assert patch_before == patch_after
    print(f"diffusion quantize roundtrip ok: {OUT.name} parameters={loaded.parameters}")
    print("Done.")


if __name__ == "__main__":
    main()
