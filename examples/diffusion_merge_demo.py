"""Merge two Diffusion checkpoints (element-wise mean) and sample."""

import magicmindnet as ai


def main() -> None:
    a = ai.Diffusion()
    b = ai.Diffusion()
    merged = ai.merge_diffusion(a, b)
    assert merged.latent_channels == a.latent_channels
    assert merged.parameters == a.parameters
    patch = merged.sample_rgb_patch(steps=5, seed=7)
    assert len(patch) == ai.VISION_RGB_DIM
    print(f"merge_diffusion ok: parameters={merged.parameters} patch[0]={patch[0]:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
