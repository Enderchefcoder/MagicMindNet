"""Diffusion latent sampling + VAE decode demo."""

import magicmindnet as ai


def main() -> None:
    d = ai.Diffusion()
    patch = d.sample_rgb_patch(steps=8, seed=42)
    assert len(patch) == ai.VISION_RGB_DIM
    assert all(v == v for v in patch)
    print(f"sample_rgb_patch len={len(patch)} first={patch[0]:.4f} last={patch[-1]:.4f}")
    print("Done.")


if __name__ == "__main__":
    main()
