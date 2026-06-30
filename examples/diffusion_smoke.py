"""Diffusion foundation smoke — one finite training step."""

import magicmindnet as ai


def main() -> None:
    d = ai.Diffusion()
    ok = d.smoke_step()
    print(f"Diffusion smoke_step finite: {ok}")
    assert ok, "diffusion training step produced non-finite values"
    print("Done.")


if __name__ == "__main__":
    main()
