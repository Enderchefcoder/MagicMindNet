"""Diffusion smoke via Python API."""

import magicmindnet as ai


def test_diffusion_smoke_step_finite():
    d = ai.Diffusion()
    assert d.smoke_step() is True
