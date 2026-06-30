import magicmindnet as ai


def test_diffusion_repr_includes_latent_channels():
    d = ai.Diffusion()
    text = repr(d)
    assert "latent_channels" in text
    assert str(d.latent_channels) in text
