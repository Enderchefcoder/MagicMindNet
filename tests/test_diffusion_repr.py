import magicmindnet as ai


def test_diffusion_repr_includes_latent_channels_and_parameters():
    d = ai.Diffusion()
    text = repr(d)
    assert "latent_channels" in text
    assert str(d.latent_channels) in text
    assert "parameters" in text
    assert str(d.parameters) in text


def test_diffusion_parameters_positive():
    d = ai.Diffusion()
    assert d.parameters > 10_000
