import magicmindnet as ai


def test_diffusion_latent_channels_getter_positive():
    d = ai.Diffusion()
    assert d.latent_channels >= 1
