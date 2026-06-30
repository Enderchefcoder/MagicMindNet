import magicmindnet as ai


def test_train_config_repr_contains_fields():
    cfg = ai.TrainConfig(epochs=2, batch_size=4, cuda=False, optimizer="adamw", learning_rate=0.01)
    text = repr(cfg)
    assert "TrainConfig" in text
    assert "epochs=2" in text
    assert "batch_size=4" in text
    assert "adamw" in text
