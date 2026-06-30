import magicmindnet as ai


def test_train_config_default_constructor_values():
    cfg = ai.TrainConfig()
    assert cfg.epochs == 1
    assert cfg.batch_size == 8
    assert cfg.cuda is False
    assert cfg.optimizer == "hybrid"
    assert abs(cfg.learning_rate - 3e-4) < 1e-9
