import magicmindnet as ai


def test_train_config_setters():
    cfg = ai.TrainConfig()
    cfg.epochs = 5
    cfg.batch_size = 16
    cfg.cuda = False
    cfg.optimizer = "adamw"
    cfg.learning_rate = 0.02
    assert cfg.epochs == 5
    assert cfg.batch_size == 16
    assert cfg.cuda is False
    assert cfg.optimizer == "adamw"
    assert abs(cfg.learning_rate - 0.02) < 1e-9
