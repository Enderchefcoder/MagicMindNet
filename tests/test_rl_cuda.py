import pytest

import magicmindnet as ai


def test_rl_cuda_without_gpu_raises():
    path = __file__.replace("test_rl_cuda.py", "fixtures/qa_valid.json")
    ds = ai.DatasetQA(file=path, user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32)
    cfg = ai.TrainConfig(epochs=1, batch_size=1, cuda=True)
    with pytest.raises(ai.CUDAError):
        ai.RL(bot, ds, cfg, reward_amount=1.0, punishment_amount=0.5, rl_type="policy")
