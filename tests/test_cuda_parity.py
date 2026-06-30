import pytest

import magicmindnet as ai


def test_cuda_requested_without_gpu_raises():
    cfg = ai.TrainConfig(epochs=1, batch_size=1, cuda=True)
    path = __file__.replace("test_cuda_parity.py", "fixtures/qa_valid.json")
    ds = ai.DatasetQA(file=path, user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32)
    with pytest.raises(ai.CUDAError):
        ai.Train(bot, ds, cfg)
