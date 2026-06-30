import magicmindnet as ai


def test_spin_runs_on_cpu_without_cuda_error():
    path = __file__.replace("test_spin_cuda.py", "fixtures/qa_valid.json")
    ds = ai.DatasetQA(file=path, user_row="input", ai_row="output")
    bot = ai.Chatbot(vocab_size=256, n_layer=2, d_model=32)
    ai.SPIN(bot, selfplay_epochs=1, dataset=ds)
