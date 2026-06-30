"""Smoke-test runnable example scripts."""


def test_benchmark_train_example_runs(run_example):
    proc = run_example("benchmark_train.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean loss before" in proc.stdout


def test_benchmark_train_learned_pe_example_runs(run_example):
    proc = run_example("benchmark_train.py", "--learned-pe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_learned_pos_embed: True" in proc.stdout
    assert "mean loss before" in proc.stdout


def test_benchmark_train_bpe_example_runs(run_example):
    proc = run_example("benchmark_train.py", "--bpe")
    assert proc.returncode == 0


def test_benchmark_train_rope_example_runs(run_example):
    proc = run_example("benchmark_train.py", "--rope")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_rope: True" in proc.stdout
    assert "mean loss before" in proc.stdout


def test_bpe_roundtrip_example_runs(run_example):
    proc = run_example("bpe_roundtrip.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "bpe roundtrip ok:" in proc.stdout


def test_bpe_roundtrip_train_example_runs(run_example):
    proc = run_example("bpe_roundtrip.py", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "trained with BPE:" in proc.stdout


def test_quickstart_example_runs(run_example):
    proc = run_example("quickstart.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_quickstart_learned_pe_example_runs(run_example):
    proc = run_example("quickstart.py", "--learned-pe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_learned_pos_embed: True" in proc.stdout


def test_quickstart_rope_example_runs(run_example):
    proc = run_example("quickstart.py", "--rope")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_rope: True" in proc.stdout


def test_quickstart_bpe_example_runs(run_example):
    proc = run_example("quickstart.py", "--bpe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "bpe merges:" in proc.stdout
    assert "Training finished." in proc.stdout
    assert "Training finished." in proc.stdout


def test_checkpoint_roundtrip_example_runs(run_example):
    proc = run_example("checkpoint_roundtrip.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_learned_pos_embed_roundtrip_example_runs(run_example):
    proc = run_example("learned_pos_embed_roundtrip.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "learned pos_embed roundtrip ok" in proc.stdout
    assert "max_seq_len=" in proc.stdout


def test_learned_pos_embed_roundtrip_train_example_runs(run_example):
    proc = run_example("learned_pos_embed_roundtrip.py", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "trained before export" in proc.stdout
    assert "learned pos_embed roundtrip ok" in proc.stdout


def test_rope_roundtrip_example_runs(run_example):
    proc = run_example("rope_roundtrip.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "rope roundtrip ok" in proc.stdout
    assert "rope_theta=" in proc.stdout


def test_rope_roundtrip_train_example_runs(run_example):
    proc = run_example("rope_roundtrip.py", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "trained before export" in proc.stdout
    assert "rope roundtrip ok" in proc.stdout


def test_classifier_roundtrip_example_runs(run_example):
    proc = run_example("classifier_roundtrip.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout


def test_generate_reply_example_runs(run_example):
    proc = run_example("generate_reply.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "reply:" in proc.stdout


def test_rl_spin_example_runs(run_example):
    proc = run_example("rl_spin.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "after SPIN" in proc.stdout


def test_classification_example_runs(run_example):
    proc = run_example("classification.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "labels:" in proc.stdout


def test_classification_benchmark_example_runs(run_example):
    proc = run_example("classification_benchmark.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "loss before:" in proc.stdout
    assert "loss after:" in proc.stdout
    assert "predict:" in proc.stdout


def test_corpus_benchmark_example_runs(run_example):
    proc = run_example("corpus_benchmark.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean corpus loss before" in proc.stdout


def test_corpus_benchmark_learned_pe_example_runs(run_example):
    proc = run_example("corpus_benchmark.py", "--learned-pe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_learned_pos_embed: True" in proc.stdout
    assert "mean corpus loss before" in proc.stdout


def test_corpus_benchmark_rope_example_runs(run_example):
    proc = run_example("corpus_benchmark.py", "--rope")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_rope: True" in proc.stdout
    assert "mean corpus loss before" in proc.stdout


def test_corpus_benchmark_bpe_example_runs(run_example):
    proc = run_example("corpus_benchmark.py", "--bpe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "bpe merges:" in proc.stdout
    assert "mean corpus loss before" in proc.stdout


def test_diffusion_smoke_example_runs(run_example):
    proc = run_example("diffusion_smoke.py", timeout=30)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "smoke_step finite: True" in proc.stdout


def test_eval_mean_loss_qa_runs(run_example):
    proc = run_example("eval_mean_loss.py", "qa")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean QA loss:" in proc.stdout


def test_eval_mean_loss_cls_runs(run_example):
    proc = run_example("eval_mean_loss.py", "cls")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean classification loss:" in proc.stdout


def test_eval_mean_loss_corpus_runs(run_example):
    proc = run_example("eval_mean_loss.py", "corpus")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean corpus loss:" in proc.stdout


def test_eval_mean_loss_qa_learned_pe_runs(run_example):
    proc = run_example("eval_mean_loss.py", "qa", "--learned-pe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_learned_pos_embed: True" in proc.stdout
    assert "mean QA loss:" in proc.stdout


def test_eval_mean_loss_corpus_learned_pe_runs(run_example):
    proc = run_example("eval_mean_loss.py", "corpus", "--learned-pe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_learned_pos_embed: True" in proc.stdout
    assert "mean corpus loss:" in proc.stdout


def test_eval_mean_loss_qa_rope_runs(run_example):
    proc = run_example("eval_mean_loss.py", "qa", "--rope")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_rope: True" in proc.stdout
    assert "mean QA loss:" in proc.stdout


def test_eval_mean_loss_corpus_rope_runs(run_example):
    proc = run_example("eval_mean_loss.py", "corpus", "--rope")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_rope: True" in proc.stdout
    assert "mean corpus loss:" in proc.stdout


def test_eval_mean_loss_qa_rope_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "qa", "--rope", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean QA loss before:" in proc.stdout
    assert "mean QA loss after:" in proc.stdout


def test_eval_mean_loss_corpus_rope_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "corpus", "--rope", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean corpus loss before:" in proc.stdout
    assert "mean corpus loss after:" in proc.stdout


def test_eval_mean_loss_qa_bpe_runs(run_example):
    proc = run_example("eval_mean_loss.py", "qa", "--bpe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "bpe merges:" in proc.stdout
    assert "mean QA loss:" in proc.stdout


def test_eval_mean_loss_corpus_bpe_runs(run_example):
    proc = run_example("eval_mean_loss.py", "corpus", "--bpe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "bpe merges:" in proc.stdout
    assert "mean corpus loss:" in proc.stdout


def test_eval_mean_loss_qa_bpe_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "qa", "--bpe", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean QA loss before:" in proc.stdout
    assert "mean QA loss after:" in proc.stdout


def test_eval_mean_loss_qa_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "qa", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean QA loss before:" in proc.stdout
    assert "mean QA loss after:" in proc.stdout


def test_eval_mean_loss_qa_learned_pe_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "qa", "--train", "--learned-pe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_learned_pos_embed: True" in proc.stdout
    assert "mean QA loss before:" in proc.stdout
    assert "mean QA loss after:" in proc.stdout


def test_eval_mean_loss_corpus_learned_pe_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "corpus", "--train", "--learned-pe")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "use_learned_pos_embed: True" in proc.stdout
    assert "mean corpus loss before:" in proc.stdout
    assert "mean corpus loss after:" in proc.stdout


def test_eval_mean_loss_corpus_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "corpus", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean corpus loss before:" in proc.stdout
    assert "mean corpus loss after:" in proc.stdout


def test_eval_mean_loss_cls_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "cls", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean classification loss before:" in proc.stdout
    assert "mean classification loss after:" in proc.stdout


def test_eval_mean_loss_diffusion_runs(run_example):
    proc = run_example("eval_mean_loss.py", "diffusion")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean diffusion denoise loss:" in proc.stdout


def test_eval_mean_loss_diffusion_edit_runs(run_example):
    proc = run_example("eval_mean_loss.py", "diffusion-edit")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean diffusion-edit denoise loss:" in proc.stdout


def test_eval_mean_loss_diffusion_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "diffusion", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean diffusion loss before:" in proc.stdout
    assert "mean diffusion loss after:" in proc.stdout


def test_eval_mean_loss_diffusion_edit_train_runs(run_example):
    proc = run_example("eval_mean_loss.py", "diffusion-edit", "--train")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean diffusion-edit loss before:" in proc.stdout
    assert "mean diffusion-edit loss after:" in proc.stdout


def test_diffusion_benchmark_example_runs(run_example):
    proc = run_example("diffusion_benchmark.py", timeout=120)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean denoise loss before:" in proc.stdout
    assert "mean denoise loss after:" in proc.stdout


def test_diffusion_benchmark_edit_example_runs(run_example):
    proc = run_example("diffusion_benchmark.py", "--edit", timeout=120)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "mean denoise loss before:" in proc.stdout
    assert "mean denoise loss after:" in proc.stdout


def test_diffusion_roundtrip_example_runs(run_example):
    proc = run_example("diffusion_roundtrip.py", timeout=120)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "diffusion roundtrip ok:" in proc.stdout


def test_diffusion_edit_roundtrip_example_runs(run_example):
    proc = run_example("diffusion_edit_roundtrip.py", timeout=120)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "diffusion edit roundtrip ok:" in proc.stdout


def test_diffusion_quantize_roundtrip_example_runs(run_example):
    proc = run_example("diffusion_quantize_roundtrip.py", timeout=60)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "diffusion quantize roundtrip ok:" in proc.stdout
    assert "parameters=" in proc.stdout


def test_diffusion_merge_demo_example_runs(run_example):
    proc = run_example("diffusion_merge_demo.py", timeout=60)
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "merge_diffusion ok:" in proc.stdout


def test_vision_chatbot_example_runs(run_example):
    proc = run_example("vision_chatbot.py")
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert "has_vision:" in proc.stdout
    assert "loaded has_vision:" in proc.stdout
