# Training coverage matrix

Regression coverage for `mmn-train` and Python `Train` / `TrainClassifier` / `RL` / `SPIN`.

## Language modeling (`Train`)

| Behavior | Rust (`mmn-train`) | Python (`tests/`) |
|----------|-------------------|-------------------|
| Reduces per-sample loss | `train_reduces_loss` | `test_mean_qa_loss.py` |
| Reduces mean QA loss | `train_batch_size_two_accumulates_and_reduces_loss` | `test_mean_qa_loss.py` |
| `batch_size=2` gradient accumulation | yes | `test_train_batch_size.py` |
| `hybrid` optimizer (default) | default in `train` | `test_mean_qa_loss.py` |
| `adamw` optimizer | yes | `test_train_batch_size.py` |
| `n_layer=2` block 1 FFN updates | — | `test_train_multiblock_py.py` |
| Train after import roundtrip | — | `test_train_multiblock_py.py` |
| Learned `pos_embed` trains (QA) | `train_step_updates_learned_pos_embed` | `test_train_block_params_py.py`, `test_chatbot_loss.py` |
| Learned `pos_embed` trains (corpus) | `train_corpus_updates_learned_pos_embed` | `test_train_corpus_py.py` |
| Post-import train + learned PE | — | `test_train_multiblock_py.py`, `test_train_corpus_py.py` |
| Train → export → import learned PE | `train_learned_pos_embed_export_import_preserves_mean_loss`, `train_corpus_learned_pos_embed_export_import_preserves_mean_loss` | `test_export_import_preserves_learned_pos_embed_after_train` |
| `max_seq_len` overflow error | `learned_pos_embed_rejects_long_sequence` | `test_learned_pos_embed_limits_py.py` |
| Rejects `DatasetClassification` | `validate_dataset_for_chatbot` | `test_data_mismatch.py` |
| Corpus LM (`train_corpus`) | `train_corpus_reduces_mean_loss` | `test_train_corpus_py.py` |
| BPE tokenization in `Train()` | `train_with_bpe_reduces_loss`, `tokenize_lm` | `test_bpe_tokenizer_py.py` |
| `compute_mean_loss` on corpus | `mean_corpus_loss` | `test_train_corpus_py.py` |
| Attn weights update in `train_step_lm` | `train_step_updates_attn_q_weights` | `test_train_block_params_py.py` (`test_train_changes_attn_q_weights`) |
| LN γ/β update in `train_step_lm` | `train_step_updates_layernorm_gamma` | `test_train_block_params_py.py` (`test_train_changes_ln1_gamma`) |
| FFN2 updates (positive control) | `train_step_updates_embed_and_ffn2` | `test_train_block_params_py.py` |
| Embed + lm_head update | — | `test_train_block_params_py.py` (`test_train_changes_embed_and_lm_head`) |

## Classification (`TrainClassifier`)

| Behavior | Rust | Python |
|----------|------|--------|
| Reduces loss on label | `train_classifier_reduces_loss` | `test_train_classifier.py` |
| Reduces mean classification loss | `train_classifier_batch_size_two_accumulates_and_reduces_loss` | `test_classifier_mean_loss.py` |
| `batch_size=2` accumulation | yes | `test_train_classifier_batch_size.py` |
| Rejects `DatasetQA` | yes | `test_data_mismatch.py` |

## RL / SPIN

| Behavior | Rust | Python |
|----------|------|--------|
| RL updates `lm_head` | `rl_changes_lm_head_weight` | `test_train_rl_spin_py.py` |
| `reward_only` RL mode | `rl_reward_only_updates_lm_head` | `test_train_rl_spin_py.py` |
| `punish_only` RL mode | `rl_punish_only_updates_lm_head` | `test_train_rl_spin_py.py` |
| Attn unchanged under RL | — | `test_train_rl_spin_py.py` (`test_rl_does_not_change_attn_q_weights`) |
| LN γ unchanged under RL | — | `test_train_rl_spin_py.py` (`test_rl_does_not_change_ln1_gamma`) |
| Learned `pos_embed` frozen under RL | — | `test_train_rl_spin_py.py` (`test_rl_does_not_change_learned_pos_embed`) |
| Learned `pos_embed` updates under SPIN | — | `test_train_rl_spin_py.py` (`test_spin_changes_learned_pos_embed`) |
| Attn updates under SPIN (Train phase) | — | `test_train_rl_spin_py.py` (`test_spin_changes_attn_q_weights`) |
| LN γ updates under SPIN (Train phase) | — | `test_train_rl_spin_py.py` (`test_spin_changes_ln1_gamma`) |
| `selfplay` uses reward-only scale | yes (in `rl`) | `examples/rl_spin.py` |
| SPIN completes, finite mean loss | `spin_completes_on_fixture` | `test_train_rl_spin_py.py` |
| Rejects classification dataset | — | `test_data_mismatch.py` |
| Smoke callable | — | `test_rl_callable.py`, `test_spin_callable.py` |

| Vision chatbot trains, keeps flag | `vision_chatbot_trains_and_keeps_vision_flag` | `test_vision_chatbot_py.py` |

## Example scripts (learned `pos_embed` flags)

| Script | Flags | pytest |
|--------|-------|--------|
| `corpus_benchmark.py` | `--learned-pe` | `test_corpus_benchmark_learned_pe_example_runs` |
| `eval_mean_loss.py` | `qa\|corpus --learned-pe` | `test_eval_mean_loss_qa_learned_pe_runs`, `test_eval_mean_loss_corpus_learned_pe_runs` |
| `eval_mean_loss.py` | `qa\|corpus --train --learned-pe` | `test_eval_mean_loss_qa_learned_pe_train_runs`, `test_eval_mean_loss_corpus_learned_pe_train_runs` |
| `benchmark_train.py` | `--learned-pe` | `test_benchmark_train_learned_pe_example_runs` |
| `quickstart.py` | `--learned-pe` | `test_quickstart_learned_pe_example_runs` |
| `learned_pos_embed_roundtrip.py` | `--train` | `test_learned_pos_embed_roundtrip_train_example_runs` |

## Example scripts (monitoring)

| Script | Mode | pytest |
|--------|------|--------|
| `eval_mean_loss.py` | `corpus` | `test_eval_mean_loss_corpus_runs` |
| `eval_mean_loss.py` | `--train` (qa/corpus/cls) | `test_eval_mean_loss_*_train_runs` |

## Monitoring

| API | Dataset | Test |
|-----|---------|------|
| `Chatbot.compute_mean_loss` | `DatasetQA` | `test_mean_qa_loss.py` |
| `Classifier.compute_mean_loss` | `DatasetClassification` | `test_classifier_mean_loss.py` |
| Wrong dataset type | both | `test_chatbot_mean_loss_mismatch.py`, `test_classifier_mean_loss.py` |

## Running

```powershell
cargo test -p mmn-train
pytest tests/test_train*.py tests/test_mean_qa_loss.py tests/test_classifier_mean_loss.py -q
.\scripts\verify_gate.ps1
```

See also [training.md](training.md), [position_encoding_coverage.md](position_encoding_coverage.md), and [limitations.md](limitations.md).
