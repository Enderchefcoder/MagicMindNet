# Dataset coverage matrix

Regression coverage for `mmn-data` loaders and Python dataset classes.

## DatasetQA

| Behavior | Rust | Python |
|----------|------|--------|
| JSON load | yes | `test_errors.py`, matrix |
| JSONL load (2 rows) | `jsonl_loads_two_rows` | `test_dataset_matrix_py.py` |
| Missing `user_row` | — | `test_errors.py` |
| Missing `ai_row` | `missing_ai_row_raises` | matrix |
| `format` getter (`json` / `jsonl`) | yes | matrix |
| `type_` == `qa` | — | matrix |
| `format_sample` system + thinktag | — | matrix |
| `format_sample` out of range | — | `test_dataset_qa_format_sample.py` |
| QA on diffusion rejected | `validate_for_model_rejects_diffusion` | — |

## DatasetCorpus

| Behavior | Rust | Python |
|----------|------|--------|
| Two-file load | — | `test_datasets.py`, corpus getters |
| Sort by complexity (short first) | `sort_by_complexity_orders_short_first` | — |
| `type_` / `format` / `corpus_batch_size` | — | `test_dataset_corpus_getters.py` |
| `Train` + `compute_mean_loss` (LM) | `train_corpus_reduces_mean_loss` | `test_train_corpus_py.py` |

## DatasetClassification

| Behavior | Rust | Python |
|----------|------|--------|
| Load with tag column | yes | `test_datasets.py` |
| Auto `class_N` when tag missing | `auto_tags_when_tag_column_missing` | matrix |
| `unique_labels` sorted deduped | `unique_labels_sorted_deduped` | `test_dataset_classification_unique_labels.py` |
| `type_` / `format` | — | matrix, repr tests |

## DatasetImageGen / DatasetImageEdit

See [image_coverage.md](image_coverage.md) for the full matrix and fixtures.

| Behavior | Rust | Python |
|----------|------|--------|
| Load manifest | `image_gen_loads_negative_prompt` | `test_image_fixtures_py.py` |
| Edit mask + negative | `image_edit_loads_mask_and_negative_prompt` | `test_image_fixtures_py.py` |
| `type_` / `format` getters | — | `test_dataset_image_format.py` |

## ChatXML

| Behavior | Rust |
|----------|------|
| Thinktag split | `thinktag_split` |
| `cot=false` omits think wrappers | `cot_false_omits_think_wrappers` |

## Running

```powershell
cargo test -p mmn-data
pytest tests/test_dataset_matrix_py.py tests/test_errors.py -q
.\scripts\verify_gate.ps1
```
