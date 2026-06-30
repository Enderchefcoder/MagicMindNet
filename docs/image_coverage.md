# Image dataset coverage

Regression coverage for `DatasetImageGen` and `DatasetImageEdit` loaders.

## DatasetImageGen

| Behavior | Rust | Python |
|----------|------|--------|
| Load prompt/image/negative_prompt | `image_gen_loads_negative_prompt` | `test_dataset_matrix_py.py` |
| Fixture manifest | — | `test_image_fixtures_py.py` |
| `format` / `type_` getters | — | `test_dataset_image_format.py` |
| `repr` | — | `test_dataset_image_gen_repr.py` |
| `resolve_image_path` / `image_path_at` / `prompt_at` | `image_gen_resolve_image_path_relative_to_manifest` | `test_dataset_image_paths_py.py` |

## DatasetImageEdit

| Behavior | Rust | Python |
|----------|------|--------|
| Load mask + negative_prompt | `image_edit_loads_mask_and_negative_prompt` | `test_dataset_matrix_py.py` |
| Fixture manifest | — | `test_image_fixtures_py.py` |
| `format` / `type_` getters | — | `test_dataset_image_format.py` |
| `resolve_*` / `image_path_at` / `mask_path_at` | `image_edit_resolve_paths_relative_to_manifest` | `test_dataset_image_paths_py.py` |

## Diffusion validation

| Behavior | Test |
|----------|------|
| Rejects corpus type | `dataset_validation_tests::diffusion_rejects_corpus_dataset_type` |
| Accepts image_gen type | `dataset_validation_tests::diffusion_accepts_image_gen_dataset_type` |

Fixtures: `tests/fixtures/image_gen.json`, `image_edit.json`.

See [diffusion_coverage.md](diffusion_coverage.md) and [dataset_coverage.md](dataset_coverage.md).
