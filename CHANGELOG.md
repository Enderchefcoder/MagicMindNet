# Changelog

## 0.1.0 — 2026-05-31

### Added
- Rust workspace (`mmn-core` … `mmn-py`) and Python package `magicmindnet`
- Datasets: QA, Corpus, Classification, ImageGen, ImageEdit; ChatXML formatting
- Models: `Chatbot` (autoset), `Classifier`, `Diffusion` foundation
- Training: `Train`, `RL`, `SPIN` with hybrid AdamW + Muon optimizer
- IO: export/import (`mmn-safetensors-v1`), merge, quantize (`int8`/`int4`), `limit()`
- CI: Windows + Linux (`cargo test`, `pytest`)

### Fixed
- Training applied fake gradients on cloned weights; real CE + linear backward now update weights
- `softmax(1)` on `[batch, classes]` normalized columns instead of rows (classifier probs summed to N)
- Export/import ignored model shape; checkpoints now include `meta` and restore architecture
- IO export now includes `lm_head` weights; merge averages embed + lm_head

### Fixed (pass 41)
- Safetensors import requires `vocab_size` in meta (no silent fallback from caller arg)
- Tests for invalid/empty checkpoint files, missing classifier head, bin `{}` defaults

### Fixed (pass 42)
- Safetensors import validates every block tensor shape vs `d_model` and `ffn_dim` (not just embed/lm_head)
- Classifier import rejects invalid JSON, empty files, and backbone shape mismatches (tests + docs)

### Added (pass 43)
- Import tests: missing `d_model` meta, ffn/ln block shapes, missing block tensor, lm_head shape, first-path-only, bin invalid/empty JSON
- Merge/quantize tests: n_layer mismatch, int4 head/block quantize parity (Rust + Python)

### Added (pass 44)
- Merge averaging tests for chatbot block weights and classifier head (Rust + pytest export JSON)
- ffn2 shape import test; classifier int8 head quantize; Rust int4 block ffn quantize
- Subagent `magicmindnet-checkpoint-strict` for IO regression gap scans

### Added (pass 117)
- `export(bot, "safetensors", path, bpe_encoder=)` writes `{stem}.bpe.mmn` + `meta.bpe_checkpoint`
- `load_bpe_sidecar(checkpoint_path)` helper; Rust `export_includes_bpe_checkpoint_meta` test

### Added (multi-patch vision cross-attention memory)
- `vision_rgb_patches_from_image_path(path, grid)` splits one image into `grid×grid` 8×8 tiles
- QA `image` column accepts comma-separated paths or JSON array; `vision_patch_grid` on `DatasetQA`
- Cross-attn uses all prefix rows as memory; Python `image_patches=` and `sample_image_paths`

### Added (Hugging Face binary safetensors interchange)
- `export(bot, "hf-safetensors", path)` / `import_model("hf-safetensors", [path])` via `safetensors` crate (F32, MMN key names)
- `import_model("safetensors", …)` auto-detects binary HF files vs JSON `mmn-safetensors-v1`
- Header metadata `format: mmn-hf-safetensors-v1` + JSON `meta`; Llama/GPT-style tensor name aliases on import
- Rust `hf_safetensors` module; pytest `test_hf_safetensors_py.py`; `examples/hf_safetensors_roundtrip.py`

### Added (external HF weight layout adapters on import)
- Split fused GPT-2 `c_attn` / `qkv_proj` into separate Q/K/V with Conv1d→Linear transpose
- Fuse Llama SwiGLU `gate_proj`×`up_proj` into MMN `ffn`; `down_proj`→`ffn2`; custom `ffn_dim` in meta
- Tie missing `lm_head` to `embed`; default RMSNorm-only γ=1 / β=0 for missing layernorm tensors

### Added (inference KV cache for generation)
- Per-layer K/V cache in `mmn-nn::kv_cache` with RoPE position offsets and GQA-aware attention
- `Chatbot.forward_logits_with_kv_cache` / `reset_kv_cache_prefill`; `GenerateConfig.use_kv_cache` (default `true`)
- Python `use_kv_cache=` on `generate` / `generate_tokens`; parity tests vs full forward
- `examples/gqa_rope_generate.py` GQA+RoPE train + KV benchmark smoke

### Added (sliding context window and min-p sampling)
- Generation continues past `max_seq_len` / 512-byte context via rolling window (KV re-prefill + full-forward slice)
- `GenerateConfig.min_p` tail-probability filter after nucleus sampling; Python `min_p=` kwarg
- Tests: `sliding_window_generates_past_max_ctx`, `test_sliding_window_past_learned_max_seq_len`

### Added (RoPE KV-cache slide and frequency/presence penalties)
- Incremental RoPE sliding window: drop oldest K/V row + re-index K positions instead of full re-prefill when context grows by one token
- `Chatbot.slide_kv_cache_one`; `LayerKvCache.truncate_front` + `slide_rope_kv_window_one` in `mmn-nn`
- `GenerateConfig.frequency_penalty` / `presence_penalty` (OpenAI-style logit penalties); Python kwargs on `generate` / `generate_tokens`
- Tests: `rope_kv_slide_matches_windowed_block_forward`, `rope_sliding_kv_generation_matches_full_forward`

### Added (unigram tokenizer export sidecar)
- `export(bot, "safetensors", path, unigram_encoder=)` writes `{stem}.unigram.mmn` + `meta.unigram_checkpoint`
- `load_unigram_sidecar(checkpoint_path)` Python helper; Rust `export_includes_unigram_checkpoint_meta` test
- `TokenizerSidecarRefs` for BPE + unigram meta on JSON and HF safetensors export

### Added (vision KV-cache generation and unigram vocab pruning)
- Vision prefix patches in KV-cache prefill with cached cross-attention memory for incremental decode
- `GenerateConfig.vision_patches`; Python `image_patch=` / `image_patches=` on `generate` / `generate_tokens`
- Full-forward vision generation re-applies patches each step (correctness fix)
- `UnigramEncoder.prune_pieces_below_logprob(min_log_prob)` drops low-score merged pieces
- Tests: `vision_kv_generation_matches_full_forward`, `test_vision_kv_cache_generate_py.py`

### Added (unigram tokenizer and nucleus sampling)
- `UnigramEncoder`: Viterbi segmentation, EM training, `mmn-unigram-v1` JSON save/load; Python `train` / `train_from_qa` / `train_from_corpus`
- `Train` / `RL` / `SPIN` / `compute_mean_loss` accept `unigram_encoder=` (mutually exclusive with `bpe_encoder`)
- Generation: `top_p` nucleus sampling and `repetition_penalty` on `GenerateConfig` / `Chatbot.generate`
- `examples/unigram_train_generate.py` + smoke; `tests/test_unigram_tokenizer_py.py`

### Added (generation stop sequences and long prompts)
- `stop_token_ids` and `stop_strings` on `GenerateConfig` / `Chatbot.generate`
- `generate_token_ids` + Python `generate_tokens`; `tokenize_for_generate` without training 32-token cap
- Stabilize `train_batch_size_two` test with fixed init seed

### Added (Chatbot autoregressive generation)
- `Chatbot.generate(prompt, ...)` with greedy (`temperature=0`) and temperature/top-k sampling
- `BytePairEncoder.decode` for BPE token roundtrip; `mmn-train::generate_text`
- `examples/generate_reply.py` + smoke; bin format stores `n_heads`/`n_kv_heads`/`ffn_dim`

### Added (Python GQA API)
- `Chatbot(..., n_heads=, n_kv_heads=)` constructor kwargs; getters `n_heads`, `n_kv_heads`
- `tests/test_gqa_chatbot_py.py`: HF/JSON roundtrip, fewer params than MHA, training reduces loss

### Added (native grouped-query attention)
- `ModelShape.n_kv_heads` (defaults to `n_heads`); `MultiHeadAttention` uses `[n_kv_heads * head_dim, d_model]` for `k_proj`/`v_proj`
- GQA-aware `scaled_dot_product_attention` forward/backward maps query heads to shared KV heads; RoPE rotates K over `n_kv_heads`
- HF import keeps native KV tensor shapes (`ensure_gqa_meta`); export writes `n_kv_heads` in meta when `!= n_heads`
- Rust tests: GQA forward vs expanded MHA parity, backward finite-diff, KV grad shapes through `TransformerBlock`

### Added (GQA expansion and BF16/F16 dtype import)
- ~~Expand grouped-query `k_proj`/`v_proj` to full `[d_model, d_model]`~~ superseded by native GQA above
- HF safetensors import decodes **F16** and **BF16** tensors to F32 via `half` crate

### Added (Classifier Hugging Face binary safetensors)
- `export_classifier(clf, "hf-safetensors", path)` / `import_classifier("hf-safetensors", [path])` — `mmn-hf-classifier-v1`
- `import_classifier("safetensors", …)` auto-detects binary vs JSON; cross-format guards vs Chatbot HF
- Shared `hf_tensor_codec` module for F32/F16/BF16 decode; `examples/classifier_hf_safetensors_roundtrip.py`

### Added (DatasetQA disk image paths for vision training)
- Optional `image` JSON column (`image_row` config); resolves relative paths against the QA manifest directory
- `vision_rgb_patch_from_image_path` resizes PNG/JPEG to 8×8×3 NCHW; grayscale fallback for legacy patch-only models
- Python `sample_image_path`, `ai.vision_rgb_patch_from_image_path`; train/mean-loss use file patches when present

### Fixed (vision cross-attention multi-layer training)
- `backward_lm_grads` iterated blocks in wrong order (`enumerate().rev()`); gradient apply now matches push order for `n_layer >= 2`

### Added (vision text-to-image cross-attention)
- `CrossAttention` after block 0: text rows query image prefix K/V with residual
- `vision_cross_attn.{q,k,v,out}` checkpoint tensors; merge/quantize; train backward
- Python `has_vision_cross_attn`; tests and updated `vision_coverage.md`

### Added (RGB conv vision patch encoder)
- `vision_patch_conv` (`3×8×8 → 1×8×8` Conv2d) before linear `vision_patch_proj` on `Chatbot(vision=True)`
- `vision_rgb_patch_from_text`, `VISION_RGB_DIM` (192); training defaults to RGB when conv is loaded
- Checkpoint `vision_patch_conv` tensor + meta `vision_rgb_patch`; merge/quantize support; conv backward for training

### Added (RoPE checkpoint roundtrip and trained export parity)
- `examples/rope_roundtrip.py` with optional `--train` before export/import mean-loss check
- Rust `import_preserves_forward_loss_rope`, `bin_rope_roundtrip_preserves_meta`, `train_rope_export_import_preserves_mean_loss`
- pytest `test_rope_roundtrip_*`, `test_rope_export_import_preserves_mean_loss`; smoke gate wiring

### Added (real Conv2d forward for diffusion blocks)
- `Conv2d::forward` NCHW convolution with same padding (`kernel/2`) instead of identity clone
- Tests `conv2d_same_padding_preserves_spatial_dims`, `vae_encoder_preserves_8x8_latent_shape`

### Added (RoPE example flags and corpus train test)
- `--rope` on `eval_mean_loss.py`, `corpus_benchmark.py`, and `quickstart.py` (mutually exclusive with `--learned-pe`)
- Rust `train_corpus_rope_reduces_mean_loss`; pytest merge mismatch + example smokes

### Added (RoPE position encoding)
- Opt-in rotary position embedding on Q/K after projection (`use_rope=True`, `rope_theta=10000`)
- `apply_rope` / `apply_rope_backward` in `mmn-nn`; mutually exclusive with learned `pos_embed`
- Checkpoint meta `use_rope` + `rope_theta`; merge requires matching RoPE settings
- Python getters `use_rope`, `rope_theta`; `benchmark_train.py --rope`; `tests/test_rope_chatbot_py.py`

### Added (vision patch encoder)
- `Chatbot(vision=True)` linear patch prefix projector (`VISION_PATCH_DIM=64`): prepends projected 8×8 patch row before text embeddings in forward/train
- `forward_hidden_with_patches`, `loss_on_batch_with_patches`, QA `Train` auto-uses `vision_patch_from_text(input)`
- Safetensors `vision_patch_proj` tensor + meta `vision_patch_dim`; merge/quantize support
- Python: `has_vision_patch_encoder`, `vision_patch_dim`, `compute_loss(..., image_patch=)`, `ai.vision_patch_from_text`
- `tests/test_vision_patch_encoder_py.py`; updated `docs/vision_coverage.md`

### Added (pass 116)
- `RL(..., bpe_encoder=)` and `SPIN(..., bpe_encoder=)` via `rl_with_bpe` / `spin_with_bpe`
- `examples/rl_spin.py --bpe`; pytest `test_rl_and_spin_with_bpe_encoder_smoke`

### Added (pass 115)
- `eval_mean_loss.py` `--bpe` and `--bpe-file PATH` for QA/corpus modes
- `quickstart.py --bpe` trains BPE, saves `tokenizer.mmn`, trains with `bpe_encoder`
- `tests/test_eval_mean_loss_bpe_py.py`; training_coverage BPE example matrix

### Added (pass 114)
- `examples/bpe_roundtrip.py` — BPE save/load parity + optional `--train` with `bpe_encoder`

### Added (pass 113)
- `mmn-bpe-v1` JSON checkpoints: `BytePairEncoder.export_json` / `import_json`, Python `save()` / `load()`
- Rust roundtrip + format validation tests; `corpus_benchmark.py --bpe` example + smoke

### Added (pass 112)
- `Chatbot.compute_mean_loss` / `compute_loss` optional `bpe_encoder=` (matches `Train` tokenization)
- `benchmark_train.py --bpe` example + smoke/pytest; `mean_qa_loss_with_bpe` / `mean_corpus_loss_with_bpe`

### Added (pass 111)
- Python `BytePairEncoder` (`train`, `train_from_qa`, `train_from_corpus`, `encode`)
- `Train(..., bpe_encoder=)` for QA and corpus LM via `train_with_bpe` / `train_corpus_with_bpe`
- Rust `train_with_bpe_reduces_loss`; `tests/test_bpe_tokenizer_py.py`; API.md + training_coverage

### Added (pass 108)
- int4 quantize-after-train learned PE; `test_merge_trained_learned_pos_embed_averages_weights`

### Added (pass 107)
- `test_quantize_int8_learned_pos_embed_after_train_within_tolerance`; `quantize_coverage.md` row

### Added (pass 106)
- Rust `train_corpus_learned_pos_embed_export_import_preserves_mean_loss`; coverage docs

### Added (pass 105)
- `mmn-io` dev-dep on `mmn-train`; `train_learned_pos_embed_export_import_preserves_mean_loss`
- `training_coverage.md` / `checkpoint_coverage.md` train→export rows

### Added (pass 104)
- `learned_pos_embed_roundtrip.py --train`; pytest + smoke; coverage docs

### Added (pass 102)
- `test_export_import_preserves_learned_pos_embed_after_train`; `checkpoint_coverage.md` row

### Added (pass 101)
- `benchmark_train.py --learned-pe`; pytest + smoke; docs updates

### Added (pass 100)
- `eval_mean_loss --train --learned-pe` pytest (QA + corpus); `training_coverage.md` learned-PE example table

### Added (pass 99)
- `eval_mean_loss.py --learned-pe` for QA/corpus; pytest smoke tests; API/examples docs

### Added (pass 98)
- `corpus_benchmark.py --learned-pe` flag; smoke + `test_corpus_benchmark_learned_pe_example_runs`

### Added (pass 97)
- README learned PE example row; `position_encoding_coverage.md` pytest smoke test name

### Added (pass 96)
- `test_learned_pos_embed_roundtrip_example_runs` in `test_examples_scripts_py.py`
- `docs/API.md` learned PE example link + examples table row; `examples_coverage.md` pytest column

### Added (pass 95)
- `import_preserves_forward_loss_learned_pos_embed` Rust test; `examples/learned_pos_embed_roundtrip.py` in smoke gate
- README + quickstart comment for `use_learned_pos_embed`; `examples_coverage.md` row

### Added (pass 94)
- `pos_embed` parametric IO matrix tests (missing/shape/merge/quantize) in `test_io_checkpoint_matrix_py.py`
- `test_export_import_preserves_learned_pos_embed_compute_loss`

### Added (pass 93)
- Quantize learned `pos_embed`: forward/mean loss finite with &lt;50% relative drift (Rust + pytest)
- `checkpoint_coverage.md` learned PE section; `quantize_coverage.md` pos_embed rows

### Added (pass 92)
- Post-import `Train()` with learned `pos_embed` on QA + corpus datasets (pytest)
- `training_coverage.md` learned PE rows; link to `position_encoding_coverage.md`

### Added (pass 91)
- Corpus `Train()` updates learned `pos_embed` (Rust + pytest); vision+bin PE roundtrip test
- `training.md` corrected train scope (attn/LN/PE); `vision_coverage.md` bin PE row

### Added (pass 90)
- `test_learned_pos_embed_compute_mean_loss_decreases_after_train`; `limitations.md` learned PE + bin meta notes

### Added (pass 89)
- `mmn-bin-v1` stores `use_learned_pos_embed` / `max_seq_len`; import uses `new_with_pe_options`
- `docs/API.md` position-encoding section; `checkpoints.md` updated for `pos_embed`
- `max_seq_len` overflow guard tests (Rust + pytest)

### Added (pass 88)
- RL keeps learned `pos_embed` frozen; SPIN updates it via Train phase (pytest)
- `import_rejects_pos_embed_shape_mismatch`; `parameters()` includes `max_seq_len * d_model`

### Added (pass 87)
- Merge/quantize regression tests for learned `pos_embed` (Rust + pytest)
- `test_train_changes_learned_pos_embed`; merge rejects learned vs sinusoidal PE mismatch

### Added (pass 86)
- Opt-in learned `pos_embed` table on `Chatbot` (`use_learned_pos_embed`, `max_seq_len`); checkpoint IO + merge + quantize
- `train_step_lm` updates learned PE; Python getters; `test_learned_pos_embed_io_py.py`; RoPE design sketch

### Added (pass 85)
- Sinusoidal position encoding on `Chatbot` embed (runtime; no new checkpoint keys)
- `Chatbot.uses_causal_attention` Python/Rust getter; `encode_text` byte normalization test
- `docs/position_encoding_coverage.md`; classifier encoder docs in `classifier_coverage.md`

### Fixed (pass 84)
- Causal self-attention mask (default for `MultiHeadAttention`) — LM blocks no longer attend to future tokens
- Renamed `test_train_frozen_attn_ln_py.py` → `test_train_block_params_py.py`; added embed/lm_head update test

### Fixed (pass 83)
- `TransformerBlock` forward missing FFN residual (`out = x2 + ffn`); backward now routes `grad_out` through both skip connections
- LN γ/β finite-diff test; block-level backward finite-diff regression

### Added (pass 82)
- `layernorm_backward` + finite-diff test; `Train()` updates ln1/ln2 γ/β via `backward_attn_ffn` (10 grads/block)
- RL still frozen on LN; SPIN updates LN via Train phase

### Added (pass 81)
- `scaled_dot_product_attention_backward` + finite-diff test; `Train()` updates attn q/k/v/out via `backward_attn_ffn`
- RL still frozen on attn; SPIN updates attn via Train phase

### Added (pass 80)
- `mmn-nn::scaled_dot_product_attention` — real multi-head self-attention forward (QKᵀ/√d, softmax, @V)
- 3 new `mmn-nn` tests; attn still frozen in `train_step_lm` (backward next)

### Added (pass 79)
- `docs/mmn_py_coverage.md` — module map for split `mmn-py` bindings
- `tests/test_mmn_py_bindings_py.py` — PyO3 smoke + IO roundtrip (7 tests)

### Added (pass 78)
- `mmn-py/train/mod.rs`, `io/mod.rs` — `Train`/`RL`/`SPIN` and checkpoint wrappers; `lib.rs` thin registry (58 lines)

### Added (pass 77)
- `mmn-py/models/classifier.rs` — `PyClassifier` split

### Added (pass 76)
- `mmn-py/models/chatbot.rs` — `PyChatbot` split

### Added (pass 75)
- `mmn-py/datasets/classification.rs`, `datasets/image.rs` — completes datasets/ split (step 4)

### Added (pass 74)
- `mmn-py/datasets/corpus.rs` — `PyDatasetCorpus` split (step 4b)

### Added (pass 73)
- `mmn-py/datasets/qa.rs`, `models/diffusion.rs` — split plan steps 4a + 5a

### Added (pass 72)
- `mmn-py/src/train_config.rs`, `resource.rs` — split plan steps 2–3

### Added (pass 71)
- `mmn-py/src/errors.rs` — PyO3 exceptions + `mmn_err_to_py` (split plan step 1)

### Added (pass 70)
- `load_checkpoint` conftest helper; IO matrix tests DRY; `docs/mmn_py_split_plan.md`; `checkpoint_coverage` helper table

### Added (pass 69)
- `tensor_entry_first_f32` / `tamper_tensor_entry_first_f32` in conftest; IO matrix tests DRY; `test_conftest_helpers_py.py`; `nn_coverage` roadmap link

### Added (pass 68)
- `conftest` checkpoint helpers; frozen train/RL tests DRY; attention backward design sketch; `limitations.md` in CONTRIBUTING/testing index

### Added (pass 67)
- `test_spin_does_not_change_ln1_gamma`; `verify_gate.sh` venv fallback; `limitations.md` roadmap table + RL/SPIN frozen note

### Added (pass 66)
- RL/SPIN frozen LN+attn pytest; `attention_coverage` / `layernorm_coverage` RL rows; `CONTRIBUTING` venv_python docs

### Added (pass 65)
- `scripts/venv_python.ps1` / `.sh` — CI/smoke/count scripts use `.venv` Python when present; `classification.py` in smoke; RL attn frozen pytest

### Added (pass 64)
- `docs/testing.md` coverage index; `AGENTS.md` coverage links; `test_train_frozen_attn_ln_py.py` (attn/LN frozen + FFN positive control)

### Added (pass 63)
- `docs/layernorm_coverage.md`, `docs/nn_coverage.md`; `train_step_does_not_update_layernorm_params`; +4 `mmn-nn` attention/block tests; CONTRIBUTING coverage matrix links

### Added (pass 62)
- `mmn-io/io_tests/` split (78 chatbot + 20 classifier regression tests); `docs/examples_coverage.md`; `docs/attention_coverage.md`; attn frozen test; example smokes for quickstart/roundtrips

### Added (pass 61)
- `mmn-io/chatbot_io.rs` (safetensors/bin/merge/quantize); `eval_mean_loss.py --train` for qa/corpus/cls; GHA smoke comment

### Added (pass 60)
- `eval_mean_loss.py corpus` mode; conftest `run_example` script args; example smokes for eval_mean_loss + vision_chatbot
- `mmn-io/tensor_merge.rs` + `classifier_io.rs` split from `lib.rs`

### Added (pass 59)
- Vision chatbot path tests/docs (`vision_coverage.md`); LN quantize non-default γ/β tests (`quantize_coverage.md`); `examples/vision_chatbot.py`; `punish_only` RL pytest; training.md RL mode table

### Added (pass 58)
- `mmn-io/block_tensors.rs`; RL `reward_only`/`selfplay`/`punish_only` modes; image fixtures + `image_coverage.md`

### Added (pass 57)
- Corpus LM training (`train_corpus`, `mean_corpus_loss`); `Diffusion.smoke_step()`; `mmn-io/checkpoint_util.rs`; corpus + diffusion examples/smoke

### Added (pass 56)
- `tests/conftest.py` shared example harness; `test_classifier_edge_cases_py.py`; `classification_benchmark` in smoke; `docs/classifier_coverage.md`; Rust hybrid train + unknown-tag mean loss tests

### Fixed (pass 55)
- Muon `newton_schulz5` returned all-zero orthogonalized updates (wrong iteration formula); hybrid optimizer now updates 2D matrices
- Muon Nesterov momentum blend aligned with Keller Jordan reference

### Added (pass 55)
- `docs/optimizers_coverage.md`; Rust optim/autograd tests (+9); `test_optimizer_integration_py.py`

### Added (pass 54)
- Expanded `docs/API.md` (TOC, `__all__` table, cross-links); `examples/README.md` + `rl_spin.py`; `test_api_surface_py.py`, `test_examples_scripts_py.py`; smoke adds benchmark_train + rl_spin

### Added (pass 53)
- Dataset coverage matrix (`docs/dataset_coverage.md`); QA jsonl/missing-ai-row; classification auto-tags; corpus sort; ChatXML cot test; `test_dataset_matrix_py.py`

### Added (pass 52)
- Training coverage matrix (`docs/training_coverage.md`); RL lm_head weight tests; multi-block train + post-import train tests

### Added (pass 51)
- Classifier IO parametric matrix (`test_io_classifier_matrix_py.py`); multi-block `n_layer=2` tests (`test_io_multiblock_chatbot_py.py`); Rust block1 missing/merge tests; quickstart uses project `.venv`

### Added (pass 50)
- 100% chatbot IO contract matrix (`test_io_checkpoint_matrix_py.py`, `docs/checkpoint_coverage.md`); massive README; remaining missing-block import tests

### Added (pass 49)
- Missing attn.k/ln1.gamma import; merge ln beta/gamma parity; int8 attn.q/ffn2 and int4 attn.k/q quantize tests

### Added (pass 48)
- Missing block attn.q/ffn2 import tests; merge ffn/ffn2/ln1.gamma averaging; int8 attn.k and int4 attn.out/ffn2 quantize parity

### Added (pass 47)
- ln1/ln2 beta shape import tests; merge attn k/v/out averaging (Rust + pytest); int8 attn.out and int4 attn.v quantize parity

### Added (pass 46)
- Import attn.v/attn.out shape mismatch tests (Rust + pytest); Python merge lm_head and classifier backbone averaging; int8 block attn.v quantize test

### Added (pass 45)
- Merge embed/lm_head averaging; import attn.k, ln2.gamma, missing lm_head tests; int8 block ffn quantize

### Added (pass 42)
- Python/Rust tests: block shape mismatch, `n_layer` meta vs tensor count, classifier corrupt files, corpus fixed batch, int4 quantize weight change

### Added (pass 41)
- Python/Rust tests: vocab_size meta, invalid JSON, empty file, missing head, corpus `row` batch, bin defaults

### Fixed (pass 40)
- Import validates tensor shapes vs meta (chatbot embed/lm_head; classifier backbone/head); classifier requires `input_dim` and non-empty labels
- Safe tensor byte parsing (no panic on malformed JSON bytes)
- `DatasetCorpus.corpus_batch_size` getter documents invalid batch_size fallback to 24

### Added (pass 40)
- Rust/Python strict import tests, classifier int8 quantize weight change, corpus batch_size test

### Fixed (pass 39)
- **Import safety:** safetensors/classifier import now fails on missing required tensors, incomplete meta (`n_layer`/`d_model`), or tensor data length mismatch (no silent partial load)
- Docs: classifier factories, classifier IO, int4 quantize, merge vision OR, import validation notes

### Added (pass 39)
- Rust tests: int4 quantize, d_model merge mismatch, vision OR merge, corrupt/missing checkpoint paths
- Python tests: classifier import errors, d_model merge, vision merge, int4 weight change, classifier seed import

### Fixed (pass 38)
- Documented `merge()` vocab_size guard (fixed pass 37); regression tests for vocab mismatch and quantize shape stability

### Added (pass 38)
- Tests: autoset sub-10B, merge vocab mismatch, quantize preserves getters, classifier nested import, DataMismatchError message, merge_classifier callable

### Fixed (pass 37)
- `merge()` now rejects mismatched `vocab_size` with `ModelMismatchError` instead of panicking on embed average

### Added (pass 37)
- Tests: autoset sub-1B budget, autoset+seed, classifier loss/mean-loss finite, bin nested export, IO alias equivalence, ModelMismatchError message
- Rust: `export_bin_creates_parent_directory`; GHA runs `count_tests.sh` on all OS

### Added (pass 36)
- Tests: Train/RL/merge callables, classifier `init_seed`, safetensors `has_vision` roundtrip, mean-loss finite, classifier nested export
- Rust: `export_classifier_creates_parent_directory` test

### Added (pass 35)
- Tests: public exceptions, corpus getters, SPIN callable, merge preserves `parameters`, finite `compute_loss`, export creates nested path
- `scripts/lint.sh`; GHA uses `smoke_examples.sh` for all example steps

### Fixed (pass 35)
- `mmn-io` export paths create parent directories before write (`write_file_create_parents`)

### Added (pass 34)
- Tests: `layer_size`/`has_vision`, Diffusion `repr`, `__all__` export surface, `TrainClassifier` callable
- Linux scripts: `ci_local.sh`, `count_tests.sh`, `smoke_examples.sh`; GHA prints test counts on Ubuntu

### Added (pass 33)
- Tests: public IO aliases, classifier ctor labels, Chatbot `tokenizer`, Diffusion `latent_channels`
- `scripts/verify_gate.sh` for bash merge gate; docs for Linux verify

### Added (pass 32)
- Tests: classifier predict probs sum to 1, TrainConfig defaults, package getters, merge meta, limit without `%`
- `__version__` in `magicmindnet.__all__`; API docs for `limit_percent`

### Added (pass 31)
- Tests: quantize unknown mode, dataset `rows`/`type_`, merge/safetensors shape getters
- `scripts/verify_gate.ps1` — `ci_local` + `count_tests` merge gate

### Added (pass 30)
- Tests: bin shape/vision getters, classifier roundtrip `input_dim`/`num_labels`, autoset getters, import missing file
- GHA: `examples/quickstart.py`; `magicmindnet-gate` subagent; docs/checkpoints getter examples

### Added (pass 29)
- Chatbot getters: `vocab_size`, `n_layer`, `d_model`; Classifier `num_labels` getter
- Tests: shape getters, `num_labels`, merge `input_dim` mismatch, `import_model("bin", [])`

### Added (pass 28)
- `Classifier.input_dim` getter (PyO3); docs/API.md updated
- Tests: same-seed Chatbot `compute_loss`, empty import file lists, `input_dim` getter
- CI: `eval_mean_loss.py cls` in GHA + `smoke_examples.ps1`

### Added (pass 27)
- `mmn-bin-v1` format guard on `import_model("bin")`; export_bin writes format + vision
- Tests: `test_bin_io.py`, `test_classifier_unknown_label.py`, `test_chatbot_autoset.py`
- `eval_mean_loss.py qa` in examples smoke

### Added (pass 26)
- Tests: export/import unknown format, QA `format_sample`, int4 quantize (chatbot + classifier)
- Documented `int4` quantize in `docs/checkpoints.md` (dependency-groups deferred: pip `--group` breaks maturin on older pip)
- GitHub Actions: checkpoint + classifier roundtrip examples after pytest

### Added (pass 25)
- `import_safetensors` rejects wrong `format` (classifier / unknown); Python + Rust tests
- `test_quantize_classifier.py`, `test_dataset_classification_unique_labels.py`
- `scripts/smoke_examples.ps1` wired into `ci_local.ps1`

### Fixed (pass 25)
- Loading a classifier checkpoint via `import_model` no longer silently produces a broken Chatbot

### Added (pass 24)
- `Diffusion.__repr__` + `latent_channels` getter; image datasets `format` getter
- `RL` + `cuda=True` regression test; `__version__` vs `pyproject.toml` test
- `examples/classifier_roundtrip.py`, `tests/fixtures/labels_small.json`, `magicmindnet-examples` agent

### Added (pass 23)
- `DatasetImageGen` / `DatasetImageEdit` `repr()`; `test_train_classifier_cuda.py`
- `examples/checkpoint_roundtrip.py`; `scripts/count_tests.ps1`

### Added (pass 22)
- `merge()` preserves first model `init_seed`; Chatbot repr shows `init_seed` when set
- `DatasetCorpus.__repr__`; tests `test_merge_chatbot_seed`, `test_dataset_corpus_repr`

### Added (pass 21)
- `DatasetClassification.__repr__`; Classifier repr shows `init_seed` when set
- Checkpoint export omits `meta.seed` when unset; `merge_classifier` keeps first `init_seed`
- `.pre-commit-config.yaml` (ruff); `magicmindnet-docs` subagent; tests for repr/merge seed

### Added (pass 20)
- Checkpoint `meta.seed` + `Chatbot.init_seed` / `Classifier.init_seed` getters
- `DatasetQA.__repr__`; tests `test_checkpoint_meta_seed.py`, `test_dataset_qa_repr.py`

### Added (pass 19)
- `TrainClassifier` honors `TrainConfig.batch_size` via `GradAccumulator` (parity with `Train`)
- Tests: `test_train_classifier_batch_size.py`, Rust `train_classifier_batch_size_two_*`
- GitHub Actions: `ruff check`; chatbot export loss test uses `seed=`

### Added (pass 18)
- `Chatbot.__repr__`, `Classifier.__repr__`; ruff in `[dev]` + `scripts/lint.ps1`
- `magicmindnet-ci` subagent; README/CONTRIBUTING CI + `eval_mean_loss` links
- Tests: `test_chatbot_repr.py`, `test_classifier_repr.py`

### Added (pass 17)
- LM `Train()` honors `TrainConfig.batch_size` via gradient accumulation (`GradAccumulator`)
- `tests/test_train_batch_size.py`; `mmn-optim` grad accumulator unit test

### Added (pass 16)
- `Chatbot.compute_mean_loss` rejects non-QA datasets with `DataMismatchError`
- `TrainConfig.__repr__`; shuffled epochs in `train_classifier`

### Added (pass 15)
- `Classifier.compute_mean_loss(DatasetClassification)`; `mean_classification_loss` in Rust
- `TrainConfig` Python setters (writable fields)

### Added (pass 14)
- `Classifier` optional `seed=`; `TrainConfig` Python getters
- `docs/testing.md`, `examples/classification_benchmark.py`
- `tests/test_classifier_seed.py`, `test_train_config_getters`

### Added (pass 13)
- `Chatbot(..., seed=)` for reproducible weight init; `merge_classifier` for classifiers
- RL CE targets use aligned output tokens (not input tokens)
- Tests: `test_seed`, `test_merge_classifier`, `test_autoset`

### Added (pass 12)
- `align_qa_token_pairs` — input/output byte tokens truncated to matching length before CE
- `compute_mean_loss(dataset_qa)` on Chatbot; export roundtrip loss tests
- `scripts/ci_local.ps1`, `magicmindnet-python` subagent

### Added (pass 11)
- `Chatbot.compute_loss`, `Classifier.compute_loss`, `DatasetClassification.unique_labels` (Python)
- `RL` / `SPIN` raise `DataMismatchError` for non-QA datasets
- `tests/test_chatbot_loss.py`, `test_dataset_labels.py`; benchmark prints before/after loss

### Added (pass 10)
- `embedding_backward` in `mmn-core`; `train_step_lm` now updates embedding weights
- `Train()` validates dataset type at Python boundary (`DataMismatchError` for classification data)
- `tests/test_train_rejects_classification_dataset` (pytest), `chatbot_tests` module split

### Added (pass 9)
- `train_step_lm` backprops FFN through **all** transformer blocks (not only the last)
- `docs/checkpoints.md`, README classification + IO links, `import_classifier_rejects_chatbot_checkpoint` test

### Added (pass 8)
- Classifier IO: `export_classifier` / `import_classifier` / `quantize_classifier` (`mmn-classifier-v1`)
- `tests/test_classifier_io.py`, `test_train_classifier_rejects_qa_dataset`, `examples/classification.py`

### Added (pass 7)
- `Classifier::train_step`, `TrainClassifier`, CE backward through backbone + head
- `validate_dataset_for_classifier`, `tests/test_train_classifier.py`

### Added (pass 6)
- Real FFN backward on last block (`gelu_backward`, `forward_with_ffn_cache`, `linear_backward` chain)
- `docs/training.md`, `limit_percent()` Python API, `magicmindnet-classify` subagent

### Added (pass 5)
- Checkpoint LayerNorm γ/β per block (`ln1`/`ln2`); merge/quantize include them
- `Classifier.with_labels`, `Classifier.from_classification`, `DatasetClassification.unique_labels()`
- `tests/test_classifier_labels.py`, `tests/test_limit.py`; `mmn-resource` limit parse unit tests

### Added (pass 4)
- Full transformer linear checkpoint export/import (`blocks.*` attn + ffn)
- `import_preserves_forward_loss` IO test; quantize applies to all exported linears
- `CONTRIBUTING.md`, `tests/test_package.py`, `magicmindnet-train` subagent

### Added (pass 3)
- Real `LayerNorm` forward (per-row) with unit test
- `docs/limitations.md`, `tests/test_quickstart.py`, `magicmindnet-io` subagent

### Known limitations
- Embedding gather not fully differentiable through blocks
- `mmn-cuda` CPU parity path unless built with `--features cuda`
- Diffusion UNet conv forward is structural stub
- Safetensors format is JSON wrapper, not Hugging Face binary safetensors
