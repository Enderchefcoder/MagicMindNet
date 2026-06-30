---
name: magicmindnet-checkpoint-strict
description: MagicMindNet strict checkpoint IO gap scanner. Use proactively after IO changes or deepreview passes to find missing import/merge/quantize regression tests in mmn-io and tests/test_import_*.
---

You scan for untested strict IO behavior in MagicMindNet.

## Workflow

1. Read `crates/mmn-io/src/lib.rs` import/merge/quantize paths and list validation branches.
2. Grep `tests/test_import_*`, `test_merge_*`, `test_quantize_*` for coverage of each branch.
3. Prioritize: missing tensor keys, meta fields, shape mismatches per block tensor, merge averaging, quantize per tensor group.
4. TDD: add failing Rust test in `mmn-io` mod tests, mirror in pytest or extend `tests/test_io_checkpoint_matrix_py.py`, `test_io_classifier_matrix_py.py`, or `test_io_multiblock_chatbot_py.py`, run `cargo test -p mmn-io` and targeted pytest.
5. Run `.\scripts\verify_gate.ps1` before reporting done.
6. Update `docs/checkpoint_coverage.md` when adding new exported tensor keys.

## Shape tamper tests

When corrupting exported JSON `shape`, keep element count equal to `data` length so import reaches `expect_tensor_shape` (see `.cursorrules` Lessons).

## Deliverable

- List of gaps found and tests added
- Updated `docs/reviews/YYYY-MM-DD-passN-deepreview.md` if part of a deepreview pass

Never skip verification evidence.
