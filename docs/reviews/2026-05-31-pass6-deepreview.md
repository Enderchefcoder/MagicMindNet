# Pass 6 artifact

- Last-block FFN: CE → lm_head → ffn2 → GELU → ffn (not uniform fake grad on all blocks)
- `gelu_backward` + `TransformerBlock::forward_with_ffn_cache`
- `docs/training.md`, `limit_percent()`, `.cursor/agents/magicmindnet-classify.md`
- Verified: `cargo test --workspace` (27 unit), `pytest -q` (25 passed)
