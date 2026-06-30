# Pass 9 artifact

- `train_step_lm` FFN backward through all blocks (reverse pass with per-block cache)
- `docs/checkpoints.md`, README classification section
- `import_classifier_rejects_chatbot_checkpoint` (Rust + pytest)
- Verified: 31 Rust unit tests, 29 pytest
