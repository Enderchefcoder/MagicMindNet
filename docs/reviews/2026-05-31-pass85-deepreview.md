# Deep review pass 85 — 2026-05-31

## Scope

Sinusoidal position encoding, causal getter, classifier encoder documentation.

## Changes

- **`sinusoidal_position_encoding` / `add_sinusoidal_position_encoding`** in `mmn-nn`; wired in `forward_hidden` + `backward_lm_grads`
- **`Chatbot::uses_causal_attention`** + Python `uses_causal_attention` getter
- **`encode_text_normalizes_bytes`** Rust test; **`test_chatbot_position_pe_py.py`**
- **`docs/position_encoding_coverage.md`**; classifier byte-encoder section in `classifier_coverage.md`

## Verification

```
.\scripts\verify_gate.ps1  → verify_gate: OK
Rust #[test]: 186 (+5)
pytest: 432 (+2)
ruff: clean
```

## Merge-ready

YES

## Next (pass 86)

- Learned `pos_embed` table (checkpoint + trainable) behind opt-in flag
- RoPE design note in `position_encoding_coverage.md`
- `test_mmn_py_bindings_py.py` — assert `uses_causal_attention` on Chatbot
