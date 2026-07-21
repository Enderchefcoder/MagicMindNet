# Benchmarks & eval harness

MagicMindNet ships a unified offline eval harness that covers the full feature
surface (LM, Glint, GQA, PE/RoPE/BPE/Unigram, classifier, diffusion, hub
synthetics, IO formats, RL/SPIN, merge/quantize) — not just scattered example
scripts.

## Quick start

```bash
# Smoke suite (CI-friendly)
python -m magicmindnet.eval smoke

# Everything
python -m magicmindnet.eval all --json -o /tmp/mmn_eval.json

# One family
python -m magicmindnet.eval lm
python -m magicmindnet.eval cls
python -m magicmindnet.eval io

# Or via the example wrapper
python examples/eval_harness.py --list-suites
python examples/eval_harness.py --list-tasks lm
```

Python API:

```python
import magicmindnet as ai

report = ai.run_suite("smoke")
print("\n".join(report.summary_lines()))

# Custom task list
harness = ai.EvalHarness(seed=1)
report = harness.run(tasks=["lm_qa_train", "cls_train", "io_roundtrip_gguf"])
report.write_json("report.json")
```

## Suites

| Suite | Scope |
|-------|--------|
| `smoke` | Fast CI subset (loss, Glint, vision flag, hub caps, safetensors IO, arrays) |
| `lm` | QA/corpus CE, PE/RoPE/BPE/Unigram/GQA/head_dim/Glint, generate latency |
| `cls` | Classifier mean loss, train delta, accuracy |
| `diffusion` | Denoise + edit + train deltas |
| `io` | safetensors / HF / GGUF / Q8 / npz / pt roundtrip timing + sizes |
| `hub` | Synthetic causal / cls / rerank / diffusion / seq2seq + capabilities |
| `glint` | Looped RMS+SwiGLU+tied train |
| `generate` | Native + hub generate latency |
| `rl` | RL policy + SPIN smoke |
| `train` | All train-delta tasks |
| `all` | Every registered task |

Full task matrix: [eval_coverage.md](eval_coverage.md).

## Legacy example scripts

Thin demos remain for focused runs (also covered by the harness):

| Script | Purpose |
|--------|---------|
| `examples/benchmark_train.py` | QA train delta (`--learned-pe` / `--rope` / `--bpe`) |
| `examples/corpus_benchmark.py` | Corpus LM train delta |
| `examples/classification_benchmark.py` | Classifier train delta |
| `examples/diffusion_benchmark.py` | Denoise train delta (`--edit`) |
| `examples/interop_benchmark.py` | Multi-format save/load table |
| `examples/eval_mean_loss.py` | Mean CE / denoise printer |
| `examples/eval_harness.py` | CLI wrapper for `magicmindnet.eval` |

## Rust / unit baselines

| Command | Purpose |
|---------|---------|
| `cargo test --workspace` | Full Rust suite |
| `pytest -q` | Full Python suite |
| `bash scripts/verify_gate.sh` | Merge gate (tests + smoke + lint counts) |

Record date, OS, and CPU/GPU in PRs when changing hot paths.
