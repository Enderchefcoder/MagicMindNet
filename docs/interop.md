# Global format interop — GGUF, PyTorch, NumPy, TensorFlow/Keras

MagicMindNet reads and writes the major model/array formats of the wider ML
ecosystem. **Every codec is implemented from scratch in the Rust core** — no
llama.cpp, no libtorch, no zlib, no HDF5 library, and no Python-side
numpy/torch/h5py dependency.

| Format | Extension | Read | Write | Implementation |
| --- | --- | --- | --- | --- |
| MMN JSON safetensors | `.mmn` | ✅ | ✅ | `mmn-safetensors-v1` wrapper |
| HF binary safetensors | `.safetensors` | ✅ | ✅ | `safetensors` container |
| GGUF (llama.cpp ecosystem) | `.gguf` | ✅ | ✅ (F32/F16/Q8_0/Q4_0) | from-scratch container + dequant |
| PyTorch state dict (zip, ≥1.6) | `.pt` / `.pth` | ✅ | ✅ | from-scratch ZIP + pickle VM |
| PyTorch legacy (pre-1.6) | `.pt` / `.pth` | ✅ | — | pickle-stream + raw storages |
| NumPy archive | `.npz` | ✅ | ✅ | from-scratch ZIP + NPY codec |
| NumPy array | `.npy` | ✅ | ✅ | from-scratch NPY codec |
| HDF5 / Keras weights | `.h5` / `.weights.h5` | ✅ | — | from-scratch HDF5 reader |
| Keras v3 archive | `.keras` | ✅ | — | ZIP + HDF5 reader |
| Architecture stub | `.bin` | ✅ | ✅ | `mmn-bin-v1` JSON |

`ai.load(path)` detects checkpoint formats automatically by magic bytes and
archive contents — no format argument needed.

## GGUF — run llama.cpp-ecosystem models, from scratch

The GGUF reader implements the container spec directly (versions 2 and 3):
header, typed metadata key/values, tensor infos, and the aligned data section.
Quantized tensors are dequantized by from-scratch block codecs:

- **Classic quants:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1
- **K-quants (full family):** Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K
  (the Q4_K_M / Q3_K_M / Q5_K_M pairings used by most published models)
- **Non-linear lookup quants:** IQ4_NL, IQ4_XS
- **Codebook-grid IQ family (complete):** IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS,
  IQ2_S, IQ3_XXS, IQ3_S — the QuIP#-style lattice codebooks ship as compact
  packed tables decoded once at runtime
- **Ternary BitNet quants:** TQ1_0, TQ2_0
- **MXFP4** (E8M0-scaled FP4 blocks, as used by gpt-oss) and **NVFP4**
  (unsigned-E4M3-scaled FP4, the newest ggml addition)
- **Floats/ints:** F32, F16, BF16, F64, I8/I16/I32/I64

Removed ggml type ids (Q4_2/Q4_3, repacked `Q4_0_x_x`) are rejected with
actionable messages. Large checkpoints dequantize **in parallel across all
cores**.

**Every codec is cross-validated against llama.cpp's reference `gguf`
Python package** (`tests/test_interop_quant_crossval_py.py`): identical
outputs on random block payloads for all 24 types, agreement on
reference-quantized float data, and our GGUF files parse in the reference
`GGUFReader`. Raw payloads are also exposed as `ai._native.dequantize_ggml`.

Tensor names follow the llama.cpp convention (`token_embd.weight`,
`blk.N.attn_q.weight`, `blk.N.ffn_gate.weight`, …) and are mapped onto the
MagicMindNet transformer. SwiGLU `gate×up` pairs are fused, GQA head counts
come from `{arch}.attention.head_count_kv`, and RoPE frequency comes from
`{arch}.rope.freq_base`. Loaded models run on the existing KV-cache generation
engine:

```python
import magicmindnet as ai

bot = ai.load("model.gguf")       # any supported quantization
print(bot.chat("hello!"))

bot.save("model_f32.gguf", format="gguf")  # also: gguf-f16, gguf-q8_0, gguf-q4_0
```

### GGUF inspection and embedded tokenizers

`ai.gguf_info(path)` reads **only the header** (multi-GB files are loaded
incrementally, never fully into memory) and returns the version, alignment,
full metadata dict — including `tokenizer.chat_template` when present — and
per-tensor name/shape/type summaries.

GGUF models embed their vocabulary. `ai.load_gguf_tokenizer(path)` converts a
SentencePiece-unigram vocab (`tokenizer.ggml.model == "llama"`) into a
`UnigramEncoder` whose token ids match the model rows (`▁` space markers and
`<0xNN>` byte-fallback tokens are decoded). Exports can embed a vocabulary the
same way, producing a single self-contained model file:

```python
tok = ai.UnigramEncoder.train(corpus_lines, vocab_size=8192)
bot.save("packed.gguf", format="gguf-q8_0", unigram_encoder=tok)

bot = ai.load("packed.gguf")
tok = ai.load_gguf_tokenizer("packed.gguf")
print(bot.chat("hello", unigram_encoder=tok))
```

Byte-level BPE (`gpt2`) vocabularies — GPT-2, Llama-3, Qwen families — load
via `ai.load_gguf_bpe_tokenizer(path)`, returning a `Gpt2BpeEncoder` (from-
scratch bytes↔unicode table, ranked merges, approximate GPT-2
pretokenization; ids follow the vocabulary order). `Gpt2BpeEncoder.from_vocab`
also accepts HF-style token lists + merge rules directly.

Limitations: vision chatbots cannot be exported to GGUF (use safetensors or
npz); the writer emits F32/F16/Q8_0/Q4_0 tensors.

## PyTorch `.pt` — no torch required

`torch.save` files are ZIP archives holding a pickled object graph
(`data.pkl`) plus one raw storage blob per tensor. MagicMindNet parses that
with a from-scratch pickle virtual machine (protocols 2–4, including
`FRAME`/`MEMOIZE`, persistent storage IDs, strided/transposed views, and
F16/BF16/F64/int storages):

```python
bot.save("bot.pt", format="pt")   # torch.load()-compatible state dict
bot = ai.load("external.pt")      # llama-style HF state dicts adapt too

ai.save_pt("tensors.pt", {"w": [[1.0, 2.0]]})  # generic named arrays
tensors = ai.load_pt("tensors.pt")
```

The **legacy pre-1.6 format** (raw pickle stream with the
`0x1950a86a20f9469cfc6c` magic, protocol/sys-info pickles, and appended raw
storages) is auto-detected and read by the same APIs.

**Sharded checkpoints** (`pytorch_model.bin.index.json` /
`model.safetensors.index.json` + shard files, the Hugging Face layout for
large models) load through `ai.load(index_path)`: the `weight_map` resolves
shards relative to the index, mixing safetensors and torch shard formats
freely.

Exports include an `_mmn_meta` JSON entry so shape/seed/RoPE settings survive
the roundtrip; external checkpoints without it infer architecture from tensor
shapes exactly like the HF safetensors importer.

## HDF5 — TensorFlow/Keras weights without h5py

A from-scratch HDF5 reader covers the layout `h5py`/Keras write by default
("earliest" libver): superblock v0/1, version-1 object headers with
continuation blocks, symbol-table groups (B-tree v1 + local heap + SNOD), and
compact or contiguous datasets of fixed-point / IEEE-float types (F16 through
F64, all int widths, both endiannesses). Chunked/compressed datasets are
rejected with a clear message.

```python
weights = ai.load_h5("model.weights.h5")   # {"dense/kernel": [[...]], ...}
weights = ai.load_keras("model.keras")     # Keras v3 zip archive
```

## NumPy `.npy` / `.npz` — also a TensorFlow bridge

The NPY codec handles format 1.0/2.0 headers, every common dtype
(`f2/f4/f8`, signed/unsigned ints, bools, big-endian variants), and
Fortran-ordered arrays. The ZIP layer reads DEFLATE-compressed entries via a
from-scratch RFC 1951 decompressor, so `np.savez_compressed` archives load
too — and writes them with a from-scratch **DEFLATE compressor**
(fixed-Huffman + hash-chain LZ77): `ai.save_npz(path, arrays, compress=True)`
mirrors `np.savez_compressed` and is verified against CPython's `zlib`.

```python
bot.save("bot.npz", format="npz")   # numpy.load()-compatible checkpoint
bot = ai.load("weights.npz")        # HF/MMN-named arrays adapt

ai.save_npy("x.npy", [[1.0, 2.0]])  # generic arrays — numpy optional
x = ai.load_npy("x.npy")
ai.save_npz("many.npz", {"w": [[1.0]], "b": [0.5]})
```

TensorFlow/Keras interchange also goes through `np.savez` on
`model.get_weights()`, or read a MagicMindNet `.npz` from TF with `np.load`.

Anything exposing `.tolist()` (numpy arrays, torch tensors) is accepted by
`save_npy` / `save_npz` / `save_pt` directly.

## Universal detection

`detect_checkpoint_kind` (Rust) and `ai.load()` / `Chatbot.load()` (Python)
recognize files by content, not extension:

| Magic | Detected as |
| --- | --- |
| `GGUF` | GGUF chatbot |
| `PK\x03\x04` + `data.pkl` entry | PyTorch state dict |
| `PK\x03\x04` + `.npy` entries | NumPy npz checkpoint |
| pickle PROTO + torch legacy magic | Legacy PyTorch checkpoint |
| JSON with `weight_map` | Sharded HF checkpoint index |
| safetensors binary header | HF safetensors (chatbot or classifier) |
| `{` JSON | MMN JSON formats (`mmn-safetensors-v1`, …) |

## Performance notes

- GGUF tensor dequantization **and** PyTorch storage decoding fan out across
  `available_parallelism` threads.
- `gguf_info` / `load_gguf_tokenizer` parse the header only, growing the read
  buffer geometrically instead of loading the tensor data.
- The CRC-32 table is computed once per process (`OnceLock`); IQ codebook
  grids decode once into `OnceLock` caches.
- Unigram Viterbi encoding indexes pieces in a hash map — O(n·window)
  lookups even for 32k+ piece GGUF vocabularies (was a linear vocab scan).
- The DEFLATE compressor uses hash-chain LZ77 (32-deep chains, 32 KiB
  window) with fixed-Huffman blocks; entries that don't shrink stay stored.

## Tensor ops (mmn-core)

The interop work also expanded `mmn_core::Tensor` with NumPy/PyTorch-style
operations: `sub`, `div`, `neg`, `exp`, `log`, `sqrt`, `abs`, `pow_scalar`,
`sigmoid`, `tanh`, `clamp`, `add_scalar`, `mul_scalar`, `reshape`,
`transpose`, `flatten`, `sum_axis`, `mean_axis`, `max_value`, `min_value`,
`argmax`, `argmin`, `argmax_rows`, `from_vec`, and `to_vec`. Unary math ops
register autograd nodes like the existing `relu`.

## Regression tests

- Rust: `crates/mmn-io/src/interop/*` module tests (container roundtrips,
  every quant block codec against hand-built blocks, pickle VM opcodes,
  legacy torch streams, inflate vectors, HDF5 fixture parsing, detection)
  and `crates/mmn-core/src/elementwise.rs`.
- Python: `tests/test_interop_npy_py.py`, `tests/test_interop_gguf_py.py`,
  `tests/test_interop_gguf_info_py.py`, `tests/test_interop_pt_py.py`,
  `tests/test_interop_legacy_pt_py.py`, `tests/test_interop_h5_py.py`,
  `tests/test_interop_npz_chatbot_py.py`, `tests/test_universal_formats_py.py`
  — including CPython-`pickle` cross-checks in both directions and (when
  numpy/h5py are installed) `numpy.load` / `h5py` compatibility.
- Fixture: `tests/fixtures/simple.h5` (written by h5py) validates the HDF5
  parser without any Python dependency in Rust tests.
