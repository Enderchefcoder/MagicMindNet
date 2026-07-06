# Global format interop — GGUF, PyTorch, NumPy, TensorFlow/Keras

MagicMindNet reads and writes the major model/array formats of the wider ML
ecosystem. **Every codec is implemented from scratch in the Rust core** — no
llama.cpp, no libtorch, no zlib, no HDF5/LevelDB/protobuf libraries, and
since wave 7 **not even the `safetensors` crate**: the container codec
(`st_codec.rs`) is from scratch too, so the entire format layer carries zero
external format dependencies. No Python-side numpy/torch/h5py/tensorflow/onnx
packages are needed either.

| Format | Extension | Read | Write | Implementation |
| --- | --- | --- | --- | --- |
| MMN JSON safetensors | `.mmn` | ✅ | ✅ | `mmn-safetensors-v1` wrapper |
| HF binary safetensors | `.safetensors` | ✅ | ✅ | from-scratch container codec |
| GGUF (llama.cpp ecosystem) | `.gguf` | ✅ | ✅ (F32/F16/Q8_0/Q4_0/**Q4_K/Q5_K/Q6_K**) | from-scratch container + codecs |
| PyTorch state dict (zip, ≥1.6) | `.pt` / `.pth` | ✅ | ✅ | from-scratch ZIP + pickle VM |
| PyTorch legacy (pre-1.6) | `.pt` / `.pth` | ✅ | — | pickle-stream + raw storages |
| Sharded HF checkpoint | `*.index.json` | ✅ | — | weight_map + shard readers |
| NumPy archive | `.npz` | ✅ | ✅ (stored or deflate) | from-scratch ZIP + NPY codec |
| NumPy array | `.npy` | ✅ | ✅ | from-scratch NPY codec |
| HDF5 / Keras weights | `.h5` / `.weights.h5` | ✅ | ✅ | from-scratch HDF5 reader + writer |
| Keras v3 archive | `.keras` | ✅ | — | ZIP + HDF5 reader |
| TensorFlow checkpoint v2 | `.index` + `.data-…` | ✅ | ✅ | from-scratch LevelDB table + protobuf |
| ONNX model weights | `.onnx` | ✅ | ✅ | from-scratch protobuf walker + writer |
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

### Quant encoding

The writer encodes every practical target — classic quants **byte-identical
to the reference implementation** (`gguf-q4_0` / `q4_1` / `q5_0` / `q5_1` /
`q8_0`, ggml's exact truncating rounding), plus TQ1_0/TQ2_0 ternary blocks
and MXFP4 (reference-exact codebook + E8M0 scale selection), plus **the full
k-quant family** (`gguf-q2_k` through `q6_k`, plus Q8_K) via from-scratch
ports of ggml's reference quantization searches (`make_qx_quants` iscale
refinement, `make_q3_quants` iterative RMSE refinement, `make_qkx2_quants`
joint scale+min least squares in both squared-error and MAD variants,
round-half-to-even `nearest_int`), plus the **IQ4 lookup quants**
(`gguf-iq4_nl` / `gguf-iq4_xs`, the ggml `ntry` scale search over the
non-linear codebook — the only IQ types encodable without calibration data).
The reference llama.cpp Python package cannot encode k-quants or IQ4 at all;
ours produces blocks it decodes identically, with a monotone quality ladder
(Q2_K → Q8_K reconstruction error strictly improves). Per the ggml spec,
quantization is **row-wise**: tensors whose fastest dimension is not a block
multiple stay F32 automatically. Tensor payload encoding runs **in parallel
across cores**.

**Vision chatbots export too**: vision prefix tensors travel under
mmproj-style `v.*` names (`v.patch_proj.weight`, `v.cross_attn_q.weight`, …)
with an `mmn.vision` metadata flag, and roundtrip through `ai.load`.

Limitations: the codebook-grid IQ1/IQ2/IQ3 families decode but do not encode
(their quantizers require importance-matrix calibration data by design).

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

A from-scratch HDF5 reader covers the layouts `h5py`/Keras write: superblock
v0/1, version-1 object headers with continuation blocks, symbol-table groups
(B-tree v1 + local heap + SNOD), and compact, contiguous, or **chunked**
datasets (B-tree v1 chunk index with edge-chunk clipping) of fixed-point /
IEEE-float types (F16 through F64, all int widths, both endiannesses). The
**gzip filter** (zlib wrapper, from-scratch inflate + Adler-32 verification)
and **shuffle filter** (byte transpose) are undone per chunk, so
`compression="gzip", shuffle=True` files read fine.

```python
weights = ai.load_h5("model.weights.h5")   # {"dense/kernel": [[...]], ...}
weights = ai.load_keras("model.keras")     # Keras v3 zip archive
ai.save_h5("out.h5", weights)              # h5py opens this natively
```

The **writer** emits superblock v0, symbol-table groups (fixed-allocation
B-tree v1 + SNOD nodes as libhdf5 expects), local heaps with free-list
descriptors, and contiguous F32 datasets with h5py's exact IEEE-float
datatype encoding. Names with `/` create nested groups. `h5py.File` reads the
output directly (verified in tests).

**Validated against real TensorFlow output**: committed fixtures under
`tests/fixtures/tf/` were written by TensorFlow 2.21 Keras (`model.save` and
`save_weights`), and live tests (skipped unless `tensorflow` is installed)
compare our reader against `model.get_weights()` exactly.

## TensorFlow checkpoint v2 — `.index` + `.data` without TensorFlow

`tf.train.Checkpoint` bundles are a LevelDB-style SSTable index
(prefix-compressed key blocks, varint block handles, masked-CRC32C trailers)
whose values are `BundleEntryProto` protobuf messages pointing into raw data
shards. MagicMindNet parses all of it from scratch — including a from-scratch
CRC-32C (Castagnoli) with TensorFlow's masking — and verifies per-tensor
checksums:

```python
arrays = ai.load_tf_checkpoint("ckpt")        # or "ckpt.index"
arrays = ai.load_tf_checkpoint("saved_model_dir")  # SavedModel variables/
# {"w": [[...]], "b": [...]}  (/.ATTRIBUTES/VARIABLE_VALUE stripped)

ai.save_tf_checkpoint("out", arrays)          # tf.train.load_checkpoint reads it
```

The **writer** builds the sorted LevelDB table (data/metaindex/index blocks
with masked-CRC32C trailers and the 48-byte footer) plus the raw data shard;
`tf.train.load_checkpoint` reads the output, and a full-circle test
(TF write → our read → our write → TF read) passes bit-for-bit.

All numeric `DataType`s decode to f32 (float/double/half/bfloat16, all int
widths, bool); the object-graph metadata entry is skipped. Fixtures written
by real TensorFlow are committed, and live tests compare against
`tf.train.load_checkpoint` bit-for-bit.

## ONNX — model weights without onnx/protobuf

A from-scratch protobuf wire-format walker extracts every graph initializer
from `.onnx` files: `dims` (packed or repeated), all numeric tensor types
(`raw_data` little-endian plus typed `float_data`/`int*_data` fields),
external-data models rejected with a re-export hint:

```python
weights = ai.load_onnx("model.onnx")   # {"w": [[...]], "b": [...]}
ai.save_onnx("out.onnx", weights)      # passes onnx.checker.check_model
```

Validated against models built and saved by the official `onnx` package;
our writer's output loads with `onnx.load` and passes the official checker.

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

- GGUF tensor dequantization, PyTorch storage decoding, ZIP entry inflation,
  **and HDF5 chunk decompression** fan out across `available_parallelism`
  threads.
- `gguf_info` / `load_gguf_tokenizer` parse the header only, growing the read
  buffer geometrically instead of loading the tensor data.
- The CRC-32 table is computed once per process (`OnceLock`); IQ codebook
  grids decode once into `OnceLock` caches.
- Unigram Viterbi encoding indexes pieces in a hash map — O(n·window)
  lookups even for 32k+ piece GGUF vocabularies (was a linear vocab scan).
- The DEFLATE compressor uses hash-chain LZ77 (32-deep chains, 32 KiB
  window) with fixed-Huffman blocks; entries that don't shrink stay stored.
- The inflate decoder uses a **10-bit one-hit Huffman lookup table** over a
  64-bit bit accumulator (bit-by-bit canonical walk only for rare long
  codes) — the classic zlib fast path, from scratch.

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
