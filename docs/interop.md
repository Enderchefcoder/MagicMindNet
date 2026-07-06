# Global format interop — GGUF, PyTorch, NumPy/TensorFlow

MagicMindNet reads and writes the major model/array formats of the wider ML
ecosystem. **Every codec is implemented from scratch in the Rust core** — no
llama.cpp, no libtorch, no zlib, and no Python-side numpy/torch dependency.

| Format | Extension | Read | Write | Implementation |
| --- | --- | --- | --- | --- |
| MMN JSON safetensors | `.mmn` | ✅ | ✅ | `mmn-safetensors-v1` wrapper |
| HF binary safetensors | `.safetensors` | ✅ | ✅ | `safetensors` container |
| GGUF (llama.cpp ecosystem) | `.gguf` | ✅ | ✅ (F32, Q8_0) | from-scratch container + dequant |
| PyTorch state dict | `.pt` / `.pth` | ✅ | ✅ | from-scratch ZIP + pickle VM |
| NumPy archive | `.npz` | ✅ | ✅ | from-scratch ZIP + NPY codec |
| NumPy array | `.npy` | ✅ | ✅ | from-scratch NPY codec |
| Architecture stub | `.bin` | ✅ | ✅ | `mmn-bin-v1` JSON |

`ai.load(path)` detects all of them automatically by magic bytes and archive
contents — no format argument needed.

## GGUF — run llama.cpp-ecosystem models, from scratch

The GGUF reader implements the container spec directly (versions 2 and 3):
header, typed metadata key/values, tensor infos, and the aligned data section.
Quantized tensors are dequantized by from-scratch block codecs:

- **Classic quants:** Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- **K-quants:** Q4_K, Q6_K (the Q4_K_M pairing used by most published models)
- **Floats/ints:** F32, F16, BF16, F64, I8/I16/I32/I64

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

bot.save("model_f32.gguf", format="gguf")       # write GGUF back out
bot.save("model_q8.gguf", format="gguf-q8_0")   # 8-bit block-quantized
```

Limitations: vision chatbots cannot be exported to GGUF (use safetensors or
npz); the writer emits F32 or Q8_0 tensors.

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

Exports include an `_mmn_meta` JSON entry so shape/seed/RoPE settings survive
the roundtrip; external checkpoints without it infer architecture from tensor
shapes exactly like the HF safetensors importer.

## NumPy `.npy` / `.npz` — also the TensorFlow bridge

The NPY codec handles format 1.0/2.0 headers, every common dtype
(`f2/f4/f8`, signed/unsigned ints, bools, big-endian variants), and
Fortran-ordered arrays. The ZIP layer reads DEFLATE-compressed entries via a
from-scratch RFC 1951 decompressor, so `np.savez_compressed` archives load
too.

```python
bot.save("bot.npz", format="npz")   # numpy.load()-compatible checkpoint
bot = ai.load("weights.npz")        # HF/MMN-named arrays adapt

ai.save_npy("x.npy", [[1.0, 2.0]])  # generic arrays — numpy optional
x = ai.load_npy("x.npy")
ai.save_npz("many.npz", {"w": [[1.0]], "b": [0.5]})
```

TensorFlow/Keras interchange goes through the same path — export weights with
`np.savez(path, **{name: w})` from `model.get_weights()` and load them here,
or read a MagicMindNet `.npz` from TF with `np.load`.

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
| safetensors binary header | HF safetensors (chatbot or classifier) |
| `{` JSON | MMN JSON formats (`mmn-safetensors-v1`, …) |

## Tensor ops (mmn-core)

The interop work also expanded `mmn_core::Tensor` with NumPy/PyTorch-style
operations: `sub`, `div`, `neg`, `exp`, `log`, `sqrt`, `abs`, `pow_scalar`,
`sigmoid`, `tanh`, `clamp`, `add_scalar`, `mul_scalar`, `reshape`,
`transpose`, `flatten`, `sum_axis`, `mean_axis`, `max_value`, `min_value`,
`argmax`, `argmin`, `argmax_rows`, `from_vec`, and `to_vec`. Unary math ops
register autograd nodes like the existing `relu`.

## Regression tests

- Rust: `crates/mmn-io/src/interop/*` module tests (container roundtrips,
  quant block codecs, pickle VM opcodes, inflate vectors, detection) and
  `crates/mmn-core/src/elementwise.rs`.
- Python: `tests/test_interop_npy_py.py`, `tests/test_interop_gguf_py.py`,
  `tests/test_interop_pt_py.py`, `tests/test_interop_npz_chatbot_py.py`,
  `tests/test_universal_formats_py.py` — including CPython-`pickle`
  cross-checks and (when numpy is installed) `numpy.load` compatibility.
