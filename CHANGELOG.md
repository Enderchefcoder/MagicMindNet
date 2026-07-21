# Changelog

## 0.1.0 — 2026-07-21

### Added (feature parity Wave 1)
- llama.cpp sampling: `typical_p`, Mirostat v2 (`mirostat` / `mirostat_tau` / `mirostat_eta`)
- Ollama-style `Chatbot.generate_stream` (token chunks) and `Chatbot.embed` (mean-pool hidden)
- ChatML: `ai.format_chat_messages` + `Chatbot.chat_messages`
- PyTorch-style `TrainConfig`: `weight_decay`, `lr_schedule` (`constant`|`cosine`), `warmup_steps`
- Docs: [docs/feature_parity.md](docs/feature_parity.md); eval tasks `gen_typical_p_smoke`,
  `stream_generate`, `embed_mean_pool`
- Tests: `tests/test_feature_parity_py.py` + Rust sampling helpers

## 0.1.0 — 2026-07-06

### Added (final_norm + LoopLoRA QKV adapters)
- `Chatbot(final_norm=True)` — optional final LayerNorm/RMSNorm after all loops
  (GGUF `output_norm.*` / meta `mmn.final_norm`)
- `Chatbot(lora_rank=N)` — per-loop LoRA QKV adapters (zero-init up = identity at
  init; trained via `Train()`); checkpoint keys `loop_lora.{i}.down/up`
- Defaults `final_norm=False`, `lora_rank=0` preserve classic Chatbot bit-identity
- Tests: `tests/test_final_norm_loop_lora_py.py`; Rust nn/models coverage

### Added (unified eval harness / benchmarking suite)
- `magicmindnet.eval`: `EvalHarness` / `BenchmarkRunner`, `Metric`, `TaskResult`,
  `SuiteReport`, `list_tasks` / `list_suites` / `get_task` / `run_suite`
- Suites: `smoke`, `lm`, `cls`, `diffusion`, `io`, `hub`, `glint`, `generate`,
  `rl`, `train`, `all` — 35+ tasks covering QA/corpus CE, PE/RoPE/BPE/Unigram,
  GQA/`head_dim`, Glint, vision flag, classifier accuracy, diffusion denoise/edit,
  hub synthetics (causal/cls/rerank/diffusion/seq2seq), IO timing (safetensors/HF/
  GGUF/Q8/npz/pt), arrays roundtrip, RL/SPIN, merge/quantize
- CLI: `python -m magicmindnet.eval` + `examples/eval_harness.py`
- Docs: rewritten [benchmarks.md](docs/benchmarks.md), [eval_coverage.md](docs/eval_coverage.md)
- Tests: `tests/test_eval_harness_py.py`; smoke wires `eval_harness.py smoke`

### Added (hub wave-2 — callable families + offline coverage)
- HubModel: `capabilities()`, `score_pairs` / `rerank`, `embed`, diffusion/TTS/ASR/embedding/zero-shot/fill-mask/QA/vlm routing
- Diffusers `generate` → pipeline images/frames; native Diffusion `generate` → `sample_rgb_patch`
- Ollama `/api/chat` (+ system / messages); clearer GGUF-fail errors
- `DatasetQA.as_pairs()`; CoT empty `thinktag` defaults to `<think>…</think>` when `cot=True`
- `list_hub_families()`, `ModelCard.from_dict`, ModelScope URL parse
- `pip install -e ".[hub]"` optional deps; detect Diffusers dirs + nested checkpoints
- GGUF/HF Glint meta roundtrip (`n_loops`, RMS/SwiGLU, tie, `loop_embed`, `ffn_gate`)
- Examples: `hub_local_roundtrip.py`, `glint_tiny.py`; docs: `hub_coverage.md`
- Tests: `tests/test_hub_wave2_py.py` (+ smoke hub local)

### Fixed (classic MHA HF/npz/pt head_dim roundtrip)
- `ensure_gqa_meta` no longer guesses `n_heads=1` for square Q/K when head counts are
  absent (that broke HF/npz/pt loss roundtrips after optional `head_dim`)
- HF safetensors export always writes `n_heads` / `n_kv_heads` (and HF aliases)

### Added (optional head_dim for Qwen-style GQA)
- `MultiHeadAttention` / `TransformerBlock` / `Chatbot` accept optional `head_dim`
  independent of `d_model // n_heads` (Qwen3: d_model=1024, n_heads=16, head_dim=128
  → q_proj `[2048, 1024]`)
- GGUF import reads `{arch}.attention.key_length` / `value_length` (or infers from
  `attn_q` shape); HF/safetensors import infers the same; shape validation uses
  `q_dim = n_heads * head_dim`
- Python: `Chatbot(head_dim=…)` + `bot.head_dim` getter; hub `from_pretrained`
  marks native GGUF loads with `card.backend="native"`
- Tests: Rust MHA/block forward + synthetic Qwen GGUF; `tests/test_head_dim_py.py`

### Added (universal hub `ai.from_pretrained`)
- `ai.from_pretrained(source)` loads Hugging Face / ModelScope / Ollama / local models into a native Chatbot/Classifier/Diffusion when possible, otherwise a `HubModel` with `generate` / `predict` / `finetune` / `train` / `save`
- Routing by pipeline tag + architecture + file layout (GGUF, causal LM, seq-cls/reranker, seq2seq, diffusers/video, TTS) — not per-repo special cases
- `DatasetClassification.as_pairs()` / `DatasetCorpus.as_texts()` for hub finetune loops
- Optional `head_dim` for Qwen-style GQA (`n_heads * head_dim != d_model`) — native `Qwen/Qwen3-0.6B-GGUF` import
- Docs: [docs/hub.md](docs/hub.md); hands-on: `scripts/hub_hands_on.py`; tests: `tests/test_hub_from_pretrained_py.py`

### Added (Glint-style Chatbot architecture knobs)
- `Chatbot(n_loops=…, norm="rms"|"layer", ffn="swiglu"|"gelu", tie_embeddings=…, loop_embed=…, ffn_dim=…)` — recreate a tiny Glint-like LM in a few lines; defaults preserve classic Chatbot behavior
- Native **RMSNorm**, **SiLU/SwiGLU** FFN (`blocks.N.ffn_gate` + `ffn_kind` meta), **weight tying**, **loop embeddings**, and **shared-weight block looping** with accumulated train grads
- Safetensors/bin meta roundtrip for the new fields; KV-cache generation uses full forward when `n_loops>1`
- Tests: Rust unit (`silu`/`rms`/`swiglu`/`n_loops`/`tie`) + `tests/test_glint_arch_py.py`

### Added (load hardening + tiny train DX)
- Hardened `ai.load` / `detect_checkpoint_kind`: structural safetensors sniff (no more false positives on noise), TorchScript `constants.pkl` ZIP, HDF5/TFLite/pickle redirect hints, extension-aware errors; `Chatbot.load` supports GGML/GGJT legacy
- Training uses model `max_seq_len` (lifted hard 32-token cap); CoT `thinktag="think"` wraps Train targets; autoset presets `sub-1M` / `sub-10M` / `sub-50M`
- Tests: `tests/test_detect_load_hardening_py.py`, `tests/test_seq_len_autoset_cot_py.py`

### Added (interop wave 21: from-scratch Zstandard — the last compression gap)
- **zstd decoder (RFC 8878)**: frame headers, raw/RLE/compressed blocks,
  Huffman-coded literals (direct + FSE-compressed weight tables, 1- and
  4-stream), FSE-coded sequences (predefined / RLE / custom / repeat tables),
  the interleaved backward bitstream, repeat-offset history, skippable frames —
  ~700 lines, no zstd library
- Unlocks **zarr-python 3.x out-of-the-box stores** (zstd is the default codec
  for both new v2 and v3 stores), **blosc-zstd** chunks, and raw `numcodecs.Zstd`
  frames at every compression level (1 through 22 tested)
- Validated against numcodecs-written fixtures (repetitive/text/random/
  multi-block/level-19) + live zarr default-store matrix
- Debugging note: the initial failure traced to a misremembered predefined
  match-length distribution; fixed against the published format spec
- **zstd throughput 2x** (~120 → ~240 MB/s on literal-heavy data): the backward
  bit reader moved from per-bit loops to single unaligned u64 window loads, and
  the Huffman literal decoder to a peek-window loop over signed bit positions
- **Snappy decoder** (raw block format: varint preamble, literal/copy tags with
  1-4-byte lengths and offsets) — Blosc's last inner codec; validated against
  cramjam-written fixtures + a hand-built blosc-snappy container
- **Standalone numcodecs LZ4 codec** in zarr v2 (4-byte size prefix + block),
  live-validated
- **Zarr v3 sharding codec** (`sharding_indexed`): shard files decode their
  (offset, nbytes) inner-chunk index (CRC-32C verified, start/end locations),
  inner codec chains (gzip/zstd/...), sparse shards via `fill_value` —
  live-validated against zarr-python sharded stores incl. the zstd default
- **Zarr v3 writing**: `ai.save_zarr(..., zarr_format=3)` emits `zarr.json`
  array/group nodes with gzip-codec chunks under `c/` — `zarr.open_group`
  reads the output
- **Randomized cross-writer hardening** (`test_interop_fuzz_roundtrip_py`):
  seeded random shapes (rank 0-3), nested names, extreme f32 values, and
  scalars roundtrip through all nine array writers; the pass flushed out and
  fixed three real bugs (zarr rank-0 chunk-key conventions in both writers,
  a missing zarr branch in the numpy save fast path, and a msgpack→flax
  format alias)

### Added (interop wave 13: numpy arrays in any pickle — PaddlePaddle, sklearn)
- **Generic pickle array IO** (`ai.load_pickle_arrays` / `ai.save_pickle_arrays`):
  the pickle VM now reconstructs numpy ndarrays anywhere in a pickled object
  graph — `_reconstruct`+`BUILD` states (protocols 2-4 incl. the protocol-2
  `_codecs.encode` latin-1 byte path), protocol-5 `_frombuffer` reduces,
  numpy scalars, big-endian dtypes, Fortran-order conversion, and
  `NEWOBJ`-built objects whose `__dict__` holds arrays (sklearn estimators)
- Covers **PaddlePaddle `.pdparams`** state dicts and scikit-learn model pickles;
  names are dotted object-graph paths
- Writer emits pickles plain `pickle.load` + numpy deserializes (F32 C-order)
- Pickle VM: `BUILD` now keeps state on symbolic objects, `NEWOBJ`/`NEWOBJ_EX`
  and protocol-5 `BYTEARRAY8` opcodes supported (torch stub extraction
  unwraps `Build` transparently)
- `ai.load_arrays` / `save_arrays` detect and write generic pickles
  (`.pkl`/`.pickle`/`.pdparams`)
- Cross-validated against CPython `pickle` + numpy both directions, protocols 2-5
- **`ai.load_arrays(path, numpy=True)` fast path**: decoded bytes go straight to
  `np.frombuffer` float32 ndarrays instead of nested lists — ~80 ms → ~7 ms for a
  1.3M-value file (numpy optional, plain lists remain the default)
- **Sharded safetensors writing** (`ai.save_safetensors_sharded`): greedy packing
  into numbered `model-XXXXX-of-XXXXX.safetensors` shards under a size budget +
  the HF `weight_map` index; `load_arrays` reads shard indexes generically
  (safetensors or torch `.bin` shards) and reports format `"sharded"`
- **`ai.save_arrays` numpy fast path**: all-ndarray inputs move as `tobytes`
  bytes instead of float lists (byte-identical output, ~2x on 5 MB saves)
- **Deterministic diffusion init**: `Diffusion::new_with_seed` (+ seeded
  `Conv2d`/VAE/UNet constructors) — the stochastic loss-decrease training tests
  no longer flake on unlucky random inits (was intermittent under full-load
  `cargo test --workspace`)
- **ZIP64 reading**: EOCD64 + locator + per-entry `0x0001` extra fields — `.npz`
  and `.pt` archives over 4 GiB (and anything `zipfile` writes with
  `force_zip64`) now load; cross-validated against CPython `zipfile`
- **Gzip-compressed HDF5 writing** (`ai.save_h5(..., compress=True)`): one
  deflate chunk per dataset behind a raw-data-chunk B-tree (padded to libhdf5's
  fixed node allocation) + v1 filter pipeline; h5py reports `gzip` compression
  and reads values exactly; full-circle h5py roundtrip passes
- **dtype-parameterized NumPy writes**: `ai.save_npy` / `ai.save_npz` accept
  `dtype=` in numpy spellings (`f2`/`f4`/`f8`, `i1`-`i8`, `u1`-`u8`, `b1`) —
  `numpy.load` reports the exact dtype; integer conversion truncates like
  `astype`
- **Half-precision safetensors writes**: `ai.save_safetensors(..., dtype="f16")`
  / `"bf16"` (HF conventions); dtype verified by the official package, BF16
  roundtrips through our reader
- **Default-format serializer rewrite**: hand-rolled parallel JSON writer
  (manual digit expansion, per-tensor fragments, byte-identical to serde) and
  parallel tensor-entry parsing — save ~60 ms → ~50 ms, load ~55 ms → **~30 ms**
  on 1.3M params (cumulative vs the original serde `Value` path: 337→50 / 487→30)
- **GGUF array writes in any encodable type**: `ai.save_gguf_arrays(...,
  dtype="f16"/"q8_0"/"q4_k"/...)` — verified against gguf-py's `dequantize`
- **Zarr v2 stores** (`ai.load_zarr` / `ai.save_zarr`): from-scratch reader for
  directory stores — group trees, chunk grids with edge padding, `fill_value`
  for missing chunks, zlib/uncompressed chunks, all numeric dtypes — and a
  writer `zarr.open_group` reads; wired into `load_arrays`/`save_arrays`
  (`.zarr` extension, `"zarr"` detection); cross-validated against zarr-python
  both directions; blosc stores rejected with a re-encode hint
- **npy decode fast paths**: bulk `<f4`/`<f8`/`<f2` decoding skips per-element
  dtype dispatch — `.npz` checkpoint loads ~34 ms → ~17 ms (1.3M params)
- **From-scratch LZ4 + Blosc decoders**: Blosc1 frames (zarr's classic default
  compressor) decode completely — header flags, per-block starts, split
  byte-lane streams, byte shuffle, memcpy mode, LZ4 or zlib inner codecs —
  so default `Blosc(cname="lz4")` zarr stores read without any C library;
  validated against numcodecs fixtures + live zarr-python stores
- **BloscLZ decoder** (numcodecs' own default codec): FastLZ-family token
  stream with 255-run length extensions and far-distance escapes, verified
  byte-for-byte against captured c-blosc output + live stores
- **Bit-shuffle undo**: bitplane transpose (byte lane × bit × packed elements,
  8·typesize-aligned region + raw tail) — `shuffle=Blosc.BITSHUFFLE` stores
  read across lz4/blosclz/zlib inner codecs
- **Zarr v3 reading**: `zarr.json` array/group nodes, regular chunk grids,
  `c/`-prefixed chunk keys, `bytes` endian codec + `gzip` (RFC-1952 framing,
  from-scratch CRC-checked) or `blosc` compressors, all v3 numeric data
  types; live-validated including group trees; v3's zstd default rejected
  with a codec-naming error

### Added (interop wave 12: oldest and newest — legacy GGML, GGUF v1, TFLite, TorchScript)
- **Legacy GGML/GGMF/GGJT reader** (`ai.load_ggml_legacy`): the pre-GGUF llama.cpp
  containers, including the original f32-scale Q4_0/Q4_1 block layouts with
  consecutive-pair nibble packing (pre-GGJT-v2) and the modern layouts (GGJT v2/v3);
  hparams + scored vocab exposed
- **GGUF v1 read support**: the oldest GGUF revision (u32 counts, lengths, dims)
  parses alongside v2/v3 in `read_gguf`, `gguf_info`, and `load_gguf_arrays`
- **TFLite reader** (`ai.load_tflite`): from-scratch flatbuffer wire-format walker
  over the Model schema — subgraph tensors, inline + out-of-band (TF ≥ 2.13)
  buffers, INT8/UINT8/INT32 quantization dequantized per-tensor or per-channel,
  f16/bf16/all-int decode; validated against live TensorFlow conversions (float +
  dynamic-range int8) and a committed converter fixture
- **TorchScript archives**: `torch.jit.save` zips (`constants.pkl` tuple + shared
  `data/` storages) read through the existing pickle VM as `constants.N` arrays
- **Legacy llama models load as Chatbots**: `ai.load("model.ggjt")` adapts
  `tok_embeddings` / `layers.N.attention.wq` / SwiGLU `w1/w2/w3` / RMSNorm names
  through the HF fusion pipeline and generates
- `ai.load_arrays` / `detect_arrays_format` detect all of the above by content
- Tests: +16 Rust and +10 pytest (`test_interop_tflite_ggml_legacy_py`) — totals 540 / 964

### Added (interop wave 9: default-format fast path, block-parallel GGUF writes, safetensors arrays, Flax)
- **5–6x faster default checkpoint format**: the `mmn-safetensors-v1` /
  `mmn-classifier-v1` / `mmn-diffusion-v1` JSON paths moved from `serde_json::Value`
  trees to typed `TensorEntry` structs plus a hand-rolled structural scanner
  (`mmn_json.rs`, tight digit loops for byte arrays, serde as validation fallback):
  save ~337 ms → ~60 ms, load ~487 ms → ~80 ms on a 1.3M-param model; file bytes
  unchanged
- **Fast checkpoint detection**: `detect_checkpoint_kind` and the sharded-index
  probe scan only the top-level `format` / `weight_map` keys instead of
  `Value`-parsing whole multi-megabyte files (~208 ms → ~25 ms)
- **Block-parallel GGUF quantized writes**: large tensors split into block-aligned
  ~64K-element segments encoded by a work-stealing pool — q6_k/q4_k exports now
  scale across cores even with few big matrices (~65 ms → ~38 ms on 4 cores);
  segmented output is regression-tested byte-identical to whole-tensor encoding
- **Generic safetensors arrays**: `ai.load_safetensors` reads every dtype in the
  spec (BOOL through F64/I64/U64) into floats, `ai.save_safetensors` writes F32
  files the official package loads — cross-validated both directions against
  `safetensors.numpy` (and `safetensors.torch` for BF16)
- **Flax / JAX checkpoints**: from-scratch MessagePack codec (every wire type
  including all ext forms) + the `flax.serialization` ndarray ExtType convention;
  `ai.load_flax` flattens pytrees to `/`-joined names (all numpy dtypes + JAX
  `bfloat16`), `ai.save_flax` writes trees `flax.serialization.from_bytes` reads —
  cross-validated against the official `msgpack` package both directions
- **Universal array IO**: `ai.load_arrays` detects any container by content
  (GGUF, PyTorch zip/legacy, npy/npz, HDF5/Keras, safetensors, Flax msgpack,
  ONNX, TF checkpoint prefixes) and `ai.save_arrays` picks the writer from the
  extension; `ai.load_gguf_arrays` / `ai.save_gguf_arrays` expose GGUF as a
  plain tensor container (every quantization dequantizes)
- Tests: +25 Rust (scanner, segmented-encode identity, st_arrays, msgpack, flax,
  arrays_auto) and +61 pytest (`test_interop_safetensors_arrays_py`,
  `test_interop_flax_py`, `test_universal_arrays_py`) — totals 524 / 952

### Added (interop wave 8: IQ4 encoders, vision GGUF export, SavedModel dirs)
- **IQ4_NL / IQ4_XS encoders** (`gguf-iq4_nl` / `gguf-iq4_xs` exports): from-scratch
  port of ggml's `ntry` scale search over the non-linear codebook with binary-search
  nearest-entry lookup — the only IQ types encodable without calibration data, and
  another capability the reference Python package lacks; blocks decode identically
  in gguf-py
- **Vision chatbot GGUF export**: vision prefix tensors travel under mmproj-style
  `v.*` names with an `mmn.vision` flag and roundtrip through `ai.load` — the last
  export gap is closed
- **TF SavedModel directories**: `ai.load_tf_checkpoint("saved_model_dir")` resolves
  the `variables/variables` bundle (validated against `tf.saved_model.save` output)
- **Parallel HF-safetensors import decode** (F16/BF16 conversion across cores)
- Regression guard: every supported GGML type id roundtrips through `from_id`
  (caught a dropped IQ4_XS id during this wave)
- Tests: +7 Rust and +7 pytest (`test_interop_wave8_py`)

### Added (interop wave 7: zero format dependencies, complete k-quant encoder set)
- **From-scratch safetensors codec** (`st_codec.rs`) replaces the external
  `safetensors` crate — header JSON + offset validation + aligned serialization;
  the dependency is gone from the tree entirely, making the whole format layer
  zero-external-libraries. Cross-validated: the official `safetensors` Python
  package opens our files (`safe_open` + metadata) and we import its
  `safetensors.numpy.save_file` output
- **Complete k-quant encoder set**: Q2_K (MAD-variant scale/min search), Q3_K
  (`make_q3_quants` iterative RMSE refinement with the 6-bit scale shuffle), and
  Q8_K join Q4_K/Q5_K/Q6_K; new `gguf-q2_k` / `gguf-q3_k` export formats; blocks
  decode identically in gguf-py; reconstruction error forms a strict Q2→Q8
  quality ladder
- **Parallel GGUF tensor payload encoding** in the writer (quantization searches
  dominate k-quant export time)
- Tests: +9 Rust (encoder roundtrips, quality ladder, st_codec roundtrips/corrupt
  inputs) and +8 pytest (`test_interop_wave7_py`: official-package cross-reads in
  both directions, loss parity through the new container, reference-decode
  identity for Q2_K/Q3_K)

### Added (interop wave 6: writers for HDF5 / TF checkpoint / ONNX — every ecosystem bidirectional)
- **HDF5 writer** (`ai.save_h5`): superblock v0, symbol-table groups with the
  fixed-allocation B-tree v1 + SNOD node sizes libhdf5 requires, local heaps with
  free-list descriptors, contiguous F32 datasets with h5py's exact IEEE-float
  datatype encoding, nested groups via `/` names — **`h5py.File` reads the output
  natively** (verified incl. 30-dataset multi-SNOD groups)
- **TF checkpoint v2 writer** (`ai.save_tf_checkpoint`): sorted LevelDB table
  (data/metaindex/index blocks, masked-CRC32C trailers, 48-byte footer) + raw data
  shard — **`tf.train.load_checkpoint` reads the output**; full-circle test
  (TF write → our read → our write → TF read) passes bit-for-bit
- **ONNX writer** (`ai.save_onnx`): ModelProto with ir_version/opset/graph
  initializers via shared protobuf emit helpers — **passes
  `onnx.checker.check_model`** and loads with `onnx.load`
- **Reference-exact MXFP4 and TQ1_0 encoders** (E8M0 scale selection + FP4
  codebook nearest; 5-trits-per-byte ternary packing) joining the byte-exact
  classic-quant encoder set; GGUF writer now covers TQ1_0/MXFP4 targets
- **Parallel HDF5 chunk decompression** (gzip inflation across cores)
- Tests: +12 Rust (writer roundtrips, multi-SNOD, ternary-exact encode) and
  +11 pytest (`test_interop_wave6_py`: h5py/tf/onnx read our writers,
  full-circle TF, encoder reference-decode equality)

### Added (interop wave 5: ONNX, chunked HDF5, byte-exact encoders, fast inflate)
- **ONNX reader** (`ai.load_onnx`): from-scratch protobuf wire-format walker
  (shared `interop/proto.rs`) extracting every graph initializer — packed/repeated
  dims, all numeric tensor types via `raw_data` or typed fields, external-data
  models rejected with a re-export hint. Validated against models saved by the
  official `onnx` package
- **Chunked HDF5**: B-tree v1 chunk index with edge-chunk clipping, plus the
  **gzip filter** (zlib wrapper undone by our inflate, Adler-32 verified) and
  **shuffle filter** (byte transpose) — `compression="gzip", shuffle=True` h5py
  files now read; committed `tests/fixtures/chunked.h5` + live h5py odd-shape tests
- **Classic-quant encoders, byte-identical to the reference**: Q4_1, Q5_0, Q5_1,
  TQ2_0 join Q4_0/Q8_0 (ggml's exact truncating rounding); verified as
  re-quantization fixed points against gguf-py; new `gguf-q4_1`/`q5_0`/`q5_1`
  export formats
- **Fast inflate**: 10-bit one-hit Huffman lookup table over a 64-bit bit
  accumulator (canonical bit-walk only for rare >10-bit codes) — the classic
  zlib fast path, from scratch
- Tests: +9 Rust (proto walker, ONNX handcrafted models, chunked/gzip/shuffle
  fixture, Adler-32) and +9 pytest (`test_interop_wave5_py`: real-onnx equality,
  live h5py gzip+shuffle with edge chunks, encoder fixed-point checks)

### Added (interop wave 4: k-quant encoders, TF checkpoint v2, real-TF validation)
- **K-quant encoders** (`gguf-q4_k` / `gguf-q5_k` / `gguf-q6_k` exports): from-scratch
  ports of ggml's reference quantization searches (`make_qx_quants` iscale
  refinement, `make_qkx2_quants` joint scale+min least squares, round-half-to-even
  `nearest_int`) — the reference llama.cpp Python package cannot encode k-quants at
  all. Our blocks decode identically in gguf-py; Q6_K reconstruction < 2% relative
  RMSE. Export now follows ggml's **row-wise quantization rule** (fastest dimension
  must be a block multiple, else F32 fallback)
- **TensorFlow checkpoint v2 reader** (`ai.load_tf_checkpoint("ckpt")`): from-scratch
  LevelDB-table parsing (prefix-compressed blocks, varint handles, masked-CRC32C
  trailers via a from-scratch CRC-32C/Castagnoli), minimal protobuf walk of
  `BundleHeaderProto`/`BundleEntryProto`/`TensorShapeProto`, multi-shard data files,
  per-tensor checksum verification, all numeric DataTypes → f32
- **Real-TensorFlow validation**: committed fixtures written by TF 2.21
  (`tests/fixtures/tf/`: `.keras`, `.weights.h5`, checkpoint index+data) exercise the
  HDF5 and TF-checkpoint readers in CI without TensorFlow; live tests (importorskip)
  compare against `model.get_weights()` / `tf.train.load_checkpoint` exactly
- **Parallel ZIP inflation**: `read_zip` decompresses entries across cores
  (compressed npz / pt archives)
- `examples/interop_benchmark.py`: save/load timing + file sizes across all formats
  (safetensors 5.1 MiB → Q4_K 0.7 MiB on the demo model); wired into smoke + pytest
- Tests: +7 Rust (k-encode roundtrips/quality ordering, TF fixture, CRC corruption,
  narrow-row F32 fallback) and +11 pytest (`test_interop_wave4_py`: k-quant
  reference-decode identity + quality bounds, TF fixtures, live TF/Keras equality)

### Added (interop wave 3: complete GGML quant matrix, BPE vocabs, sharded HF, deflate)
- **The IQ codebook-grid family, complete**: IQ1_S, IQ1_M, IQ2_XXS, IQ2_XS, IQ2_S,
  IQ3_XXS, IQ3_S dequantize from scratch — QuIP#-style lattice codebooks ship as
  compact 2-/4-bit-packed tables (`gguf_iq_grids.rs`) decoded once at runtime; the
  sign-parity table is generated, not embedded. Plus **NVFP4** (unsigned-E4M3-scaled
  FP4, ggml's newest type). Every GGML tensor type that exists in current ggml now
  loads
- **Reference cross-validation**: all 24 quant codecs verified against llama.cpp's
  official `gguf` Python package — identical dequantization on random payloads,
  agreement on reference-quantized float data, and our GGUF exports parse in the
  reference `GGUFReader`. New `_native.dequantize_ggml(type, bytes, numel)` API;
  `gguf` added to dev extras
- **GPT-2 byte-level BPE** (`Gpt2BpeEncoder` in mmn-data): from-scratch
  bytes↔unicode bijection, ranked-merge BPE, approximate GPT-2 pretokenization;
  `ai.load_gguf_bpe_tokenizer(path)` extracts gpt2-model GGUF vocabs (GPT-2 /
  Llama-3 / Qwen style) with model-aligned ids; `Gpt2BpeEncoder.from_vocab` accepts
  HF vocab+merges directly; `TextEncoderRef::Gpt2` wires it into train/generate
  plumbing
- **Sharded HF checkpoints**: `ai.load("pytorch_model.bin.index.json")` /
  `model.safetensors.index.json` resolve `weight_map` shards relative to the index
  (safetensors and torch shard formats mix freely); new
  `CheckpointKind::ChatbotSharded` detection
- **From-scratch DEFLATE compressor** (fixed-Huffman + 32-deep hash-chain LZ77):
  `ai.save_npz(path, arrays, compress=True)` mirrors `np.savez_compressed`,
  cross-checked against CPython's `zlib`; incompressible entries stay stored
- **Performance**: PyTorch storage decode parallelized across cores (matching GGUF
  dequant); IQ grids cached in `OnceLock`
- Tests: +25 Rust (IQ layouts/dispatch, grids, deflate roundtrips incl. LCG
  incompressible data, sharded st/pt indexes, gpt2 BPE) and +43 pytest
  (`test_interop_quant_crossval_py`, `test_interop_wave3_py`)

### Added (interop wave 2: full quant coverage, HDF5, legacy torch, GGUF tokenizers)
- **Every practical GGML quantization**: Q2_K / Q3_K / Q5_K / Q8_K join Q4_K/Q6_K
  (full k-quant family); IQ4_NL / IQ4_XS non-linear lookup quants; TQ1_0 / TQ2_0
  ternary BitNet quants; **MXFP4** (E8M0-scaled FP4, gpt-oss era); Q8_1. Removed
  ids (Q4_2/Q4_3, repacked Q4_0_x_x) and grid-codebook IQ1/IQ2/IQ3 rejected with
  actionable messages. GGUF writer gains **F16 and Q4_0** encodings
  (`format="gguf-f16"` / `"gguf-q4_0"`)
- **GGUF inspection + embedded tokenizers**: `ai.gguf_info(path)` returns version,
  full metadata (incl. `tokenizer.chat_template`), and tensor summaries from a
  header-only incremental read (multi-GB files never fully load);
  `ai.load_gguf_tokenizer(path)` converts embedded SentencePiece vocabs
  (`▁` markers, `<0xNN>` byte tokens) into a `UnigramEncoder` with model-aligned
  ids; `bot.save(..., format="gguf", unigram_encoder=tok)` embeds the vocab for a
  **self-contained model file** (llama.cpp convention)
- **Legacy PyTorch (pre-1.6) checkpoints**: the non-zip pickle-stream format
  (magic `0x1950a86a20f9469cfc6c`, protocol/sys-info pickles, appended raw
  storages) is auto-detected by `ai.load` / `ai.load_pt`; pickle VM adds LONG/
  big-LONG1 opcodes and prefix parsing
- **From-scratch HDF5 reader** — the TensorFlow/Keras bridge: superblock v0/1,
  v1 object headers + continuations, B-tree v1 symbol-table groups, local heaps,
  compact/contiguous datasets of all fixed/float types. `ai.load_h5(path)` and
  `ai.load_keras(path)` (`.keras` v3 zip archives, `.weights.h5`, plain `.h5`);
  committed `tests/fixtures/simple.h5` validates against real h5py output
- **Performance**: GGUF tensors dequantize in parallel across all cores; CRC-32
  table cached in a `OnceLock`; unigram Viterbi uses a hash-map piece index
  (O(n·window) instead of a linear vocab scan — required for 32k+ GGUF vocabs)
- `UnigramEncoder::from_pieces` builds encoders from external vocabularies,
  preserving id order; Viterbi window sizes to the longest piece
- Tests: +32 Rust (every quant codec against hand-built blocks, legacy torch
  streams, HDF5 fixture, embedded tokenizer roundtrip, parallel dequant) and
  +31 pytest (`test_interop_h5_py`, `test_interop_gguf_info_py`,
  `test_interop_legacy_pt_py`, self-contained GGUF chat); h5py added to dev
  extras for cross-validation

### Added (global format interop: GGUF, PyTorch, NumPy/TensorFlow — all from scratch)
- **GGUF read/write with zero llama.cpp code**: from-scratch container parser (v2/v3
  headers, typed metadata KV, aligned tensor data) plus block-dequant codecs for
  Q4_0 / Q4_1 / Q5_0 / Q5_1 / Q8_0 / **Q4_K / Q6_K** / F16 / BF16 / F64 / ints.
  llama.cpp tensor names (`token_embd.weight`, `blk.N.attn_q.weight`, …) map onto the
  MMN transformer with SwiGLU fusion, GQA head counts, and RoPE theta from `{arch}.*`
  metadata. `bot.save(path, format="gguf")` / `"gguf-q8_0"` writes GGUF back out;
  loaded models generate through the existing KV-cache engine
- **PyTorch `.pt` state dicts without torch**: from-scratch pickle virtual machine
  (protocols 2–4: memo, FRAME, persistent IDs, strided views, F16/BF16/int storages)
  + from-scratch ZIP reader/writer. Imports HF llama-style state dicts; exports are
  `torch.load`-compatible (verified against CPython's own `pickle`); `_mmn_meta`
  entry preserves shape/seed/RoPE across roundtrips. Generic array API:
  `ai.save_pt` / `ai.load_pt`
- **NumPy `.npy`/`.npz` without numpy**: from-scratch NPY 1.0/2.0 codec (all common
  dtypes, big-endian, Fortran order) and from-scratch **RFC 1951 DEFLATE inflate** so
  `np.savez_compressed` archives read too. `bot.save(path, format="npz")` writes
  `numpy.load`-compatible checkpoints (the TensorFlow/Keras interchange path);
  `ai.save_npy` / `ai.load_npy` / `ai.save_npz` / `ai.load_npz` accept nested lists,
  numpy arrays, or torch tensors (anything with `.tolist()`)
- **Universal detection**: `detect_checkpoint_kind` now recognizes GGUF magic and ZIP
  archives (torch vs npz); `ai.load()` / `Chatbot.load()` open every format with no
  format argument. New `CheckpointKind::{ChatbotGguf, ChatbotNpz, ChatbotTorch}`
- **Tensor op expansion (mmn-core)**: `sub`, `div`, `neg`, `exp`, `log`, `sqrt`,
  `abs`, `pow_scalar`, `sigmoid`, `tanh`, `clamp`, `add_scalar`, `mul_scalar`,
  `reshape`, `transpose`, `flatten`, `sum_axis`, `mean_axis`, `max_value`,
  `min_value`, `argmax`, `argmin`, `argmax_rows`, `from_vec`, `to_vec` — unary math
  ops register autograd nodes
- External MLP checkpoints with only `up_proj`/`ffn_up` (no gate) now import as the
  FFN weight instead of failing
- New docs: `docs/interop.md`; example `examples/global_formats_roundtrip.py`
- Tests: ~60 Rust (interop modules + elementwise) and 50 pytest across
  `test_interop_*` / `test_universal_formats_py.py`, including CPython-pickle
  cross-checks and `numpy.load` compatibility checks (numpy now a dev extra)

### Fixed
- Flaky `mean_denoise_loss_masked_decreases_after_fixed_t_training`: unseeded
  `Diffusion::new()` occasionally overshoots at lr=0.05 — the regression now
  allows bounded fresh-model retries while keeping the loss-decrease contract
- Workspace is clippy-clean on Rust 1.96 (`repeat_n`, `slice::from_ref`,
  `str::len` modernizations in mmn-train / mmn-models / mmn-data tests)

## 0.1.0 — 2026-05-31

### Added (beginner API overhaul: train/chat/save/load, in-memory data, typed errors)
- **Method-style workflow on every model**: `bot.train(data, epochs=...)`, `bot.chat(prompt)`,
  `model.save(path)`, `Model.load(path)`, `clf.predict_label(text)`, `clf.train(...)`, `diff.train(...)`
- **Universal `ai.load(path)`** auto-detects model family (Chatbot / Classifier / Diffusion)
  and format (JSON, binary HF safetensors, bin stub) via new `mmn-io detect_checkpoint_kind`;
  wrong-family `Model.load` raises `ValueError` naming the actual family
- **In-memory datasets**: `DatasetQA(data=[{"input": ..., "output": ...}])` and
  `DatasetClassification(data=[{"text": ..., "label": ...}])` — no files needed (`format == "memory"`)
- **Training returns loss history**: `Train` / `TrainClassifier` / `TrainDiffusion` and the
  `model.train` methods return one mean-loss value per epoch; `TrainConfig(verbose=True)`
  prints `[magicmindnet] epoch i/n - mean loss ...`
- **Validation instead of panics/silence**: unknown `TrainConfig.optimizer` raises `ValueError`
  (was silently treated as AdamW); `optimizer="muon"` now routes matrix weights through Muon;
  `Chatbot(use_learned_pos_embed=True, use_rope=True)` raises `ValueError` (was a Rust panic);
  unknown `autoset` presets and `vocab_size=0` raise `ValueError` (were silent fallbacks)
- **IDE typing**: package ships `py.typed` + complete `_native.pyi` stubs
- Python-style booleans in `Chatbot` / `TrainConfig` reprs; `vision.py` cleanup (top-level imports)
- New beginner docs: `docs/getting_started.md`, README beginner quick start, `examples/hello_ai.py`
- Tests: `test_model_train_methods.py`, `test_model_save_load.py`, `test_dataset_in_memory.py`,
  `test_constructor_validation.py`, `test_chatbot_chat.py`, `test_classifier_predict_label.py`,
  `test_type_stubs.py`; Rust `detect` module tests + optimizer/loss-history regressions

### Added (image dataset path resolvers + diffusion merge demo)
- Python `resolve_image_path`, `image_path_at`, `prompt_at` on `DatasetImageGen`
- Python `resolve_mask_path`, `mask_path_at` on `DatasetImageEdit`; inpaint sample uses manifest paths
- `examples/diffusion_merge_demo.py`; `import_diffusion_rejects_classifier_checkpoint`
- Gitignore example PNG artifacts; untrack `_inpaint_sample.png`

### Added (Diffusion parameters getter + quantize roundtrip example)
- `Diffusion.parameters()` counts VAE + UNet conv weights (Rust + Python getter/repr)
- `examples/diffusion_quantize_roundtrip.py`; pytest for diffusion roundtrip examples
- `diffusion_export_import_preserves_parameter_count` Rust regression

### Added (diffusion benchmark example + masked fixed-t loss regression)
- `examples/diffusion_benchmark.py` for image_gen and `--edit` inpainting mean-loss deltas
- `mean_denoise_loss_masked_decreases_after_fixed_t_training` Rust regression
- `eval_mean_loss diffusion-edit --train` pytest; smoke coverage

### Added (eval_mean_loss diffusion modes + mean loss training regression)
- `eval_mean_loss.py` modes `diffusion` and `diffusion-edit` with optional `--train`
- `mean_denoise_loss_decreases_after_fixed_t_training` Rust regression
- Diffusion section in `docs/checkpoint_coverage.md`; README diffusion status updated

### Added (diffusion mean denoise loss + IO matrix + edit roundtrip)
- `mean_denoise_loss` / `mean_denoise_loss_masked`; Python `Diffusion.compute_mean_denoise_loss`
- Parametric diffusion checkpoint IO matrix (`test_io_diffusion_matrix_py.py`)
- `quantize_diffusion` export/import sample roundtrip test
- Example `diffusion_edit_roundtrip.py`; masked train reduces loss at fixed `t` regression
- **309** Rust tests / **625** pytest (`verify_gate`)

### Added (diffusion inpaint sampling + quantize + PNG export)
- `sample_latent_inpaint` / `sample_image_inpaint` preserve unmasked latent from source image
- `denoise_loss_masked` + Python `denoise_loss_on_image_masked`
- `quantize_diffusion` (`int8`/`int4` on VAE + UNet conv weights); Python `quantize_diffusion`
- Python `sample_inpaint_rgb_patch`, `sample_rgb_patch_to_png`, `sample_inpaint_rgb_patch_to_png`
- Example `diffusion_inpaint_sample.py`; tests `test_diffusion_inpaint_py.py`
- **306** Rust tests / **595** pytest (`verify_gate`)

### Added (inpainting diffusion + RGB decode clamp + merge)
- `Diffusion::decode_latent` clamps RGB to `[0, 1]`; `sample_rgb_patch` stays in unit interval
- `train_step_denoise_masked` mask-weighted UNet step; `train_diffusion_edit` on `DatasetImageEdit`
- `grayscale_mask_tensor_from_image_*`, `write_rgb_nchw_tensor_to_png`; `DatasetImageEdit::resolve_*_path`
- `merge_diffusion` averages VAE + UNet weights (`mmn-diffusion-v1` models)
- Python `TrainDiffusion` accepts `DatasetImageEdit`; `merge_diffusion` alias
- Fixtures `tests/fixtures/samples/photo.png`, `mask.png`; example `diffusion_edit_train.py`
- Tests: `sample_image_rgb_is_clamped_*`, `train_step_denoise_masked_*`, `train_diffusion_edit_*`, `merge_diffusion_*`, `test_train_diffusion_edit.py`
- **301** Rust tests / **587** pytest (`verify_gate`)

### Added (diffusion sampling + checkpoint IO)
- `VaeDecoder` + `Diffusion::sample_latent` / `sample_image` / `decode_latent` (seeded reverse-diffusion loop)
- `export_diffusion` / `import_diffusion` (`mmn-diffusion-v1` JSON: VAE enc/dec + UNet conv weights)
- Python `sample_rgb_patch`, `export_diffusion` / `import_diffusion` aliases; examples `diffusion_sample.py`, `diffusion_roundtrip.py`
- Tests: `sample_latent_and_image_*`, `diffusion_export_import_*`, `test_diffusion_sample_io_py.py`, `test_train_diffusion_rejects_qa_dataset`
- **292** Rust tests / **581** pytest (`verify_gate`)

### Added (vision-prefix KV slide + diffusion training)
- RoPE KV-cache slide evicts oldest **text** row at `n_vision_prefix` (`slide_rope_kv_window_at`, `truncate_at`, `rerope_k_cache_shift_down_from`)
- Vision + RoPE sliding-window generation parity past `max_seq_len` (`vision_sliding_window_past_max_ctx_*`); RoPE models honor `max_seq_len` as generation context
- UNet2D `forward_with_cache` + `backward`; `Diffusion::train_step_denoise` / `denoise_loss` (deterministic noise per `(image, t)`)
- `train_diffusion` on `DatasetImageGen`; Python `TrainDiffusion`, `denoise_loss_on_image`
- `rgb_nchw_tensor_from_image_*`, `DatasetImageGen::resolve_image_path`, fixture `tests/fixtures/samples/cat.png`
- Tests: `rope_kv_slide_at_prefix_*`, `train_diffusion_*`, `test_train_diffusion.py`; example `examples/diffusion_train.py`
- **287** Rust tests / **571** pytest (`verify_gate`)

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
