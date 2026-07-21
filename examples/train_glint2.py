#!/usr/bin/env python3
"""Train Glint-2 *exactly* on FineWeb-Edu with MagicMindNet only (no torch/numpy).

Architecture matches Glint-Research/Glint-2 checkpoint `model_config` + Space
`app.py` forward (shared loop + coda + final_norm + LoopLoRA + window):

  vocab_size=4096, dim=96, n_heads=8, n_layer=1 (shared)
  n_loops=8, max_loops=16, lora_rank=4, ffn_hidden=2112
  prelude_layers=0, coda_layers=1, attention_window=256
  rope_theta=10000, RMSNorm, SwiGLU, tied embeds, loop_embed, final_norm

Tokenizer: official Glint-2 `tokenizer.json` via `Gpt2BpeEncoder.from_hf_tokenizer_json`
(GPT-2 byte-level BPE, vocab 4096). Dataset: HuggingFaceFW/fineweb-edu (full train
split by default; use --subset sample-10BT for the 10B-token sample).

Checkpoint step target from the public release: 12_178_000 optimizer steps.
Default --steps matches that; override for smoke runs.

Usage:
  pip install magicmindnet datasets huggingface_hub
  python examples/train_glint2.py --demo --steps 20
  python examples/train_glint2.py --subset sample-10BT --steps 1000 --shard-rows 512
  python examples/train_glint2.py   # full FineWeb-Edu, 12.178M steps

Deps: MagicMindNet for all ML. `datasets` / `huggingface_hub` only fetch FineWeb
text and Glint's tokenizer.json — never torch/numpy in this file.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import magicmindnet as ai

# Exact Glint-2 hyperparams (from Glint-Research/Glint-2 checkpoints/glint-2.pt).
GLINT2_VOCAB = 4096
GLINT2_DIM = 96
GLINT2_HEADS = 8
GLINT2_FFN = 2112
GLINT2_N_LOOPS = 8
GLINT2_MAX_LOOPS = 16
GLINT2_LORA_RANK = 4
GLINT2_CODA = 1
GLINT2_PRELUDE = 0
GLINT2_ATTN_WINDOW = 256
GLINT2_ROPE = 10000.0
GLINT2_MAX_SEQ = 4096
GLINT2_RELEASE_STEPS = 12_178_000
GLINT2_REPO = "Glint-Research/Glint-2"
FINEWEB_REPO = "HuggingFaceFW/fineweb-edu"


def build_glint2(*, max_seq_len: int, seed: int) -> ai.Chatbot:
    """Construct the Glint-2 Chatbot — same knobs as the released weights."""
    return ai.Chatbot(
        vocab_size=GLINT2_VOCAB,
        n_layer=1,
        d_model=GLINT2_DIM,
        n_heads=GLINT2_HEADS,
        ffn_dim=GLINT2_FFN,
        max_seq_len=max_seq_len,
        use_rope=True,
        rope_theta=GLINT2_ROPE,
        n_loops=GLINT2_N_LOOPS,
        max_loops=GLINT2_MAX_LOOPS,
        norm="rms",
        ffn="swiglu",
        tie_embeddings=True,
        loop_embed=True,
        final_norm=True,
        lora_rank=GLINT2_LORA_RANK,
        coda_layers=GLINT2_CODA,
        prelude_layers=GLINT2_PRELUDE,
        attention_window=GLINT2_ATTN_WINDOW,
        seed=seed,
    )


def load_glint2_tokenizer(path: str | Path | None = None) -> ai.Gpt2BpeEncoder:
    """Load the official Glint-2 HF tokenizer.json into MagicMindNet GPT-2 BPE."""
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(GLINT2_REPO, "tokenizer.json")
    enc = ai.Gpt2BpeEncoder.from_hf_tokenizer_json(str(path))
    if enc.vocab_size != GLINT2_VOCAB:
        print(
            f"warning: tokenizer vocab_size={enc.vocab_size} "
            f"(Glint-2 release uses {GLINT2_VOCAB})"
        )
    return enc


def demo_chunks() -> list[str]:
    base = [
        "Once upon a time, there was a little girl named Lily who loved the park.",
        "The sun is a star at the center of the solar system.",
        "FineWeb-Edu is a filtered educational web corpus used to train small language models.",
        "Transformers learn next-token prediction by compressing text into shared weights.",
        "A looped block reuses one transformer layer many times for depth without parameter cost.",
        "The history of the United States begins in the early modern period.",
        "Look, Mommy! I found it in the park! Tom said. He was curious.",
    ]
    return base * 64


def iter_fineweb_texts(*, subset: str | None, min_chars: int, max_chars: int):
    """Stream FineWeb-Edu texts (full train split or a named subset)."""
    from datasets import load_dataset

    kwargs: dict = {
        "path": FINEWEB_REPO,
        "split": "train",
        "streaming": True,
    }
    if subset:
        kwargs["name"] = subset
    ds = load_dataset(**kwargs)
    for ex in ds:
        text = (ex.get("text") or "").strip().replace("\n", " ")
        if len(text) < min_chars:
            continue
        yield text[:max_chars]


def train_shards(
    bot: ai.Chatbot,
    gpt2: ai.Gpt2BpeEncoder,
    texts_iter,
    *,
    steps: int,
    shard_rows: int,
    batch_size: int,
    lr: float,
    weight_decay: float,
    warmup_steps: int,
    out: Path,
    save_every: int,
) -> list[float]:
    """Shard the stream into DatasetCorpus batches and train until `steps`."""
    epoch_losses: list[float] = []
    step = 0
    shard_id = 0
    buf: list[str] = []
    t0 = time.perf_counter()

    def flush(rows: list[str]) -> None:
        nonlocal step, shard_id
        if not rows or step >= steps:
            return
        data = ai.DatasetCorpus(data=rows)
        cfg = ai.TrainConfig(
            epochs=1,
            batch_size=batch_size,
            learning_rate=lr,
            optimizer="adamw",
            weight_decay=weight_decay,
            lr_schedule="cosine",
            warmup_steps=min(warmup_steps, max(1, len(rows) // batch_size)),
            verbose=True,
        )
        losses = ai.Train(bot, data, cfg, gpt2_encoder=gpt2)
        epoch_losses.extend(losses)
        updates = max(1, (len(rows) + batch_size - 1) // batch_size)
        step += updates
        shard_id += 1
        elapsed = time.perf_counter() - t0
        print(
            f"[shard {shard_id}] rows={len(rows)} loss={losses[-1]:.4f} "
            f"steps≈{step}/{steps} elapsed={elapsed:.1f}s params={bot.parameters}"
        )
        if save_every > 0 and shard_id % save_every == 0:
            ckpt = out.with_name(f"{out.stem}_step{step}{out.suffix}")
            bot.save(str(ckpt))
            print(f"  saved {ckpt}")

    for text in texts_iter:
        buf.append(text)
        if len(buf) >= shard_rows:
            flush(buf)
            buf = []
            if step >= steps:
                break
    if step < steps and buf:
        flush(buf)

    out.parent.mkdir(parents=True, exist_ok=True)
    bot.save(str(out))
    meta = {
        "format": "mmn-glint2-train-v1",
        "steps_approx": step,
        "target_steps": steps,
        "params": bot.parameters,
        "arch": {
            "vocab_size": GLINT2_VOCAB,
            "d_model": GLINT2_DIM,
            "n_heads": GLINT2_HEADS,
            "ffn_dim": GLINT2_FFN,
            "n_loops": GLINT2_N_LOOPS,
            "max_loops": GLINT2_MAX_LOOPS,
            "lora_rank": GLINT2_LORA_RANK,
            "coda_layers": GLINT2_CODA,
            "prelude_layers": GLINT2_PRELUDE,
            "attention_window": GLINT2_ATTN_WINDOW,
            "rope_theta": GLINT2_ROPE,
            "norm": "rms",
            "ffn": "swiglu",
            "tie_embeddings": True,
            "loop_embed": True,
            "final_norm": True,
        },
        "dataset": FINEWEB_REPO,
        "tokenizer": GLINT2_REPO + "/tokenizer.json",
    }
    out.with_suffix(".train.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"saved {out} (+ {out.with_suffix('.train.json')})")
    return epoch_losses


def main() -> None:
    p = argparse.ArgumentParser(description="Glint-2 exact trainer (MagicMindNet only)")
    p.add_argument("--out", default="checkpoints/glint2.mmn")
    p.add_argument(
        "--steps",
        type=int,
        default=GLINT2_RELEASE_STEPS,
        help=f"optimizer-step budget (release used {GLINT2_RELEASE_STEPS})",
    )
    p.add_argument("--shard-rows", type=int, default=2048, help="FineWeb rows per Train() call")
    p.add_argument("--batch-size", type=int, default=8)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=0.01)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument(
        "--max-seq-len",
        type=int,
        default=4096,
        help="train context (Glint-2 exact config is 4096; use --fast or override for CPU runs)",
    )
    p.add_argument(
        "--fast",
        action="store_true",
        help="CPU smoke mode: sets max_seq_len=512 and shard_rows=64 (overrides --max-seq-len if smaller)",
    )
    p.add_argument(
        "--subset",
        default=None,
        help="FineWeb-Edu config name (e.g. sample-10BT). Default: full train split",
    )
    p.add_argument("--min-chars", type=int, default=80)
    p.add_argument("--max-chars", type=int, default=8000)
    p.add_argument("--tokenizer", default=None, help="path to Glint-2 tokenizer.json")
    p.add_argument("--save-every", type=int, default=50, help="save every N shards (0=off)")
    p.add_argument(
        "--demo",
        action="store_true",
        help="toy built-in corpus; no FineWeb download needed (combine with --fast for quick CI)",
    )
    p.add_argument(
        "--sample",
        default=None,
        help="after training, generate from this prompt (Glint default sampling)",
    )
    args = p.parse_args()

    if args.fast:
        args.max_seq_len = min(args.max_seq_len, 512)
        args.shard_rows = min(args.shard_rows, 64)

    bot = build_glint2(max_seq_len=args.max_seq_len, seed=args.seed)
    print(
        f"Glint-2 MMN: params={bot.parameters} n_loops={bot.n_loops}/{bot.max_loops} "
        f"coda={bot.coda_layers} window={bot.attention_window} "
        f"lora={bot.lora_rank} ffn_dim={bot.ffn_dim} seq={bot.max_seq_len}"
    )

    gpt2 = load_glint2_tokenizer(args.tokenizer)
    print(f"tokenizer vocab_size={gpt2.vocab_size}")

    if args.demo:
        texts_iter = iter(demo_chunks())
        # Demo: one small shard, step budget from --steps (treat as micro-updates).
        args.shard_rows = min(args.shard_rows, 128)
        args.steps = min(args.steps, 64)
    else:
        texts_iter = iter_fineweb_texts(
            subset=args.subset,
            min_chars=args.min_chars,
            max_chars=args.max_chars,
        )

    losses = train_shards(
        bot,
        gpt2,
        texts_iter,
        steps=args.steps,
        shard_rows=args.shard_rows,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        warmup_steps=args.warmup_steps,
        out=Path(args.out),
        save_every=args.save_every,
    )
    print("shard losses (last 8):", [round(x, 4) for x in losses[-8:]])

    prompt = args.sample or "Once upon a time"
    # Glint-2 generate.py defaults: temperature=0.15, top_k=5, repetition_penalty=1.05
    print(
        bot.generate(
            prompt,
            max_new_tokens=100,
            temperature=0.15,
            top_k=5,
            repetition_penalty=1.05,
            gpt2_encoder=gpt2,
        )
    )


if __name__ == "__main__":
    main()
