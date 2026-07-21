#!/usr/bin/env python3
"""Generate with a MagicMindNet Glint-2 checkpoint (exact sampling defaults).

Usage:
  python examples/generate_glint2.py "Once upon a time"
  python examples/generate_glint2.py --checkpoint checkpoints/glint2.mmn "The sun is"
"""

from __future__ import annotations

import argparse

import magicmindnet as ai

GLINT2_REPO = "Glint-Research/Glint-2"
LOOPS = 8


def load_tokenizer(path: str | None) -> ai.Gpt2BpeEncoder:
    if path is None:
        from huggingface_hub import hf_hub_download

        path = hf_hub_download(GLINT2_REPO, "tokenizer.json")
    return ai.Gpt2BpeEncoder.from_hf_tokenizer_json(path)


def main() -> None:
    p = argparse.ArgumentParser(description="Glint-2 MagicMindNet generate")
    p.add_argument("prompt")
    p.add_argument("--checkpoint", default="checkpoints/glint2.mmn")
    p.add_argument("--tokenizer", default=None)
    p.add_argument("--max-new-tokens", type=int, default=100)
    p.add_argument("--temperature", type=float, default=0.15)
    p.add_argument("--top-k", type=int, default=5)
    p.add_argument("--repetition-penalty", type=float, default=1.05)
    p.add_argument(
        "--loops",
        type=int,
        default=None,
        help="must equal trained n_loops (Glint rule: 8). Default: read checkpoint.",
    )
    args = p.parse_args()

    bot = ai.load(args.checkpoint)
    if args.loops is not None and args.loops != bot.n_loops:
        raise SystemExit(
            f"checkpoint n_loops={bot.n_loops}, you passed --loops={args.loops}. "
            "Glint rule: run at the trained loop count or quality collapses."
        )
    if bot.n_loops != LOOPS:
        print(f"warning: checkpoint n_loops={bot.n_loops} (Glint-2 release uses {LOOPS})")

    gpt2 = load_tokenizer(args.tokenizer)
    print(args.prompt, end="", flush=True)
    text = bot.generate(
        args.prompt,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_k=args.top_k,
        repetition_penalty=args.repetition_penalty,
        gpt2_encoder=gpt2,
    )
    if text.startswith(args.prompt):
        print(text[len(args.prompt) :])
    else:
        print(text)


if __name__ == "__main__":
    main()
