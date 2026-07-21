"""Smoke: Glint-2 exact train script constructs + demo path with official tokenizer."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parents[1]


def test_train_glint2_build_matches_release_shape():
    # Import helpers without running main
    sys.path.insert(0, str(ROOT / "examples"))
    import train_glint2 as tg  # noqa: PLC0415

    bot = tg.build_glint2(max_seq_len=128, seed=0)
    assert bot.n_loops == 8
    assert bot.max_loops == 16
    assert bot.coda_layers == 1
    assert bot.attention_window == 256
    assert bot.lora_rank == 4
    assert bot.ffn_dim == 2112
    assert bot.d_model == 96
    assert bot.parameters > 1_060_000


def test_train_glint2_demo_cli(tmp_path):
    tok = tmp_path / "tokenizer.json"
    # Prefer real Glint tokenizer if previously cached; else mini fixture.
    cached = Path("/tmp/glint2_hf/tokenizer.json")
    fixture = ROOT / "tests" / "fixtures" / "glint_tokenizer_mini.json"
    src = cached if cached.is_file() else fixture
    tok.write_text(src.read_text(encoding="utf-8"), encoding="utf-8")
    out = tmp_path / "g2.mmn"
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "examples" / "train_glint2.py"),
            "--demo",
            "--steps",
            "8",
            "--batch-size",
            "2",
            "--max-seq-len",
            "64",
            "--shard-rows",
            "16",
            "--tokenizer",
            str(tok),
            "--out",
            str(out),
            "--save-every",
            "0",
        ],
        check=False,
        capture_output=True,
        text=True,
        cwd=str(ROOT),
    )
    assert proc.returncode == 0, proc.stderr or proc.stdout
    assert out.is_file()
    loaded = ai.load(str(out))
    assert loaded.coda_layers == 1
    assert loaded.n_loops == 8
