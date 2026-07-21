"""Docs site assets for the 0.2.0 product landing page."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SITE = ROOT / "docs" / "site"


def test_docs_site_index_exists():
    index = SITE / "index.html"
    assert index.is_file()
    text = index.read_text(encoding="utf-8")
    assert "MagicMindNet" in text
    assert "pip install magicmindnet" in text
    assert "chart-size" in text


def test_docs_site_chart_data_present():
    interop = SITE / "data" / "interop_benchmark.json"
    smoke = SITE / "data" / "smoke_eval.json"
    assert interop.is_file()
    assert smoke.is_file()
    assert "gguf-q4_k" in interop.read_text(encoding="utf-8")
    assert '"suite": "smoke"' in smoke.read_text(encoding="utf-8")


def test_docs_site_css_js_assets():
    assert (SITE / "css" / "mmn.css").is_file()
    assert (SITE / "js" / "charts.js").is_file()
    assert (SITE / "assets" / "hero-wave.svg").is_file()
    assert (SITE / "assets" / "demo-storyboard.svg").is_file()
