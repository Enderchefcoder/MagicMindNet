"""Export a Classifier as Hugging Face binary safetensors and reload it."""

from pathlib import Path

import magicmindnet as ai

OUT = Path(__file__).resolve().parent / "_roundtrip_hf_classifier.safetensors"


def main() -> None:
    clf = ai.Classifier.with_labels(["pos", "neg"], input_dim=32)
    ai.export_classifier(clf, "hf-safetensors", str(OUT))
    raw = OUT.read_bytes()
    assert not raw.startswith(b"{"), "expected binary safetensors, not JSON"
    loaded = ai.import_classifier("safetensors", [str(OUT)])
    assert loaded.labels == clf.labels
    assert loaded.input_dim == clf.input_dim
    print(f"hf classifier safetensors roundtrip ok: {OUT}")


if __name__ == "__main__":
    main()
