"""Export a tiny Classifier checkpoint and reload it (labels + init_seed)."""

from pathlib import Path

import magicmindnet as ai

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "tests" / "fixtures" / "labels_small.json"
OUT = Path(__file__).resolve().parent / "_roundtrip_classifier.mmn"


def main() -> None:
    ds = ai.DatasetClassification(str(DATA), "text", "tag")
    clf = ai.Classifier.from_classification(ds, input_dim=16, seed=3)
    labels_before = clf.labels
    ai.export_classifier(clf, "safetensors", str(OUT))
    loaded = ai.import_classifier("safetensors", [str(OUT)])
    assert loaded.labels == labels_before
    assert loaded.init_seed == 3
    print(f"classifier roundtrip ok: {OUT} labels={labels_before} seed={loaded.init_seed}")


if __name__ == "__main__":
    main()
