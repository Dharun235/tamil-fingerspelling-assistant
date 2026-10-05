"""Evaluate a saved classifier on a prepared test split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=Path("output/mlp.joblib"))
    parser.add_argument("--test", type=Path, default=Path("data/splits/test.npz"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/evaluation"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = np.load(args.test, allow_pickle=False)
    model = joblib.load(args.model)
    prediction = model.predict(split["x"])
    labels = [str(label) for label in model.classes_]
    report = classification_report(split["y"], prediction, output_dict=True, zero_division=0)
    metrics = {
        "accuracy": float(accuracy_score(split["y"], prediction)),
        "macro_f1": float(f1_score(split["y"], prediction, average="macro")),
        "rows": int(len(split["y"])),
        "classes": len(labels),
        "classification_report": report,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    matrix = confusion_matrix(split["y"], prediction, labels=labels)
    pd.DataFrame(matrix, index=labels, columns=labels).to_csv(args.output_dir / "confusion_matrix.csv")
    print(f"accuracy={metrics['accuracy']:.4f}")
    print(f"macro_f1={metrics['macro_f1']:.4f}")
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
