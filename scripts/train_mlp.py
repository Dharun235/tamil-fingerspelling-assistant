"""Train an MLP classifier from landmark CSV data."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, f1_score


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--model", type=Path, default=Path("output/mlp.joblib"))
    parser.add_argument("--classes", type=int, default=None, help="Use first N available classes")
    parser.add_argument("--per-class", type=int, default=None, help="Cap samples per class")
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv)
    frame["label"] = frame["label"].astype(str)
    labels = sorted(frame["label"].unique(), key=lambda value: (0, int(value)) if value.isdigit() else (1, value))
    if args.classes:
        labels = labels[: args.classes]
        frame = frame[frame["label"].isin(labels)]
    if args.per_class:
        frame = frame.groupby("label", group_keys=False).sample(n=args.per_class, random_state=42)

    x = frame.iloc[:, 2:].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=42, stratify=y
    )
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=100, early_stopping=True, random_state=42),
    )
    model.fit(x_train, y_train)
    prediction = model.predict(x_test)
    print(f"classes={len(labels)} rows={len(frame)} train={len(x_train)} test={len(x_test)}")
    print(f"accuracy={accuracy_score(y_test, prediction):.4f}")
    print(f"macro_f1={f1_score(y_test, prediction, average='macro'):.4f}")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"saved={args.model}")


if __name__ == "__main__":
    main()
