"""Train and evaluate an MLP classifier from the landmark CSV."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--model", type=Path, default=Path("output/mlp.joblib"))
    parser.add_argument("--test-size", type=float, default=0.2)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv)
    ignored = {"image_path", "label"}
    x = frame.drop(columns=sorted(ignored)).to_numpy(dtype=np.float32)
    y = frame["label"].astype(str).to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=args.test_size, random_state=42, stratify=y
    )

    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(256, 128), max_iter=100, early_stopping=True, random_state=42, verbose=True),
    )
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    print(classification_report(y_test, predictions, zero_division=0))
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"Saved {args.model}")


if __name__ == "__main__":
    main()
