"""Train and save an MLP classifier from a prepared train split."""

from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=Path("data/splits/train.npz"))
    parser.add_argument("--model", type=Path, default=Path("output/mlp.joblib"))
    parser.add_argument("--max-iter", type=int, default=100)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split = np.load(args.train, allow_pickle=False)
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=args.max_iter, early_stopping=True, random_state=42),
    )
    model.fit(split["x"], split["y"])
    print(f"rows={len(split['y'])} classes={len(model.classes_)} features={split['x'].shape[1]}")
    args.model.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, args.model)
    print(f"saved={args.model}")


if __name__ == "__main__":
    main()
