"""Create stratified train/test arrays from landmarks.csv."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output-dir", type=Path, default=Path("data/splits"))
    parser.add_argument("--test-size", type=float, default=0.2)
    parser.add_argument("--max-per-class", type=int, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.csv)
    frame["label"] = frame["label"].astype(str)
    if args.max_per_class:
        frame = frame.groupby("label", group_keys=False).sample(n=args.max_per_class, random_state=42)

    features = frame.iloc[:, 2:].to_numpy(dtype=np.float32)
    labels = frame["label"].to_numpy(dtype="U")
    paths = frame["image_path"].to_numpy(dtype="U")
    x_train, x_test, y_train, y_test, paths_train, paths_test = train_test_split(
        features, labels, paths, test_size=args.test_size, random_state=42, stratify=labels
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output_dir / "train.npz", x=x_train, y=y_train)
    np.savez_compressed(args.output_dir / "test.npz", x=x_test, y=y_test)
    np.save(args.output_dir / "train_paths.npy", paths_train)
    np.save(args.output_dir / "test_paths.npy", paths_test)
    metadata = {
        "source_csv": str(args.csv),
        "rows": len(frame),
        "classes": sorted(set(labels), key=lambda value: (0, int(value)) if value.isdigit() else (1, value)),
        "feature_count": int(features.shape[1]),
        "test_size": args.test_size,
        "random_state": 42,
    }
    (args.output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print(f"rows={len(frame)} classes={len(metadata['classes'])}")
    print(f"train={len(x_train)} test={len(x_test)} features={features.shape[1]}")
    print(f"saved={args.output_dir}")


if __name__ == "__main__":
    main()
