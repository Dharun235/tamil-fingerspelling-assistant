"""Prepare landmarks, train MLP, evaluate it, and save all results."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    top_k_accuracy_score,
)
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", type=Path, default=Path("data/landmarks.csv"))
    parser.add_argument("--results-dir", type=Path, default=Path("results"))
    parser.add_argument("--test-size", type=float, default=0.30)
    parser.add_argument("--max-iter", type=int, default=100)
    return parser.parse_args()


def numeric_label_order(values: set[str]) -> list[str]:
    return sorted(values, key=lambda value: (0, int(value)) if value.isdigit() else (1, value))


def main() -> None:
    args = parse_args()
    args.results_dir.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
        handlers=[logging.FileHandler(args.results_dir / "run.log"), logging.StreamHandler()],
    )
    logger = logging.getLogger("tamil-fingerspelling")
    logger.info("loading landmarks from %s", args.csv)
    frame = pd.read_csv(args.csv)
    frame["label"] = frame["label"].astype(str)
    feature_names = list(frame.columns[2:])
    x = frame.iloc[:, 2:].to_numpy(dtype=np.float32)
    y = frame["label"].to_numpy(dtype="U")
    labels = numeric_label_order(set(y))

    paths = frame["image_path"].to_numpy(dtype="U")
    x_train, x_test, y_train, y_test, paths_train, paths_test = train_test_split(
        x, y, paths, test_size=args.test_size, random_state=42, stratify=y
    )
    logger.info("split rows=%d train=%d test=%d classes=%d features=%d", len(frame), len(x_train), len(x_test), len(labels), len(feature_names))
    model = make_pipeline(
        StandardScaler(),
        MLPClassifier(
            hidden_layer_sizes=(128, 64),
            max_iter=args.max_iter,
            batch_size=256,
            early_stopping=False,
            random_state=42,
        ),
    )
    logger.info("training MLP")
    model.fit(x_train, y_train)
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)

    joblib.dump(model, args.results_dir / "mlp.joblib")
    pd.DataFrame(confusion_matrix(y_test, predictions, labels=labels), index=labels, columns=labels).to_csv(
        args.results_dir / "confusion_matrix.csv"
    )
    pd.DataFrame(classification_report(y_test, predictions, labels=labels, output_dict=True, zero_division=0)).T.to_csv(
        args.results_dir / "classification_report.csv"
    )
    pd.DataFrame(x_train, columns=feature_names).corr().to_csv(args.results_dir / "feature_correlation.csv")

    top_k = min(3, probabilities.shape[1])
    top_indices = np.argsort(probabilities, axis=1)[:, -top_k:][:, ::-1]
    prediction_rows = []
    for row_index, (path, actual, predicted, row_probabilities) in enumerate(
        zip(paths_test, y_test, predictions, probabilities, strict=True)
    ):
        prediction_rows.append(
            {
                "image_path": path,
                "actual_class": actual,
                "predicted_class": predicted,
                "correct": bool(actual == predicted),
                "confidence": float(row_probabilities.max()),
                "top1_class": str(model.classes_[top_indices[row_index, 0]]),
                "top1_probability": float(row_probabilities[top_indices[row_index, 0]]),
                "top2_class": str(model.classes_[top_indices[row_index, 1]]) if top_k > 1 else "",
                "top2_probability": float(row_probabilities[top_indices[row_index, 1]]) if top_k > 1 else 0.0,
                "top3_class": str(model.classes_[top_indices[row_index, 2]]) if top_k > 2 else "",
                "top3_probability": float(row_probabilities[top_indices[row_index, 2]]) if top_k > 2 else 0.0,
            }
        )
    pd.DataFrame(prediction_rows).to_csv(args.results_dir / "test_predictions.csv", index=False)

    metrics = {
        "rows": int(len(frame)),
        "train_rows": int(len(x_train)),
        "test_rows": int(len(x_test)),
        "classes": len(labels),
        "features": len(feature_names),
        "test_size": args.test_size,
        "random_state": 42,
        "accuracy": float(accuracy_score(y_test, predictions)),
        "balanced_accuracy": float(balanced_accuracy_score(y_test, predictions)),
        "precision_macro": float(precision_score(y_test, predictions, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_test, predictions, average="macro", zero_division=0)),
        "f1_weighted": float(f1_score(y_test, predictions, average="weighted", zero_division=0)),
        "top_3_accuracy": float(top_k_accuracy_score(y_test, probabilities, k=3, labels=model.classes_)),
    }
    (args.results_dir / "metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    np.save(args.results_dir / "test_labels.npy", y_test)
    np.save(args.results_dir / "test_predictions.npy", predictions)
    logger.info("accuracy=%.4f macro_f1=%.4f top3=%.4f", metrics["accuracy"], metrics["f1_macro"], metrics["top_3_accuracy"])
    logger.info("saved results to %s", args.results_dir)


if __name__ == "__main__":
    main()
