#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.sentence_boundary_task_v2 import train_model


def tune_threshold(model, X_dev, y_dev):
    probs = model.predict_proba(X_dev)[:, 1]

    best_threshold = 0.5
    best_f1 = 0

    for t in np.arange(0.1, 0.9, 0.01):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_dev, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_threshold = t

    return best_threshold, best_f1


def tune_C(X_train, y_train, X_dev, y_dev, penalty="l2"):
    best_C = 1
    best_f1 = 0

    for C in [0.01, 0.1, 1, 10, 100]:
        model = train_model(X_train, y_train, penalty=penalty, C=C)
        threshold, _ = tune_threshold(model, X_dev, y_dev)

        probs = model.predict_proba(X_dev)[:, 1]
        preds = (probs >= threshold).astype(int)
        f1 = f1_score(y_dev, preds)

        if f1 > best_f1:
            best_f1 = f1
            best_C = C

    return best_C, best_f1


def main():
    parser = argparse.ArgumentParser(
        description="Tune C and decision threshold for Task 4 v2."
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default=ROOT / "dot_labeled_data.csv",
        help="Path to labeled CSV with a 'label' column.",
    )
    # parser.add_argument(
    #     "--penalty",
    #     type=str,
    #     choices=["l1", "l2"],
    #     default="l2",
    # )
    args = parser.parse_args()

    if not args.dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")

    df = pd.read_csv(args.dataset, keep_default_na=False)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    X_df = df.drop(columns=["label"]).fillna("")
    y = df["label"].values

    vec = DictVectorizer()
    X = vec.fit_transform(X_df.to_dict(orient="records"))

    X_train, X_dev, y_train, y_dev = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )

    print("=== TUNE TASK4 V2 (L1 vs L2) ===")
    print(f"dataset: {args.dataset}")
    print(f"train: {X_train.shape[0]} dev: {X_dev.shape[0]}")

    results = []

    for penalty in ["l2", "l1"]:

        print(f"\nTuning {penalty.upper()} ...")

        best_C, best_dev_f1 = tune_C(
            X_train, y_train, X_dev, y_dev, penalty=penalty
        )

        model = train_model(X_train, y_train, penalty=penalty, C=best_C)
        best_threshold, dev_f1 = tune_threshold(model, X_dev, y_dev)

        print(f"best_C: {best_C}")
        print(f"best_threshold: {best_threshold:.2f}")
        print(f"dev_f1: {dev_f1:.4f}")

        results.append({
            "penalty": penalty,
            "C": best_C,
            "threshold": best_threshold,
            "dev_f1": dev_f1,
        })

    # Compare
    best = max(results, key=lambda r: r["dev_f1"])

    print("\n=== COMPARISON ===")
    for r in results:
        print(
            f"{r['penalty'].upper()} -> "
            f"C={r['C']} "
            f"threshold={r['threshold']:.2f} "
            f"dev_f1={r['dev_f1']:.4f}"
        )

    print(f"\nBEST REGULARIZATION: {best['penalty'].upper()}")


if __name__ == "__main__":
    main()
