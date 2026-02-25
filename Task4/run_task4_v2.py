#!/usr/bin/env python3
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.sentence_boundary_task_v2 import evaluate_with_threshold, train_model
from Task4.tune_task4_v2 import tune_C, tune_threshold


def main():
    parser = argparse.ArgumentParser(
        description="Run Task 4 v2 training/evaluation with tuned C and threshold."
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

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, stratify=y, random_state=42
    )
    X_dev, X_test, y_dev, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, stratify=y_temp, random_state=42
    )

    # print("=== TUNING C ===")
    # best_C, best_dev_f1 = tune_C(
    #     X_train,
    #     y_train,
    #     X_dev,
    #     y_dev,
    #     penalty=args.penalty,
    # )
    # print("Best C:", best_C)
    # print("Best Dev F1:", f"{best_dev_f1:.4f}")

    # print("\n=== TRAIN FINAL MODEL ===")
    # model = train_model(X_train, y_train, penalty=args.penalty, C=best_C)
    # best_threshold, dev_f1 = tune_threshold(model, X_dev, y_dev)
    # print("Best Threshold:", f"{best_threshold:.2f}")
    # print("Dev F1:", f"{dev_f1:.4f}")

    # print("\n=== TEST RESULTS ===")
    # metrics = evaluate_with_threshold(model, X_test, y_test, best_threshold)
    # for k, v in metrics.items():
    #     print(f"{k}: {v:.4f}")

    print("=== TASK 4 V2 (L1 vs L2 COMPARISON) ===")
    print(f"train={len(y_train)} dev={len(y_dev)} test={len(y_test)}")

    results = []

    for penalty in ["l2", "l1"]:

        print(f"\n--- TUNING {penalty.upper()} ---")

        # Tune C
        best_C, best_dev_f1 = tune_C(
            X_train,
            y_train,
            X_dev,
            y_dev,
            penalty=penalty,
        )

        print("Best C:", best_C)
        print("Best Dev F1:", f"{best_dev_f1:.4f}")

        # Train final model
        model = train_model(X_train, y_train, penalty=penalty, C=best_C)

        # Tune threshold
        best_threshold, dev_f1 = tune_threshold(model, X_dev, y_dev)

        print("Best Threshold:", f"{best_threshold:.2f}")
        print("Dev F1:", f"{dev_f1:.4f}")

        # Evaluate on test
        metrics = evaluate_with_threshold(model, X_test, y_test, best_threshold)

        print("\nTest Results:")
        for k, v in metrics.items():
            print(f"{k}: {v:.4f}")

        results.append({
            "penalty": penalty,
            "C": best_C,
            "threshold": best_threshold,
            "accuracy": metrics["accuracy"],
            "f1": metrics["f1"],
        })

    # -------------------------
    # Final Comparison
    # -------------------------

    best = max(results, key=lambda r: r["f1"])

    print("\n=== FINAL COMPARISON ===")
    for r in results:
        print(
            f"{r['penalty'].upper()} -> "
            f"Accuracy={r['accuracy']:.4f} "
            f"F1={r['f1']:.4f}"
        )

    print(f"\nBEST MODEL: {best['penalty'].upper()}")

if __name__ == "__main__":
    main()
