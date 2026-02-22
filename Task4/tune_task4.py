#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ml import LogisticBinary, classification_metrics
from core.paths import NEWS_CORPUS_PATH, SEED
from core.sentence_boundary_task import (
    extract_dot_examples,
    select_best_threshold_by_accuracy,
    split_train_dev_test_xy,
    vectorize_dot_features,
)


def evaluate_config(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xdv: np.ndarray,
    ydv: np.ndarray,
    xte: np.ndarray,
    yte: np.ndarray,
    reg_type: str,
    lr: float,
    epochs: int,
    reg_strength: float,
) -> Dict[str, float]:
    model = LogisticBinary(
        lr=lr,
        epochs=epochs,
        reg_type=reg_type,
        reg_strength=reg_strength,
    ).fit(xtr, ytr)

    threshold, dev_acc = select_best_threshold_by_accuracy(model.predict_proba(xdv), ydv)
    pred_test = (model.predict_proba(xte) >= threshold).astype(np.int64)
    test_metrics = classification_metrics(yte, pred_test)

    return {
        "lr": float(lr),
        "epochs": float(epochs),
        "reg_strength": float(reg_strength),
        "threshold": float(threshold),
        "dev_accuracy": float(dev_acc),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["f1"]),
    }


def best_by_dev(results: List[Dict[str, float]]) -> Dict[str, float]:
    return max(results, key=lambda r: (r["dev_accuracy"], r["test_accuracy"]))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune Task 4 Logistic Regression settings using a train/dev/test split."
    )
    parser.add_argument("--news-path", type=Path, default=NEWS_CORPUS_PATH)
    parser.add_argument("--max-docs", type=int, default=30000)
    parser.add_argument("--max-examples", type=int, default=60000)
    parser.add_argument("--max-vocab-tokens", type=int, default=6000)
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save full tuning results as JSON.",
    )
    args = parser.parse_args()

    if not args.news_path.exists():
        raise FileNotFoundError(f"News corpus not found: {args.news_path}")

    feats, y = extract_dot_examples(
        args.news_path,
        max_docs=args.max_docs,
        max_examples=args.max_examples,
    )
    if len(y) < 1000:
        raise RuntimeError(f"Not enough examples for tuning: {len(y)}")

    x = vectorize_dot_features(feats, max_vocab_tokens=args.max_vocab_tokens).astype(
        np.float32, copy=False
    )
    x[:, :2] = np.clip(x[:, :2], 0.0, 30.0)

    xtr, xdv, xte, ytr, ydv, yte = split_train_dev_test_xy(
        x,
        y,
        test_ratio=0.2,
        dev_ratio_within_train=0.1,
        seed=SEED,
    )

    print("Split sizes:")
    print(f"train={len(ytr)} dev={len(ydv)} test={len(yte)}")
    print(f"class_positive_ratio={float(y.mean()):.6f}")

    l2_grid: List[Tuple[float, int, float]] = [
        (0.1, 20, 5e-4),
        (0.1, 40, 1e-4),
        (0.2, 40, 1e-4),
    ]
    l1_grid: List[Tuple[float, int, float]] = [
        (0.1, 20, 1e-4),
        (0.1, 40, 1e-4),
        (0.2, 40, 1e-4),
    ]

    l2_results: List[Dict[str, float]] = []
    for lr, epochs, reg_strength in l2_grid:
        res = evaluate_config(
            xtr, ytr, xdv, ydv, xte, yte, "l2", lr, epochs, reg_strength
        )
        l2_results.append(res)
        print(
            "L2",
            f"lr={lr}",
            f"epochs={epochs}",
            f"reg={reg_strength}",
            f"dev_acc={res['dev_accuracy']:.6f}",
            f"test_acc={res['test_accuracy']:.6f}",
            f"test_f1={res['test_f1']:.6f}",
            f"th={res['threshold']:.2f}",
        )

    l1_results: List[Dict[str, float]] = []
    for lr, epochs, reg_strength in l1_grid:
        res = evaluate_config(
            xtr, ytr, xdv, ydv, xte, yte, "l1", lr, epochs, reg_strength
        )
        l1_results.append(res)
        print(
            "L1",
            f"lr={lr}",
            f"epochs={epochs}",
            f"reg={reg_strength}",
            f"dev_acc={res['dev_accuracy']:.6f}",
            f"test_acc={res['test_accuracy']:.6f}",
            f"test_f1={res['test_f1']:.6f}",
            f"th={res['threshold']:.2f}",
        )

    best_l2 = best_by_dev(l2_results)
    best_l1 = best_by_dev(l1_results)

    print("\nBest by dev accuracy")
    print(f"L2: {best_l2}")
    print(f"L1: {best_l1}")

    if args.save_json is not None:
        payload = {
            "split": {
                "train_examples": float(len(ytr)),
                "dev_examples": float(len(ydv)),
                "test_examples": float(len(yte)),
            },
            "l2_results": l2_results,
            "l1_results": l1_results,
            "best_l2": best_l2,
            "best_l1": best_l1,
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved tuning report: {args.save_json}")


if __name__ == "__main__":
    main()
