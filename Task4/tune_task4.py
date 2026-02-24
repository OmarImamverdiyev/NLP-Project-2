#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.ml import LogisticBinary, classification_metrics
from core.paths import NEWS_CORPUS_PATH, SEED
from core.sentence_boundary_task import (
    extract_dot_examples_from_labeled_csv,
    extract_dot_examples,
    select_best_threshold_by_accuracy,
    split_train_dev_test_xy,
    vectorize_dot_features,
)

MetricValue = float | int


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
) -> Dict[str, MetricValue]:
    model = LogisticBinary(
        lr=lr,
        epochs=epochs,
        reg_type=reg_type,
        reg_strength=reg_strength,
    ).fit(xtr, ytr)

    threshold, dev_acc = select_best_threshold_by_accuracy(model.predict_proba(xdv), ydv)
    pred_dev = (model.predict_proba(xdv) >= threshold).astype(np.int64)
    dev_metrics = classification_metrics(ydv, pred_dev)
    pred_test = (model.predict_proba(xte) >= threshold).astype(np.int64)
    test_metrics = classification_metrics(yte, pred_test)

    return {
        "lr": float(lr),
        "epochs": int(epochs),
        "reg_strength": float(reg_strength),
        "threshold": float(threshold),
        "dev_accuracy": float(dev_acc),
        "dev_f1": float(dev_metrics["f1"]),
        "test_accuracy": float(test_metrics["accuracy"]),
        "test_f1": float(test_metrics["f1"]),
    }


def best_by_metric(
    results: List[Dict[str, MetricValue]],
    primary_metric: str,
) -> Dict[str, MetricValue]:
    if primary_metric == "dev_f1":
        return max(
            results,
            key=lambda r: (r["dev_f1"], r["dev_accuracy"], r["test_f1"], r["test_accuracy"]),
        )
    return max(
        results,
        key=lambda r: (r["dev_accuracy"], r["dev_f1"], r["test_accuracy"], r["test_f1"]),
    )


def fmt_best_result(reg_type: str, result: Dict[str, MetricValue]) -> str:
    return (
        f"reg_type={reg_type} "
        f"lr={result['lr']} epochs={result['epochs']} reg_strength={result['reg_strength']} "
        f"threshold={result['threshold']:.2f} "
        f"dev_acc={result['dev_accuracy']:.6f} dev_f1={result['dev_f1']:.6f} "
        f"test_acc={result['test_accuracy']:.6f} test_f1={result['test_f1']:.6f}"
    )


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(float(token))
    if not values:
        raise ValueError("Expected at least one float value.")
    return values


def parse_int_list(raw: str) -> List[int]:
    values: List[int] = []
    for item in raw.split(","):
        token = item.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("Expected at least one integer value.")
    return values


def build_grid(
    lr_values: List[float],
    epoch_values: List[int],
    reg_values: List[float],
    max_trials: int,
    always_include: Tuple[float, int, float] | None = None,
    seed: int = SEED,
) -> List[Tuple[float, int, float]]:
    full_grid: List[Tuple[float, int, float]] = []
    for lr, epochs, reg_strength in product(lr_values, epoch_values, reg_values):
        full_grid.append((float(lr), int(epochs), float(reg_strength)))
    full_grid = sorted(set(full_grid), key=lambda t: (t[0], t[1], t[2]))

    if max_trials <= 0 or len(full_grid) <= max_trials:
        return full_grid

    rng = random.Random(seed)
    selected = rng.sample(full_grid, max_trials)
    selected = sorted(selected, key=lambda t: (t[0], t[1], t[2]))

    if always_include is not None:
        anchor = (float(always_include[0]), int(always_include[1]), float(always_include[2]))
        if anchor in full_grid and anchor not in selected:
            selected[-1] = anchor
            selected = sorted(set(selected), key=lambda t: (t[0], t[1], t[2]))
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune Task 4 Logistic Regression settings using a train/dev/test split."
    )
    parser.add_argument("--news-path", type=Path, default=NEWS_CORPUS_PATH)
    parser.add_argument(
        "--labeled-csv",
        type=Path,
        default=None,
        help="Optional labeled dot dataset CSV. If provided, news corpus is ignored.",
    )
    parser.add_argument("--max-docs", type=int, default=30000)
    parser.add_argument("--max-examples", type=int, default=60000)
    parser.add_argument("--max-vocab-tokens", type=int, default=6000)
    parser.add_argument(
        "--search-mode",
        choices=["quick", "extended"],
        default="extended",
        help="quick uses the old small grid; extended uses a larger configurable grid.",
    )
    parser.add_argument(
        "--selection-metric",
        choices=["dev_accuracy", "dev_f1"],
        default="dev_accuracy",
        help="Metric used to choose best L1/L2 and overall winner.",
    )
    parser.add_argument(
        "--lr-values",
        type=str,
        default=None,
        help="Comma-separated learning rates. Overrides search-mode defaults.",
    )
    parser.add_argument(
        "--epoch-values",
        type=str,
        default=None,
        help="Comma-separated epoch counts. Overrides search-mode defaults.",
    )
    parser.add_argument(
        "--reg-values",
        type=str,
        default=None,
        help="Comma-separated regularization strengths. Overrides search-mode defaults.",
    )
    parser.add_argument(
        "--max-trials-per-reg",
        type=int,
        default=0,
        help=(
            "Optional cap on sampled configs per reg type from the Cartesian grid. "
            "Set <= 0 to run all combinations."
        ),
    )
    parser.add_argument(
        "--save-json",
        type=Path,
        default=None,
        help="Optional path to save full tuning results as JSON.",
    )
    args = parser.parse_args()

    if args.labeled_csv is not None:
        if not args.labeled_csv.exists():
            raise FileNotFoundError(f"Labeled CSV not found: {args.labeled_csv}")
        feats, y = extract_dot_examples_from_labeled_csv(
            args.labeled_csv,
            max_examples=args.max_examples,
        )
        source_name = str(args.labeled_csv)
    else:
        if not args.news_path.exists():
            raise FileNotFoundError(f"News corpus not found: {args.news_path}")
        feats, y = extract_dot_examples(
            args.news_path,
            max_docs=args.max_docs,
            max_examples=args.max_examples,
        )
        source_name = str(args.news_path)

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
    print(f"data_source={source_name}")
    print(f"train={len(ytr)} dev={len(ydv)} test={len(yte)}")
    print(f"class_positive_ratio={float(y.mean()):.6f}")

    if args.search_mode == "quick":
        default_lr_values = [0.1, 0.2]
        default_epoch_values = [20, 40]
        default_reg_values = [1e-4, 5e-4]
        default_max_trials = 0
    else:
        default_lr_values = [0.3, 0.4, 0.5, 0.6]
        default_epoch_values = [20, 30, 40, 60]
        default_reg_values = [5e-5, 1e-4, 2e-4, 3e-4]
        default_max_trials = 0

    lr_values = parse_float_list(args.lr_values) if args.lr_values is not None else default_lr_values
    epoch_values = (
        parse_int_list(args.epoch_values)
        if args.epoch_values is not None
        else default_epoch_values
    )
    reg_values = (
        parse_float_list(args.reg_values)
        if args.reg_values is not None
        else default_reg_values
    )

    max_trials = int(args.max_trials_per_reg)
    if max_trials == 0:
        max_trials = default_max_trials
    anchor = (0.4, 40, 1e-4)
    l2_grid = build_grid(
        lr_values,
        epoch_values,
        reg_values,
        max_trials=max_trials,
        always_include=anchor,
        seed=SEED,
    )
    l1_grid = build_grid(
        lr_values,
        epoch_values,
        reg_values,
        max_trials=max_trials,
        always_include=anchor,
        seed=SEED + 1,
    )

    print("Search space:")
    print(f"mode={args.search_mode}")
    print(f"lr_values={lr_values}")
    print(f"epoch_values={epoch_values}")
    print(f"reg_values={reg_values}")
    print(f"max_trials_per_reg={max_trials}")
    print(f"l2_configs={len(l2_grid)} l1_configs={len(l1_grid)}")

    l2_results: List[Dict[str, MetricValue]] = []
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

    l1_results: List[Dict[str, MetricValue]] = []
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

    best_l2 = best_by_metric(l2_results, args.selection_metric)
    best_l1 = best_by_metric(l1_results, args.selection_metric)
    best_overall_reg = (
        "l2"
        if (
            (best_l2[args.selection_metric], best_l2["dev_accuracy"], best_l2["dev_f1"])
            >= (best_l1[args.selection_metric], best_l1["dev_accuracy"], best_l1["dev_f1"])
        )
        else "l1"
    )
    best_overall = {
        "reg_type": best_overall_reg,
        **(best_l2 if best_overall_reg == "l2" else best_l1),
    }

    topk = min(5, len(l2_results), len(l1_results))
    top_l2 = sorted(
        l2_results,
        key=lambda r: (r["dev_accuracy"], r["dev_f1"], r["test_accuracy"], r["test_f1"]),
        reverse=True,
    )[:topk]
    top_l1 = sorted(
        l1_results,
        key=lambda r: (r["dev_accuracy"], r["dev_f1"], r["test_accuracy"], r["test_f1"]),
        reverse=True,
    )[:topk]

    print(f"\nBest by {args.selection_metric}")
    print(f"L2: {fmt_best_result('l2', best_l2)}")
    print(f"L1: {fmt_best_result('l1', best_l1)}")
    print(f"Overall: {fmt_best_result(best_overall_reg, best_overall)}")
    print("\nFINAL_BEST_CONFIG")
    print(fmt_best_result(best_overall_reg, best_overall))
    print("\nTop L2 configs:")
    for row in top_l2:
        print(row)
    print("\nTop L1 configs:")
    for row in top_l1:
        print(row)

    if args.save_json is not None:
        payload = {
            "split": {
                "train_examples": int(len(ytr)),
                "dev_examples": int(len(ydv)),
                "test_examples": int(len(yte)),
            },
            "search": {
                "mode": args.search_mode,
                "selection_metric": args.selection_metric,
                "lr_values": [float(v) for v in lr_values],
                "epoch_values": [int(v) for v in epoch_values],
                "reg_values": [float(v) for v in reg_values],
                "max_trials_per_reg": int(max_trials),
                "l2_configs": int(len(l2_grid)),
                "l1_configs": int(len(l1_grid)),
            },
            "l2_results": l2_results,
            "l1_results": l1_results,
            "top_l2": top_l2,
            "top_l1": top_l1,
            "best_l2": best_l2,
            "best_l1": best_l1,
            "best_overall": best_overall,
        }
        args.save_json.parent.mkdir(parents=True, exist_ok=True)
        args.save_json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"\nSaved tuning report: {args.save_json}")


if __name__ == "__main__":
    main()
