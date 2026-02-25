from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from core.paths import ROOT

SEED = 42


def train_model(X_train, y_train, penalty: str = "l2", C: float = 10.0):
    model = LogisticRegression(
        penalty=penalty,
        C=C,
        max_iter=1000,
        solver="liblinear",
        random_state=SEED,
    )
    model.fit(X_train, y_train)
    return model


def evaluate_with_threshold(model, X, y, threshold: float) -> Dict[str, float]:
    probs = model.predict_proba(X)[:, 1]
    preds = (probs >= threshold).astype(int)
    return {
        "accuracy": float(accuracy_score(y, preds)),
        "precision": float(precision_score(y, preds)),
        "recall": float(recall_score(y, preds)),
        "f1": float(f1_score(y, preds)),
    }


def _normalize_binary_labels(raw_labels: pd.Series) -> Tuple[np.ndarray, np.ndarray]:
    labels: list[int] = []
    valid_mask: list[bool] = []

    for raw in raw_labels.tolist():
        label = None
        if isinstance(raw, (int, np.integer)):
            if int(raw) in (0, 1):
                label = int(raw)
        elif isinstance(raw, (float, np.floating)):
            if np.isfinite(raw) and int(raw) in (0, 1):
                label = int(raw)
        else:
            token = str(raw).strip().lower()
            if token in {"1", "true", "yes", "y"}:
                label = 1
            elif token in {"0", "false", "no", "n"}:
                label = 0

        if label is None:
            labels.append(0)
            valid_mask.append(False)
        else:
            labels.append(label)
            valid_mask.append(True)

    return np.asarray(labels, dtype=np.int64), np.asarray(valid_mask, dtype=bool)


def _load_labeled_examples(
    dataset_path: Path,
    max_examples: int | None,
) -> Tuple[pd.DataFrame, np.ndarray]:
    df = pd.read_csv(dataset_path, keep_default_na=False)
    if "label" not in df.columns:
        raise ValueError("Dataset must contain a 'label' column.")

    if max_examples is not None and max_examples > 0:
        df = df.iloc[:max_examples].copy()

    y_all, valid_mask = _normalize_binary_labels(df["label"])
    if not bool(valid_mask.any()):
        raise ValueError("Dataset has no valid binary labels.")

    if not bool(valid_mask.all()):
        df = df.loc[valid_mask].copy()
        y_all = y_all[valid_mask]

    x_df = df.drop(columns=["label"]).fillna("")
    if len(x_df) < 100:
        raise ValueError("Not enough labeled examples for Task 4 v2.")
    if np.unique(y_all).shape[0] < 2:
        raise ValueError("Task 4 v2 dataset must contain both classes (0 and 1).")
    return x_df, y_all


def fit_task4_v2_assets(
    dataset_path: Path | str | None = None,
    max_examples: int | None = 60000,
) -> Dict[str, Any]:
    resolved_dataset = (
        Path(dataset_path)
        if dataset_path is not None
        else Path(ROOT) / "dot_labeled_data.csv"
    )
    if not resolved_dataset.exists():
        raise FileNotFoundError(f"Dataset not found: {resolved_dataset}")

    x_df, y = _load_labeled_examples(resolved_dataset, max_examples=max_examples)
    vectorizer = DictVectorizer()
    x = vectorizer.fit_transform(x_df.to_dict(orient="records"))

    try:
        x_train, x_temp, y_train, y_temp = train_test_split(
            x,
            y,
            test_size=0.3,
            stratify=y,
            random_state=SEED,
        )
        x_dev, x_test, y_dev, y_test = train_test_split(
            x_temp,
            y_temp,
            test_size=0.5,
            stratify=y_temp,
            random_state=SEED,
        )
    except ValueError as exc:
        raise ValueError(
            "Failed to split Task 4 v2 data into train/dev/test. "
            "Try increasing max_examples or fixing class balance."
        ) from exc

    # Local import avoids circular import during module initialization.
    from Task4.tune_task4_v2 import tune_C, tune_threshold

    run_rows: list[Dict[str, Any]] = []
    for penalty in ("l2", "l1"):
        best_c, best_dev_f1 = tune_C(
            x_train,
            y_train,
            x_dev,
            y_dev,
            penalty=penalty,
        )
        model = train_model(x_train, y_train, penalty=penalty, C=best_c)
        best_threshold, tuned_dev_f1 = tune_threshold(model, x_dev, y_dev)
        test_metrics = evaluate_with_threshold(model, x_test, y_test, best_threshold)

        run_rows.append(
            {
                "penalty": penalty,
                "C": float(best_c),
                "dev_f1": float(tuned_dev_f1),
                "threshold": float(best_threshold),
                "accuracy": float(test_metrics["accuracy"]),
                "precision": float(test_metrics["precision"]),
                "recall": float(test_metrics["recall"]),
                "f1": float(test_metrics["f1"]),
                "model": model,
            }
        )

    best_row = max(run_rows, key=lambda row: row["f1"])

    metrics: Dict[str, float | str] = {
        "data_source": str(resolved_dataset.resolve()),
        "num_examples": float(x.shape[0]),
        "num_features": float(x.shape[1]),
        "train_examples": float(x_train.shape[0]),
        "dev_examples": float(x_dev.shape[0]),
        "test_examples": float(x_test.shape[0]),
        "best_penalty": str(best_row["penalty"]),
        "best_threshold": float(best_row["threshold"]),
        "best_test_accuracy": float(best_row["accuracy"]),
        "best_test_f1": float(best_row["f1"]),
    }
    for row in run_rows:
        prefix = str(row["penalty"])
        metrics[f"{prefix}_best_c"] = float(row["C"])
        metrics[f"{prefix}_dev_f1"] = float(row["dev_f1"])
        metrics[f"{prefix}_threshold"] = float(row["threshold"])
        metrics[f"{prefix}_accuracy"] = float(row["accuracy"])
        metrics[f"{prefix}_precision"] = float(row["precision"])
        metrics[f"{prefix}_recall"] = float(row["recall"])
        metrics[f"{prefix}_f1"] = float(row["f1"])

    return {
        "metrics": metrics,
        "vectorizer": vectorizer,
        "best_model": best_row["model"],
        "best_threshold": float(best_row["threshold"]),
        "best_penalty": str(best_row["penalty"]),
        "models": {str(row["penalty"]): row["model"] for row in run_rows},
        "thresholds": {str(row["penalty"]): float(row["threshold"]) for row in run_rows},
        "rows": run_rows,
    }


def run_task4(
    dataset_path: Path | str | None = None,
    max_examples: int | None = 60000,
) -> Dict[str, float | str]:
    return dict(
        fit_task4_v2_assets(
            dataset_path=dataset_path,
            max_examples=max_examples,
        )["metrics"]
    )
