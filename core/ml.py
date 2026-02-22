from __future__ import annotations

import math
import random
from typing import Dict, Tuple

import numpy as np

from core.paths import SEED


random.seed(SEED)
np.random.seed(SEED)


def train_test_split_xy(
    x: np.ndarray, y: np.ndarray, test_ratio: float = 0.2
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    np.random.shuffle(idx)
    n_test = int(len(y) * test_ratio)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    return x[train_idx], x[test_idx], y[train_idx], y[test_idx]


class MultinomialNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "MultinomialNB":
        n_features = x.shape[1]
        feature_count = np.zeros((2, n_features), dtype=np.float64)
        class_count = np.zeros(2, dtype=np.float64)

        for c in [0, 1]:
            xc = x[y == c]
            class_count[c] = xc.shape[0]
            feature_count[c] = xc.sum(axis=0)

        smoothed_fc = feature_count + self.alpha
        smoothed_cc = smoothed_fc.sum(axis=1, keepdims=True)
        self.feature_log_prob_ = np.log(smoothed_fc) - np.log(smoothed_cc)
        self.class_log_prior_ = np.log(class_count / class_count.sum())
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        jll = x @ self.feature_log_prob_.T + self.class_log_prior_
        return np.argmax(jll, axis=1).astype(np.int64)


class BernoulliNB:
    def __init__(self, alpha: float = 1.0):
        self.alpha = alpha
        self.class_log_prior_: np.ndarray | None = None
        self.feature_log_prob_: np.ndarray | None = None
        self.feature_log_inv_prob_: np.ndarray | None = None

    def fit(self, x: np.ndarray, y: np.ndarray) -> "BernoulliNB":
        x = (x > 0).astype(np.float64)
        n_features = x.shape[1]
        feature_count = np.zeros((2, n_features), dtype=np.float64)
        class_count = np.zeros(2, dtype=np.float64)

        for c in [0, 1]:
            xc = x[y == c]
            class_count[c] = xc.shape[0]
            feature_count[c] = xc.sum(axis=0)

        p = (feature_count + self.alpha) / (class_count[:, None] + 2 * self.alpha)
        p = np.clip(p, 1e-9, 1 - 1e-9)
        self.feature_log_prob_ = np.log(p)
        self.feature_log_inv_prob_ = np.log(1.0 - p)
        self.class_log_prior_ = np.log(class_count / class_count.sum())
        return self

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = (x > 0).astype(np.float64)
        jll = (
            x @ self.feature_log_prob_.T
            + (1 - x) @ self.feature_log_inv_prob_.T
            + self.class_log_prior_
        )
        return np.argmax(jll, axis=1).astype(np.int64)


class LogisticBinary:
    def __init__(
        self,
        lr: float = 0.1,
        epochs: int = 20,
        reg_type: str = "l2",
        reg_strength: float = 1e-4,
    ):
        self.lr = lr
        self.epochs = epochs
        self.reg_type = reg_type
        self.reg_strength = reg_strength
        self.w: np.ndarray | None = None
        self.b: float = 0.0

    @staticmethod
    def _sigmoid(z: np.ndarray) -> np.ndarray:
        z = np.clip(z, -30, 30)
        return 1.0 / (1.0 + np.exp(-z))

    def fit(self, x: np.ndarray, y: np.ndarray) -> "LogisticBinary":
        n_samples, n_features = x.shape
        self.w = np.zeros(n_features, dtype=np.float64)
        self.b = 0.0
        y = y.astype(np.float64)

        for _ in range(self.epochs):
            z = x @ self.w + self.b
            p = self._sigmoid(z)
            err = p - y
            grad_w = (x.T @ err) / n_samples
            grad_b = float(err.mean())

            if self.reg_type == "l2":
                grad_w += self.reg_strength * self.w
                self.w -= self.lr * grad_w
            elif self.reg_type == "l1":
                self.w -= self.lr * grad_w
                shrink = self.lr * self.reg_strength
                self.w = np.sign(self.w) * np.maximum(np.abs(self.w) - shrink, 0.0)
            else:
                raise ValueError("reg_type must be 'l1' or 'l2'")

            self.b -= self.lr * grad_b
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        return self._sigmoid(x @ self.w + self.b)

    def predict(self, x: np.ndarray) -> np.ndarray:
        return (self.predict_proba(x) >= 0.5).astype(np.int64)


def classification_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    tn = int(((y_true == 0) & (y_pred == 0)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())

    acc = (tp + tn) / max(tp + tn + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


def mcnemar_exact_p(y_true: np.ndarray, pred_a: np.ndarray, pred_b: np.ndarray) -> float:
    a_correct = pred_a == y_true
    b_correct = pred_b == y_true
    n01 = int((a_correct & ~b_correct).sum())
    n10 = int((~a_correct & b_correct).sum())
    n = n01 + n10
    if n == 0:
        return 1.0
    k = min(n01, n10)
    tail = 0.0
    for i in range(0, k + 1):
        tail += math.comb(n, i) * (0.5 ** n)
    return min(1.0, 2.0 * tail)

