from __future__ import annotations

import csv
import random
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.ml import (
    BernoulliNB,
    LogisticBinary,
    MultinomialNB,
    classification_metrics,
    mcnemar_exact_p,
)
from core.paths import SEED
from core.text_utils import tokenize_words

try:
    from scipy import sparse
    from sklearn.feature_extraction.text import CountVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
    from sklearn.naive_bayes import BernoulliNB as SkBernoulliNB
    from sklearn.naive_bayes import MultinomialNB as SkMultinomialNB

    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False


random.seed(SEED)
np.random.seed(SEED)

TASK3_CUSTOM_MAX_SAMPLES_DEFAULT = 5000


AZ_POSITIVE = {
    "yaxsi",
    "gozel",
    "ela",
    "super",
    "tesekkur",
    "saqol",
    "afarin",
    "mukemmel",
    "qeseng",
    "best",
    "sevirem",
    "ugurlar",
    "halal",
    "bravo",
    "cok",
    "cox",
    "mohtesem",
}
AZ_NEGATIVE = {
    "pis",
    "berbad",
    "biyabir",
    "rezil",
    "nefret",
    "zeif",
    "sehv",
    "yalan",
    "kotu",
    "problem",
    "qezeb",
    "utanc",
    "biyabirciliq",
    "bezdim",
    "hec",
    "facie",
}
NEGATION_TOKENS = {
    "deyil",
    "yox",
    "hec",
    "none",
    "not",
    "no",
}

def sentiment_dataset_path_from_root(root: Path) -> Path:
    return root / "sentiment_dataset" / "dataset.csv"


def _parse_binary_sentiment_label(raw_value: str) -> int | None:
    value = raw_value.strip().lower()
    if not value:
        return None
    if value in {"1", "positive", "pos", "true", "yes"}:
        return 1
    if value in {"0", "-1", "negative", "neg", "false", "no"}:
        return 0
    try:
        return 1 if float(value) > 0 else 0
    except ValueError:
        return None


def load_sentiment_dataset(dataset_path: Path) -> Tuple[List[str], List[int], str]:
    texts: List[str] = []
    labels: List[int] = []

    if not dataset_path.exists():
        return texts, labels, f"missing:{dataset_path}"

    text_keys = ("text", "comment_text", "content", "review", "sentence")
    label_keys = ("label", "sentiment", "polarity", "target", "class")

    try:
        with dataset_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
            reader = csv.DictReader(f)
            if not reader.fieldnames:
                return [], [], f"invalid_csv_no_header:{dataset_path}"

            cols = {c.strip().lower(): c for c in reader.fieldnames}
            text_col = next((cols[k] for k in text_keys if k in cols), None)
            label_col = next((cols[k] for k in label_keys if k in cols), None)
            if not text_col or not label_col:
                return [], [], f"invalid_columns:{dataset_path}"

            for row in reader:
                text = (row.get(text_col) or "").strip()
                if not text:
                    continue
                label = _parse_binary_sentiment_label(row.get(label_col) or "")
                if label is None:
                    continue
                texts.append(text)
                labels.append(label)
    except Exception:
        return [], [], f"read_error:{dataset_path}"

    return texts, labels, f"sentiment_dataset:{dataset_path}"


def build_vocab_for_classification(
    texts: Sequence[str],
    min_freq: int = 2,
    max_vocab: int = 30000,
) -> Dict[str, int]:
    freq = Counter()
    for t in texts:
        freq.update(tokenize_words(t))
    items = [(w, c) for w, c in freq.items() if c >= min_freq]
    items.sort(key=lambda x: x[1], reverse=True)
    items = items[:max_vocab]
    return {w: i for i, (w, _c) in enumerate(items)}


def vectorize_bow_counts(texts: Sequence[str], vocab: Dict[str, int]) -> np.ndarray:
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, t in enumerate(texts):
        counts = Counter(tokenize_words(t))
        for w, c in counts.items():
            j = vocab.get(w)
            if j is not None:
                x[i, j] = c
    return x


def vectorize_bow_binary(texts: Sequence[str], vocab: Dict[str, int]) -> np.ndarray:
    x = np.zeros((len(texts), len(vocab)), dtype=np.float32)
    for i, t in enumerate(texts):
        seen = set(tokenize_words(t))
        for w in seen:
            j = vocab.get(w)
            if j is not None:
                x[i, j] = 1.0
    return x


def sentiment_lexicon_features(texts: Sequence[str]) -> np.ndarray:
    feats = np.zeros((len(texts), 6), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = tokenize_words(t)
        n = max(len(toks), 1)
        pos = sum(1 for w in toks if w in AZ_POSITIVE)
        neg = sum(1 for w in toks if w in AZ_NEGATIVE)
        has_negation = any(w in NEGATION_TOKENS for w in toks)
        feats[i, 0] = pos
        feats[i, 1] = neg
        feats[i, 2] = pos - neg
        feats[i, 3] = (pos - neg) / n
        feats[i, 4] = 1.0 if "!" in t else 0.0
        feats[i, 5] = 1.0 if has_negation else 0.0
    return feats


def sentiment_lexicon_nonnegative_features(texts: Sequence[str]) -> np.ndarray:
    signed = sentiment_lexicon_features(texts)
    nonneg = np.zeros_like(signed, dtype=np.float32)
    nonneg[:, 0] = signed[:, 0]
    nonneg[:, 1] = signed[:, 1]
    nonneg[:, 2] = np.maximum(signed[:, 2], 0.0)
    nonneg[:, 3] = np.maximum(-signed[:, 2], 0.0)
    nonneg[:, 4] = signed[:, 4]
    nonneg[:, 5] = signed[:, 5]
    return nonneg


def sentiment_lexicon_binary_features(texts: Sequence[str]) -> np.ndarray:
    dense = sentiment_lexicon_features(texts)
    binary = np.zeros_like(dense, dtype=np.float32)
    binary[:, 0] = (dense[:, 0] > 0).astype(np.float32)
    binary[:, 1] = (dense[:, 1] > 0).astype(np.float32)
    binary[:, 2] = (dense[:, 2] > 0).astype(np.float32)
    binary[:, 3] = (dense[:, 2] < 0).astype(np.float32)
    binary[:, 4] = dense[:, 4]
    binary[:, 5] = dense[:, 5]
    return binary


def _macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    f1_vals: List[float] = []
    for cls in (0, 1):
        tp = int(((y_true == cls) & (y_pred == cls)).sum())
        fp = int(((y_true != cls) & (y_pred == cls)).sum())
        fn = int(((y_true == cls) & (y_pred != cls)).sum())
        prec = tp / max(tp + fp, 1)
        rec = tp / max(tp + fn, 1)
        f1 = 2.0 * prec * rec / max(prec + rec, 1e-12)
        f1_vals.append(f1)
    return float(sum(f1_vals) / len(f1_vals))


def _metrics_with_macro_f1(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    out = classification_metrics(y_true, y_pred)
    out["macro_f1"] = _macro_f1(y_true, y_pred)
    return out


def _cv_splitter(y: np.ndarray) -> "StratifiedKFold | None":
    if not SKLEARN_AVAILABLE:
        return None
    counts = np.bincount(y.astype(np.int64), minlength=2)
    min_class = int(counts.min())
    n_splits = min(5, min_class)
    if n_splits < 2:
        return None
    return StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)


def _cv_mean_f1(model: object, x: object, y: np.ndarray, cv: "StratifiedKFold | None") -> float:
    if cv is None:
        return float("nan")
    scores = cross_val_score(model, x, y, scoring="f1_macro", cv=cv)
    return float(np.mean(scores))


def _select_best(
    scores: Dict[str, float],
    accuracies: Dict[str, float],
) -> str:
    names = list(scores.keys())
    names.sort(key=lambda n: (scores[n], accuracies[n]), reverse=True)
    return names[0]


def _significance_of_best(
    best_name: str,
    accuracies: Dict[str, float],
    pvals: Dict[Tuple[str, str], float],
    alpha: float = 0.05,
) -> float:
    others = [n for n in accuracies if n != best_name]
    for other in others:
        key = tuple(sorted((best_name, other)))
        p = pvals.get(key, 1.0)
        if not (accuracies[best_name] >= accuracies[other] and p < alpha):
            return 0.0
    return 1.0


def _stratified_split_indices(y: np.ndarray, test_ratio: float = 0.2) -> Tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(SEED)
    train_parts: List[np.ndarray] = []
    test_parts: List[np.ndarray] = []
    for cls in np.unique(y):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        n_test = max(1, int(round(len(cls_idx) * test_ratio)))
        test_parts.append(cls_idx[:n_test])
        train_parts.append(cls_idx[n_test:])

    train_idx = np.concatenate(train_parts)
    test_idx = np.concatenate(test_parts)
    rng.shuffle(train_idx)
    rng.shuffle(test_idx)
    return train_idx, test_idx


def _normalize_task3_max_samples(max_samples: int | None) -> int | None:
    if max_samples is None:
        if SKLEARN_AVAILABLE:
            return None
        return TASK3_CUSTOM_MAX_SAMPLES_DEFAULT
    if max_samples <= 0:
        return None
    return int(max_samples)


def _stratified_sample_indices(y: np.ndarray, max_samples: int) -> np.ndarray:
    total = int(len(y))
    if max_samples >= total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(SEED)
    classes, counts = np.unique(y, return_counts=True)
    raw_targets = (counts.astype(np.float64) / float(total)) * float(max_samples)
    targets = np.floor(raw_targets).astype(np.int64)
    targets = np.minimum(targets, counts)

    if max_samples >= len(classes):
        targets = np.maximum(targets, 1)
        targets = np.minimum(targets, counts)

    while int(targets.sum()) < max_samples:
        deficits = counts - targets
        candidates = np.where(deficits > 0)[0]
        if len(candidates) == 0:
            break
        ranked = sorted(
            candidates.tolist(),
            key=lambda i: (raw_targets[i] - targets[i], deficits[i]),
            reverse=True,
        )
        grew = False
        for idx in ranked:
            if targets[idx] < counts[idx]:
                targets[idx] += 1
                grew = True
                if int(targets.sum()) >= max_samples:
                    break
        if not grew:
            break

    while int(targets.sum()) > max_samples:
        if max_samples >= len(classes):
            reducible = np.where(targets > 1)[0]
        else:
            reducible = np.where(targets > 0)[0]
        if len(reducible) == 0:
            break
        idx = int(reducible[np.argmax(targets[reducible])])
        targets[idx] -= 1

    sampled_parts: List[np.ndarray] = []
    for cls, take in zip(classes.tolist(), targets.tolist()):
        if take <= 0:
            continue
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        sampled_parts.append(cls_idx[:take])

    if not sampled_parts:
        return np.arange(total, dtype=np.int64)

    sampled_idx = np.concatenate(sampled_parts).astype(np.int64, copy=False)
    rng.shuffle(sampled_idx)

    if len(sampled_idx) > max_samples:
        sampled_idx = sampled_idx[:max_samples]
    elif len(sampled_idx) < max_samples:
        chosen = np.zeros(total, dtype=bool)
        chosen[sampled_idx] = True
        remaining = np.where(~chosen)[0]
        rng.shuffle(remaining)
        need = max_samples - len(sampled_idx)
        sampled_idx = np.concatenate([sampled_idx, remaining[:need]])

    return sampled_idx


def _best_alpha_custom(
    xtr: np.ndarray,
    ytr: np.ndarray,
    model_kind: str,
) -> float:
    train_idx, dev_idx = _stratified_split_indices(ytr, test_ratio=0.2)
    x_train, x_dev = xtr[train_idx], xtr[dev_idx]
    y_train, y_dev = ytr[train_idx], ytr[dev_idx]

    best_alpha = 1.0
    best_score = -1.0
    for alpha in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
        if model_kind == "mnb":
            model = MultinomialNB(alpha=alpha).fit(x_train, y_train)
        else:
            model = BernoulliNB(alpha=alpha).fit(x_train, y_train)
        pred = model.predict(x_dev)
        score = _macro_f1(y_dev, pred)
        if score > best_score:
            best_score = score
            best_alpha = alpha
    return float(best_alpha)


def _best_lr_custom(xtr: np.ndarray, ytr: np.ndarray) -> Tuple[float, float]:
    train_idx, dev_idx = _stratified_split_indices(ytr, test_ratio=0.2)
    x_train, x_dev = xtr[train_idx], xtr[dev_idx]
    y_train, y_dev = ytr[train_idx], ytr[dev_idx]

    best_lr = 0.2
    best_reg = 1e-4
    best_score = -1.0
    for lr in (0.05, 0.1, 0.2):
        for reg in (1e-5, 1e-4, 1e-3):
            model = LogisticBinary(
                lr=lr,
                epochs=35,
                reg_type="l2",
                reg_strength=reg,
            ).fit(x_train, y_train)
            pred = model.predict(x_dev)
            score = _macro_f1(y_dev, pred)
            if score > best_score:
                best_score = score
                best_lr = lr
                best_reg = reg
    return float(best_lr), float(best_reg)


def _run_task3_sklearn(
    texts: Sequence[str],
    y: np.ndarray,
    data_source: str,
) -> Dict[str, object]:
    x_train_text, x_test_text, y_train, y_test = train_test_split(
        list(texts),
        y,
        test_size=0.2,
        random_state=SEED,
        stratify=y,
    )

    vectorizer = CountVectorizer(
        lowercase=True,
        token_pattern=r"(?u)\b\w+\b",
        ngram_range=(1, 2),
        min_df=2,
        max_features=30000,
    )
    xtr_counts = vectorizer.fit_transform(x_train_text)
    xte_counts = vectorizer.transform(x_test_text)

    xtr_lex_mnb = sentiment_lexicon_nonnegative_features(x_train_text)
    xte_lex_mnb = sentiment_lexicon_nonnegative_features(x_test_text)
    xtr_lex_lr = sentiment_lexicon_features(x_train_text)
    xte_lex_lr = sentiment_lexicon_features(x_test_text)
    xtr_lex_bin = sentiment_lexicon_binary_features(x_train_text)
    xte_lex_bin = sentiment_lexicon_binary_features(x_test_text)

    xtr_mnb = sparse.hstack([xtr_counts, sparse.csr_matrix(xtr_lex_mnb)], format="csr")
    xte_mnb = sparse.hstack([xte_counts, sparse.csr_matrix(xte_lex_mnb)], format="csr")

    xtr_bow_bin = (xtr_counts > 0).astype(np.float32)
    xte_bow_bin = (xte_counts > 0).astype(np.float32)
    xtr_bnb = sparse.hstack([xtr_bow_bin, sparse.csr_matrix(xtr_lex_bin)], format="csr")
    xte_bnb = sparse.hstack([xte_bow_bin, sparse.csr_matrix(xte_lex_bin)], format="csr")

    xtr_lr = sparse.hstack([xtr_counts, sparse.csr_matrix(xtr_lex_lr)], format="csr")
    xte_lr = sparse.hstack([xte_counts, sparse.csr_matrix(xte_lex_lr)], format="csr")

    cv = _cv_splitter(y_train)

    mnb_alpha = 1.0
    bnb_alpha = 1.0
    lr_best_c = 1.0
    lr_best_weight = "none"
    mnb_cv_score = -1.0
    bnb_cv_score = -1.0
    lr_cv_score = -1.0

    if cv is not None:
        mnb_cv_score = float("-inf")
        for alpha in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
            score = _cv_mean_f1(SkMultinomialNB(alpha=alpha), xtr_mnb, y_train, cv)
            if score > mnb_cv_score:
                mnb_cv_score = score
                mnb_alpha = alpha

        bnb_cv_score = float("-inf")
        for alpha in (0.05, 0.1, 0.3, 0.5, 1.0, 2.0):
            score = _cv_mean_f1(
                SkBernoulliNB(alpha=alpha, binarize=0.0),
                xtr_bnb,
                y_train,
                cv,
            )
            if score > bnb_cv_score:
                bnb_cv_score = score
                bnb_alpha = alpha

        lr_cv_score = float("-inf")
        for c in (0.1, 0.3, 0.5, 1.0, 2.0, 5.0, 10.0):
            for class_weight in (None, "balanced"):
                model = LogisticRegression(
                    C=c,
                    penalty="l2",
                    solver="liblinear",
                    max_iter=3000,
                    random_state=SEED,
                    class_weight=class_weight,
                )
                score = _cv_mean_f1(model, xtr_lr, y_train, cv)
                if score > lr_cv_score:
                    lr_cv_score = score
                    lr_best_c = c
                    lr_best_weight = "balanced" if class_weight == "balanced" else "none"

    mnb = SkMultinomialNB(alpha=mnb_alpha).fit(xtr_mnb, y_train)
    pred_mnb = mnb.predict(xte_mnb).astype(np.int64)
    m_mnb = _metrics_with_macro_f1(y_test, pred_mnb)

    bnb = SkBernoulliNB(alpha=bnb_alpha, binarize=0.0).fit(xtr_bnb, y_train)
    pred_bnb = bnb.predict(xte_bnb).astype(np.int64)
    m_bnb = _metrics_with_macro_f1(y_test, pred_bnb)

    lr = LogisticRegression(
        C=lr_best_c,
        penalty="l2",
        solver="liblinear",
        max_iter=3000,
        random_state=SEED,
        class_weight=None if lr_best_weight == "none" else "balanced",
    ).fit(xtr_lr, y_train)
    pred_lr = lr.predict(xte_lr).astype(np.int64)
    m_lr = _metrics_with_macro_f1(y_test, pred_lr)

    p_lr_vs_mnb = mcnemar_exact_p(y_test, pred_lr, pred_mnb)
    p_lr_vs_bnb = mcnemar_exact_p(y_test, pred_lr, pred_bnb)
    p_mnb_vs_bnb = mcnemar_exact_p(y_test, pred_mnb, pred_bnb)

    macro_scores = {
        "multinomial_nb": m_mnb["macro_f1"],
        "bernoulli_nb": m_bnb["macro_f1"],
        "logistic_regression": m_lr["macro_f1"],
    }
    accuracies = {
        "multinomial_nb": m_mnb["accuracy"],
        "bernoulli_nb": m_bnb["accuracy"],
        "logistic_regression": m_lr["accuracy"],
    }
    best_model = _select_best(macro_scores, accuracies)
    pvals = {
        ("logistic_regression", "multinomial_nb"): p_lr_vs_mnb,
        ("bernoulli_nb", "logistic_regression"): p_lr_vs_bnb,
        ("bernoulli_nb", "multinomial_nb"): p_mnb_vs_bnb,
    }
    pvals_norm = {tuple(sorted(k)): v for k, v in pvals.items()}
    best_significant = _significance_of_best(best_model, accuracies, pvals_norm, alpha=0.05)

    return {
        "num_samples": float(len(texts)),
        "positive_ratio": float(y.mean()),
        "num_features_bow": float(len(vectorizer.vocabulary_)),
        "data_source_code": 0.0,
        "data_source": data_source,
        "uses_sklearn_models": 1.0,
        "mnb_best_alpha": float(mnb_alpha),
        "bnb_best_alpha": float(bnb_alpha),
        "lr_best_c": float(lr_best_c),
        "lr_class_weight_balanced": 1.0 if lr_best_weight == "balanced" else 0.0,
        "mnb_cv_macro_f1": float(mnb_cv_score),
        "bnb_cv_macro_f1": float(bnb_cv_score),
        "lr_cv_macro_f1": float(lr_cv_score),
        "mnb_accuracy": m_mnb["accuracy"],
        "mnb_f1": m_mnb["f1"],
        "mnb_macro_f1": m_mnb["macro_f1"],
        "bnb_accuracy": m_bnb["accuracy"],
        "bnb_f1": m_bnb["f1"],
        "bnb_macro_f1": m_bnb["macro_f1"],
        "lr_accuracy": m_lr["accuracy"],
        "lr_f1": m_lr["f1"],
        "lr_macro_f1": m_lr["macro_f1"],
        "p_lr_vs_mnb": p_lr_vs_mnb,
        "p_lr_vs_bnb": p_lr_vs_bnb,
        "p_mnb_vs_bnb": p_mnb_vs_bnb,
        "best_classifier": best_model,
        "best_significant_vs_others_alpha0_05": best_significant,
    }


def _run_task3_custom(
    texts: Sequence[str],
    y: np.ndarray,
    data_source: str,
) -> Dict[str, object]:
    vocab = build_vocab_for_classification(texts, min_freq=2, max_vocab=20000)
    x_counts = vectorize_bow_counts(texts, vocab)
    x_binary = vectorize_bow_binary(texts, vocab)
    x_lex_lr = sentiment_lexicon_features(texts)
    x_lex_mnb = sentiment_lexicon_nonnegative_features(texts)
    x_lex_bin = sentiment_lexicon_binary_features(texts)

    x_counts_lex_mnb = np.hstack([x_counts, x_lex_mnb])
    x_counts_lex_lr = np.hstack([x_counts, x_lex_lr])
    x_binary_lex = np.hstack([x_binary, x_lex_bin])

    train_idx, test_idx = _stratified_split_indices(y, test_ratio=0.2)
    ytr, yte = y[train_idx], y[test_idx]

    xtr_mnb, xte_mnb = x_counts_lex_mnb[train_idx], x_counts_lex_mnb[test_idx]
    xtr_bnb, xte_bnb = x_binary_lex[train_idx], x_binary_lex[test_idx]
    xtr_lr, xte_lr = x_counts_lex_lr[train_idx], x_counts_lex_lr[test_idx]

    mnb_alpha = _best_alpha_custom(xtr_mnb, ytr, model_kind="mnb")
    bnb_alpha = _best_alpha_custom(xtr_bnb, ytr, model_kind="bnb")
    lr_best_lr, lr_best_reg = _best_lr_custom(xtr_lr, ytr)

    mnb = MultinomialNB(alpha=mnb_alpha).fit(xtr_mnb, ytr)
    pred_mnb = mnb.predict(xte_mnb)
    m_mnb = _metrics_with_macro_f1(yte, pred_mnb)

    bnb = BernoulliNB(alpha=bnb_alpha).fit(xtr_bnb, ytr)
    pred_bnb = bnb.predict(xte_bnb)
    m_bnb = _metrics_with_macro_f1(yte, pred_bnb)

    lr = LogisticBinary(
        lr=lr_best_lr,
        epochs=35,
        reg_type="l2",
        reg_strength=lr_best_reg,
    ).fit(xtr_lr, ytr)
    pred_lr = lr.predict(xte_lr)
    m_lr = _metrics_with_macro_f1(yte, pred_lr)

    p_lr_vs_mnb = mcnemar_exact_p(yte, pred_lr, pred_mnb)
    p_lr_vs_bnb = mcnemar_exact_p(yte, pred_lr, pred_bnb)
    p_mnb_vs_bnb = mcnemar_exact_p(yte, pred_mnb, pred_bnb)

    macro_scores = {
        "multinomial_nb": m_mnb["macro_f1"],
        "bernoulli_nb": m_bnb["macro_f1"],
        "logistic_regression": m_lr["macro_f1"],
    }
    accuracies = {
        "multinomial_nb": m_mnb["accuracy"],
        "bernoulli_nb": m_bnb["accuracy"],
        "logistic_regression": m_lr["accuracy"],
    }
    best_model = _select_best(macro_scores, accuracies)
    pvals = {
        ("logistic_regression", "multinomial_nb"): p_lr_vs_mnb,
        ("bernoulli_nb", "logistic_regression"): p_lr_vs_bnb,
        ("bernoulli_nb", "multinomial_nb"): p_mnb_vs_bnb,
    }
    pvals_norm = {tuple(sorted(k)): v for k, v in pvals.items()}
    best_significant = _significance_of_best(best_model, accuracies, pvals_norm, alpha=0.05)

    return {
        "num_samples": float(len(texts)),
        "positive_ratio": float(y.mean()),
        "num_features_bow": float(len(vocab)),
        "data_source_code": 0.0,
        "data_source": data_source,
        "uses_sklearn_models": 0.0,
        "mnb_best_alpha": float(mnb_alpha),
        "bnb_best_alpha": float(bnb_alpha),
        "lr_best_lr": float(lr_best_lr),
        "lr_best_reg_strength": float(lr_best_reg),
        "mnb_accuracy": m_mnb["accuracy"],
        "mnb_f1": m_mnb["f1"],
        "mnb_macro_f1": m_mnb["macro_f1"],
        "bnb_accuracy": m_bnb["accuracy"],
        "bnb_f1": m_bnb["f1"],
        "bnb_macro_f1": m_bnb["macro_f1"],
        "lr_accuracy": m_lr["accuracy"],
        "lr_f1": m_lr["f1"],
        "lr_macro_f1": m_lr["macro_f1"],
        "p_lr_vs_mnb": p_lr_vs_mnb,
        "p_lr_vs_bnb": p_lr_vs_bnb,
        "p_mnb_vs_bnb": p_mnb_vs_bnb,
        "best_classifier": best_model,
        "best_significant_vs_others_alpha0_05": best_significant,
    }


def run_task3(root: Path, max_samples: int | None = None) -> Dict[str, object]:
    dataset_path = sentiment_dataset_path_from_root(root)
    texts, labels, data_source = load_sentiment_dataset(dataset_path)
    original_num_samples = len(texts)
    effective_max_samples = _normalize_task3_max_samples(max_samples)
    sampled_for_memory = 0.0

    if effective_max_samples is not None and len(texts) > effective_max_samples:
        y_all = np.array(labels, dtype=np.int64)
        sample_idx = _stratified_sample_indices(y_all, effective_max_samples)
        texts = [texts[int(i)] for i in sample_idx.tolist()]
        labels = [int(y_all[int(i)]) for i in sample_idx.tolist()]
        sampled_for_memory = 1.0

    if len(texts) < 500:
        return {
            "error": 1.0,
            "num_samples": float(len(texts)),
            "num_samples_original": float(original_num_samples),
            "sampled_for_memory": sampled_for_memory,
            "task3_max_samples": (
                float(effective_max_samples) if effective_max_samples is not None else -1.0
            ),
            "data_source": data_source,
            "dataset_path": str(dataset_path),
        }

    y = np.array(labels, dtype=np.int64)
    if len(np.unique(y)) < 2:
        return {
            "error": 1.0,
            "num_samples": float(len(texts)),
            "num_samples_original": float(original_num_samples),
            "sampled_for_memory": sampled_for_memory,
            "task3_max_samples": (
                float(effective_max_samples) if effective_max_samples is not None else -1.0
            ),
            "data_source": data_source,
            "dataset_path": str(dataset_path),
        }

    if SKLEARN_AVAILABLE:
        metrics = _run_task3_sklearn(texts, y, data_source)
    else:
        metrics = _run_task3_custom(texts, y, data_source)
    metrics["uses_only_sentiment_dataset"] = 1.0
    metrics["num_samples_original"] = float(original_num_samples)
    metrics["sampled_for_memory"] = sampled_for_memory
    metrics["task3_max_samples"] = (
        float(effective_max_samples) if effective_max_samples is not None else -1.0
    )
    metrics["dataset_path"] = str(dataset_path)
    return metrics
