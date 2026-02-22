from __future__ import annotations

import csv
import random
import re
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
from core.paths import SEED, YT_COMMENTS_PATH
from core.text_utils import tokenize_words


random.seed(SEED)
np.random.seed(SEED)


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


def parse_like_count(value: str) -> int:
    if value is None:
        return 0
    value = value.strip().replace(",", "")
    if not value:
        return 0
    digits = re.findall(r"\d+", value)
    if not digits:
        return 0
    try:
        return int(digits[0])
    except ValueError:
        return 0


def build_weak_youtube_sentiment(path: Path, min_samples: int = 2000) -> Tuple[List[str], List[int]]:
    texts: List[str] = []
    labels: List[int] = []

    if not path.exists():
        return texts, labels

    with path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            text = (row.get("comment_text") or "").strip()
            if not text:
                continue

            toks = tokenize_words(text)
            if len(toks) < 3:
                continue

            pos = sum(1 for t in toks if t in AZ_POSITIVE)
            neg = sum(1 for t in toks if t in AZ_NEGATIVE)
            likes = parse_like_count(row.get("comment_likes", "0"))

            if pos > neg:
                y = 1
            elif neg > pos:
                y = 0
            else:
                if likes >= 10:
                    y = 1
                elif likes == 0:
                    y = 0
                else:
                    continue

            texts.append(text)
            labels.append(y)

    pos_idx = [i for i, y in enumerate(labels) if y == 1]
    neg_idx = [i for i, y in enumerate(labels) if y == 0]
    n = min(len(pos_idx), len(neg_idx))
    n = min(n, max(min_samples // 2, 1000))
    if n == 0:
        return [], []

    chosen = pos_idx[:n] + neg_idx[:n]
    random.shuffle(chosen)
    return [texts[i] for i in chosen], [labels[i] for i in chosen]


def find_labeled_sentiment_dataset(root: Path) -> Tuple[List[str], List[int], str]:
    candidates = list(root.rglob("*.csv"))
    label_keys = {"label", "sentiment", "polarity", "target", "class"}
    text_keys = {"text", "comment_text", "content", "review", "sentence"}

    for csv_path in candidates:
        if ".git" in str(csv_path):
            continue
        try:
            with csv_path.open("r", encoding="utf-8", errors="ignore", newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    continue
                cols = {c.strip().lower(): c for c in reader.fieldnames}
                lk = next((cols[k] for k in label_keys if k in cols), None)
                tk = next((cols[k] for k in text_keys if k in cols), None)
                if not lk or not tk:
                    continue

                texts: List[str] = []
                labels: List[int] = []
                for row in reader:
                    t = (row.get(tk) or "").strip()
                    y = (row.get(lk) or "").strip().lower()
                    if not t or not y:
                        continue

                    if y in {"1", "positive", "pos", "true", "yes"}:
                        labels.append(1)
                    elif y in {"0", "-1", "negative", "neg", "false", "no"}:
                        labels.append(0)
                    else:
                        try:
                            labels.append(1 if float(y) > 0 else 0)
                        except ValueError:
                            continue
                    texts.append(t)

                if len(texts) >= 500 and len(set(labels)) == 2:
                    return texts, labels, f"labeled_csv:{csv_path}"
        except Exception:
            continue

    texts, labels = build_weak_youtube_sentiment(YT_COMMENTS_PATH)
    return texts, labels, "weak_youtube_distant_supervision"


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
    feats = np.zeros((len(texts), 4), dtype=np.float32)
    for i, t in enumerate(texts):
        toks = tokenize_words(t)
        n = max(len(toks), 1)
        pos = sum(1 for w in toks if w in AZ_POSITIVE)
        neg = sum(1 for w in toks if w in AZ_NEGATIVE)
        feats[i, 0] = pos
        feats[i, 1] = neg
        feats[i, 2] = pos - neg
        feats[i, 3] = (pos - neg) / n
    return feats


def run_task3(root: Path) -> Dict[str, float]:
    texts, labels, data_source = find_labeled_sentiment_dataset(root)
    if len(texts) < 500:
        return {"error": 1.0, "num_samples": float(len(texts))}

    y = np.array(labels, dtype=np.int64)
    vocab = build_vocab_for_classification(texts, min_freq=2, max_vocab=20000)

    x_counts = vectorize_bow_counts(texts, vocab)
    x_binary = vectorize_bow_binary(texts, vocab)
    x_lex = sentiment_lexicon_features(texts)
    x_counts_lex = np.hstack([x_counts, x_lex])
    x_binary_lex = np.hstack([x_binary, x_lex])

    idx = np.arange(len(y))
    np.random.shuffle(idx)
    n_test = int(len(y) * 0.2)
    test_idx = idx[:n_test]
    train_idx = idx[n_test:]
    ytr, yte = y[train_idx], y[test_idx]

    xtr_mnb, xte_mnb = x_counts_lex[train_idx], x_counts_lex[test_idx]
    xtr_bnb, xte_bnb = x_binary_lex[train_idx], x_binary_lex[test_idx]
    xtr_lr, xte_lr = x_counts_lex[train_idx], x_counts_lex[test_idx]

    mnb = MultinomialNB(alpha=1.0).fit(xtr_mnb, ytr)
    pred_mnb = mnb.predict(xte_mnb)
    m_mnb = classification_metrics(yte, pred_mnb)

    bnb = BernoulliNB(alpha=1.0).fit(xtr_bnb, ytr)
    pred_bnb = bnb.predict(xte_bnb)
    m_bnb = classification_metrics(yte, pred_bnb)

    lr = LogisticBinary(lr=0.2, epochs=25, reg_type="l2", reg_strength=1e-4).fit(xtr_lr, ytr)
    pred_lr = lr.predict(xte_lr)
    m_lr = classification_metrics(yte, pred_lr)

    p_lr_vs_mnb = mcnemar_exact_p(yte, pred_lr, pred_mnb)
    p_lr_vs_bnb = mcnemar_exact_p(yte, pred_lr, pred_bnb)
    p_mnb_vs_bnb = mcnemar_exact_p(yte, pred_mnb, pred_bnb)

    return {
        "num_samples": float(len(texts)),
        "num_features_bow": float(len(vocab)),
        "data_source_code": 0.0 if data_source.startswith("labeled_csv") else 1.0,
        "mnb_accuracy": m_mnb["accuracy"],
        "mnb_f1": m_mnb["f1"],
        "bnb_accuracy": m_bnb["accuracy"],
        "bnb_f1": m_bnb["f1"],
        "lr_accuracy": m_lr["accuracy"],
        "lr_f1": m_lr["f1"],
        "p_lr_vs_mnb": p_lr_vs_mnb,
        "p_lr_vs_bnb": p_lr_vs_bnb,
        "p_mnb_vs_bnb": p_mnb_vs_bnb,
    }

