from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.ml import LogisticBinary, classification_metrics, mcnemar_exact_p
from core.paths import SEED


ABBREV_SET = {
    # Azerbaijani honorifics/titles
    "dr",
    "prof",
    "dos",
    "akad",
    "c",
    "x",
    # Azerbaijani frequent abbreviations
    "məs",
    "təxm",
    "səh",
    "madd",
    "bənd",
    "şək",
    "cədv",
    "nömr",
    # Azerbaijani month abbreviations
    "yan",
    "fev",
    "mar",
    "apr",
    "may",
    "iyn",
    "iyl",
    "avq",
    "sen",
    "okt",
    "noy",
    "dek",
    # English fallbacks in mixed-language corpora
    "mr",
    "mrs",
    "ms",
    "sr",
    "jr",
    "st",
    "no",
    "vs",
    "etc",
    "jan",
    "feb",
    "jun",
    "jul",
    "aug",
    "sep",
    "oct",
    "nov",
    "dec",
}


def extract_dot_examples(
    corpus_path: Path,
    max_docs: int | None = 50000,
    max_examples: int | None = 250000,
) -> Tuple[List[Dict[str, float | str]], np.ndarray]:
    examples: List[Dict[str, float | str]] = []
    labels: List[int] = []
    doc_count = 0

    # Sentence-start cue for Azerbaijani/Latin text after punctuation/quotes.
    eos_re = re.compile(
        r"^\s*[\u201c\u201d\u2018\u2019\"'\)\]\}\u00bb\u203a]*\s*[A-ZƏÖÜİÇĞŞ]"
    )
    lower_init_re = re.compile(
        r"^\s*[\u201c\u201d\u2018\u2019\"'\(\[\{\u00ab\u2039]*\s*[a-zəöüğışç]"
    )
    decimal_re = re.compile(r"^\d\.\d$")

    with corpus_path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            text = line.strip()
            if not text:
                continue
            doc_count += 1
            if max_docs is not None and doc_count > max_docs:
                break

            for m in re.finditer(r"\.", text):
                i = m.start()
                left = text[:i]
                right = text[i + 1 :]

                prev_match = re.search(r"(\w+)$", left, flags=re.UNICODE)
                next_match = re.match(r"^\s*([\w]+)", right, flags=re.UNICODE)
                prev_tok = prev_match.group(1) if prev_match else ""
                next_tok = next_match.group(1) if next_match else ""

                around = text[max(0, i - 1) : min(len(text), i + 2)]
                right_stripped = right.strip()
                prev_tok_l = prev_tok.lower()
                next_initial = next_tok[:1]

                if decimal_re.match(around):
                    y = 0
                elif right_stripped == "":
                    y = 1
                elif prev_tok_l in ABBREV_SET:
                    y = 0
                elif (
                    len(prev_tok) == 1
                    and prev_tok.isupper()
                    and bool(next_initial)
                    and next_initial.isupper()
                ):
                    y = 0
                elif lower_init_re.match(right):
                    y = 0
                else:
                    y = 1 if eos_re.match(right) else 0

                feat: Dict[str, float | str] = {
                    "prev_tok": prev_tok_l,
                    "next_tok": next_tok.lower(),
                    "prev_len": float(len(prev_tok)),
                    "next_len": float(len(next_tok)),
                    "prev_is_upper": float(prev_tok.isupper() and bool(prev_tok)),
                    "next_is_upper_init": float(bool(next_initial.isupper())),
                    "next_is_lower_init": float(bool(next_initial.islower())),
                    "prev_is_digit": float(prev_tok.isdigit() and bool(prev_tok)),
                    "next_is_digit": float(next_tok.isdigit() and bool(next_tok)),
                    "prev_is_abbrev": float(prev_tok_l in ABBREV_SET),
                    "prev_short": float(len(prev_tok) <= 3 and bool(prev_tok)),
                    "prev_is_single_upper": float(
                        len(prev_tok) == 1 and prev_tok.isupper()
                    ),
                }
                examples.append(feat)
                labels.append(y)

                if max_examples is not None and len(examples) >= max_examples:
                    return examples, np.array(labels, dtype=np.int64)

    return examples, np.array(labels, dtype=np.int64)


def vectorize_dot_features(
    feats: Sequence[Dict[str, float | str]],
    max_vocab_tokens: int = 40000,
) -> np.ndarray:
    prev_counts = Counter()
    next_counts = Counter()
    for f in feats:
        prev_counts[str(f["prev_tok"])] += 1
        next_counts[str(f["next_tok"])] += 1

    prev_vocab = {
        tok: i for i, (tok, _c) in enumerate(prev_counts.most_common(max_vocab_tokens // 2))
    }
    next_vocab = {
        tok: i for i, (tok, _c) in enumerate(next_counts.most_common(max_vocab_tokens // 2))
    }

    n_num = 10
    off_prev = n_num
    off_next = n_num + len(prev_vocab)
    d = n_num + len(prev_vocab) + len(next_vocab)

    x = np.zeros((len(feats), d), dtype=np.float32)
    for i, f in enumerate(feats):
        x[i, 0] = float(f["prev_len"])
        x[i, 1] = float(f["next_len"])
        x[i, 2] = float(f["prev_is_upper"])
        x[i, 3] = float(f["next_is_upper_init"])
        x[i, 4] = float(f["next_is_lower_init"])
        x[i, 5] = float(f["prev_is_digit"])
        x[i, 6] = float(f["next_is_digit"])
        x[i, 7] = float(f["prev_is_abbrev"])
        x[i, 8] = float(f["prev_short"])
        x[i, 9] = float(f["prev_is_single_upper"])

        p = prev_vocab.get(str(f["prev_tok"]))
        if p is not None:
            x[i, off_prev + p] = 1.0
        n = next_vocab.get(str(f["next_tok"]))
        if n is not None:
            x[i, off_next + n] = 1.0
    return x


def _is_array_memory_error(exc: Exception) -> bool:
    return isinstance(exc, MemoryError) or exc.__class__.__name__ == "_ArrayMemoryError"


def split_train_dev_test_xy(
    x: np.ndarray,
    y: np.ndarray,
    test_ratio: float = 0.2,
    dev_ratio_within_train: float = 0.1,
    seed: int = SEED,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    idx = np.arange(len(y))
    rng = np.random.default_rng(seed)
    rng.shuffle(idx)

    n_test = int(len(y) * test_ratio)
    n_train_pool = len(y) - n_test
    n_dev = int(n_train_pool * dev_ratio_within_train)

    test_idx = idx[:n_test]
    dev_idx = idx[n_test : n_test + n_dev]
    train_idx = idx[n_test + n_dev :]

    return (
        x[train_idx],
        x[dev_idx],
        x[test_idx],
        y[train_idx],
        y[dev_idx],
        y[test_idx],
    )


def select_best_threshold_by_accuracy(
    proba: np.ndarray,
    y_true: np.ndarray,
    start: float = 0.35,
    stop: float = 0.8,
    step: float = 0.01,
) -> Tuple[float, float]:
    best_acc = -1.0
    best_threshold = 0.5
    threshold = start
    while threshold <= stop + 1e-12:
        pred = (proba >= threshold).astype(np.int64)
        acc = float((pred == y_true).mean())
        if acc > best_acc:
            best_acc = acc
            best_threshold = threshold
        threshold += step
    return best_threshold, best_acc


def run_task4(
    news_path: Path,
    max_docs: int | None = 30000,
    max_examples: int | None = 60000,
    max_vocab_tokens: int = 6000,
) -> Dict[str, float]:
    feats, y = extract_dot_examples(news_path, max_docs=max_docs, max_examples=max_examples)
    if len(y) < 1000:
        return {"error": 1.0, "num_examples": float(len(y))}

    cur_n = len(y)
    cur_vocab = max_vocab_tokens
    min_n = 8000
    min_vocab = 1000
    x = None

    while True:
        try:
            x = vectorize_dot_features(feats[:cur_n], max_vocab_tokens=cur_vocab)
            y_work = y[:cur_n]
            break
        except Exception as exc:
            if not _is_array_memory_error(exc):
                raise
            next_n = max(min_n, cur_n // 2)
            next_vocab = max(min_vocab, cur_vocab // 2)
            if next_n == cur_n and next_vocab == cur_vocab:
                return {
                    "error": 1.0,
                    "num_examples": float(len(y)),
                    "used_examples": float(cur_n),
                    "used_vocab_cap": float(cur_vocab),
                }
            cur_n = next_n
            cur_vocab = next_vocab

    x = x.astype(np.float32, copy=False)
    x[:, :2] = np.clip(x[:, :2], 0.0, 30.0)
    xtr, xdv, xte, ytr, ydv, yte = split_train_dev_test_xy(
        x,
        y_work,
        test_ratio=0.2,
        dev_ratio_within_train=0.1,
        seed=SEED,
    )

    l2_model = LogisticBinary(
        lr=0.2, epochs=40, reg_type="l2", reg_strength=1e-4
    ).fit(xtr, ytr)
    l2_threshold, l2_dev_acc = select_best_threshold_by_accuracy(
        l2_model.predict_proba(xdv), ydv
    )
    pred_l2 = (l2_model.predict_proba(xte) >= l2_threshold).astype(np.int64)
    m_l2 = classification_metrics(yte, pred_l2)

    l1_model = LogisticBinary(
        lr=0.2, epochs=40, reg_type="l1", reg_strength=1e-4
    ).fit(xtr, ytr)
    l1_threshold, l1_dev_acc = select_best_threshold_by_accuracy(
        l1_model.predict_proba(xdv), ydv
    )
    pred_l1 = (l1_model.predict_proba(xte) >= l1_threshold).astype(np.int64)
    m_l1 = classification_metrics(yte, pred_l1)

    p_l2_vs_l1 = mcnemar_exact_p(yte, pred_l2, pred_l1)
    majority_baseline_acc = max(float((yte == 1).mean()), float((yte == 0).mean()))
    return {
        "num_examples": float(len(y)),
        "used_examples": float(len(y_work)),
        "used_vocab_cap": float(cur_vocab),
        "num_features": float(x.shape[1]),
        "train_examples": float(len(ytr)),
        "dev_examples": float(len(ydv)),
        "test_examples": float(len(yte)),
        "majority_baseline_test_accuracy": majority_baseline_acc,
        "l2_dev_accuracy": l2_dev_acc,
        "l2_threshold": l2_threshold,
        "l2_accuracy": m_l2["accuracy"],
        "l2_f1": m_l2["f1"],
        "l1_dev_accuracy": l1_dev_acc,
        "l1_threshold": l1_threshold,
        "l1_accuracy": m_l1["accuracy"],
        "l1_f1": m_l1["f1"],
        "p_l2_vs_l1": p_l2_vs_l1,
    }
