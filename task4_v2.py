#!/usr/bin/env python3
import re
import numpy as np
import random
from pathlib import Path
from typing import List, Tuple
from collections import Counter

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

SEED = 42
random.seed(SEED)
np.random.seed(SEED)


ABBREVIATIONS = {
    # Titles
    "dr", "prof", "dos", "akad", "müəll", "cən",

    # Common Azerbaijani abbreviations
    "məs",        # məsələn
    "təxm",       # təxminən
    "səh",        # səhifə
    "madd",       # maddə
    "bənd",       # bənd
    "şək",        # şəkil
    "cədv",       # cədvəl
    "nömr",       # nömrə

    # Months
    "yan", "fev", "mar", "apr", "may",
    "iyn", "iyl", "avq", "sen",
    "okt", "noy", "dek"
}


# =====================================
# 1. DATA EXTRACTION (NO LEAKAGE)
# =====================================

def extract_dot_examples(corpus_path: str,
                         max_docs: int = 50000,
                         max_examples: int = 200000):

    feats = []
    labels = []

    with open(corpus_path, "r", encoding="utf-8") as f:
        docs = 0
        for line in f:
            line = line.strip()
            if not line:
                continue

            docs += 1
            if docs > max_docs:
                break

            for i, ch in enumerate(line):
                if ch != ".":
                    continue

                if len(feats) >= max_examples:
                    break

                # -------- Extract tokens --------
                prev_token = ""
                j = i - 1
                while j >= 0 and line[j].isalpha():
                    prev_token = line[j] + prev_token
                    j -= 1

                next_token = ""
                j = i + 1
                while j < len(line) and line[j].isalpha():
                    next_token += line[j]
                    j += 1

                prev_len = len(prev_token)
                next_len = len(next_token)
                is_digit_before = int(i > 0 and line[i - 1].isdigit())

                # ---------- FEATURES (no leakage) ----------
                feats.append({
                    "prev_tok": prev_token.lower(),
                    "next_tok": next_token.lower(),
                    "prev_len": prev_len,
                    "next_len": next_len,
                    "is_digit_before": is_digit_before
                })

                # ---------- WEAK LABEL ----------
                next_char = line[i + 1] if i + 1 < len(line) else " "
                is_next_upper = next_char.isupper()
                is_abbrev = prev_token.lower() in ABBREVIATIONS

                label = 1

                # Case 1: abbreviation
                if is_abbrev:
                    label = 0

                # Case 2: initial + surname (A. Suleymanov)
                elif len(prev_token) == 1 and prev_token.isupper() and next_token and next_token[0].isupper():
                    label = 0

                # Case 3: next word not uppercase → likely not sentence boundary
                elif not is_next_upper:
                    label = 0

                labels.append(label)

    return feats, np.array(labels)

import csv

def save_labeled_data_to_csv(feats, labels, filename="dot_labeled_data.csv"):
    with open(filename, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "prev_token",
            "next_token",
            "prev_len",
            "next_len",
            "is_digit_before",
            "label"
        ])

        for feat, label in zip(feats, labels):
            writer.writerow([
                feat["prev_tok"],
                feat["next_tok"],
                feat["prev_len"],
                feat["next_len"],
                feat["is_digit_before"],
                label
            ])

# =====================================
# 2. VECTORIZE (TOKEN IDENTITY)
# =====================================

from scipy.sparse import lil_matrix

def vectorize(feats, max_vocab=2000):

    prev_counts = Counter(f["prev_tok"] for f in feats)
    next_counts = Counter(f["next_tok"] for f in feats)

    prev_vocab = {w: i for i, (w, _) in enumerate(prev_counts.most_common(max_vocab))}
    next_vocab = {w: i for i, (w, _) in enumerate(next_counts.most_common(max_vocab))}

    dim = 3 + len(prev_vocab) + len(next_vocab)
    X = lil_matrix((len(feats), dim), dtype=np.float32)

    for i, f in enumerate(feats):

        X[i, 0] = min(f["prev_len"], 20)
        X[i, 1] = min(f["next_len"], 20)
        X[i, 2] = f["is_digit_before"]

        if f["prev_tok"] in prev_vocab:
            X[i, 3 + prev_vocab[f["prev_tok"]]] = 1

        if f["next_tok"] in next_vocab:
            X[i, 3 + len(prev_vocab) + next_vocab[f["next_tok"]]] = 1

    return X.tocsr()


# =====================================
# 3. THRESHOLD TUNING (F1-based)
# =====================================

def find_best_threshold(model, X_dev, y_dev):
    probs = model.predict_proba(X_dev)[:, 1]
    best_thresh = 0.5
    best_f1 = 0

    for t in np.linspace(0.1, 0.9, 17):
        preds = (probs >= t).astype(int)
        f1 = f1_score(y_dev, preds)
        if f1 > best_f1:
            best_f1 = f1
            best_thresh = t

    return best_thresh


# =====================================
# 4. MAIN
# =====================================

def main():

    corpus_path = "Corpora/News/corpus.txt"

    feats, y = extract_dot_examples(corpus_path)
    save_labeled_data_to_csv(feats, y)
    print("Labeled dataset saved to dot_labeled_data.csv")
    X = vectorize(feats)

    # 70 / 15 / 15 split
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.15, random_state=SEED
    )

    X_train, X_dev, y_train, y_dev = train_test_split(
        X_train_full, y_train_full, test_size=0.1765, random_state=SEED
    )

    print("Train:", X_train.shape[0])
    print("Dev:", X_dev.shape[0])
    print("Test:", X_test.shape[0])

    results = {}

    for penalty in ["l2", "l1"]:

        best_model = None
        best_f1 = 0
        best_C = None

        # Tune regularization strength
        for C in [0.01, 0.1, 1, 10]:
            model = LogisticRegression(
                penalty=penalty,
                solver="liblinear",
                C=C,
                max_iter=1000,
                random_state=SEED
            )
            model.fit(X_train, y_train)

            thresh = find_best_threshold(model, X_dev, y_dev)
            preds = (model.predict_proba(X_dev)[:, 1] >= thresh).astype(int)
            f1 = f1_score(y_dev, preds)

            if f1 > best_f1:
                best_f1 = f1
                best_model = model
                best_C = C

        # Final evaluation on test
        thresh = find_best_threshold(best_model, X_dev, y_dev)
        preds = (best_model.predict_proba(X_test)[:, 1] >= thresh).astype(int)

        results[penalty] = {
            "C": best_C,
            "accuracy": accuracy_score(y_test, preds),
            "precision": precision_score(y_test, preds),
            "recall": recall_score(y_test, preds),
            "f1": f1_score(y_test, preds)
        }

    print("\n=== FINAL RESULTS ===\n")
    for k, v in results.items():
        print(f"{k.upper()} REGULARIZATION")
        for metric, value in v.items():
            print(f"{metric}: {value:.4f}")
        print()


if __name__ == "__main__":
    main()