#!/usr/bin/env python3
from __future__ import annotations

import math
import re
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
import pandas as pd
import streamlit as st

from assignment_tasks import run_task1, run_task1_task2, run_task2, run_task3, run_task4
from core.language_modeling import (
    add_markers,
    build_ngram_counts,
    load_news_sentences,
    p_bigram_mle,
    p_trigram_mle,
    p_unigram_mle,
    perplexity_bigram,
    perplexity_trigram,
    perplexity_unigram,
    replace_rare_with_unk,
    train_dev_test_split,
    tune_linear_interpolation,
)
from core.ml import LogisticBinary, classification_metrics
from core.paths import NEWS_CORPUS_PATH, ROOT
from core.sentence_boundary_task import (
    ABBREV_SET,
    extract_dot_examples,
    select_best_threshold_by_accuracy,
    split_train_dev_test_xy,
)
from core.sentiment_task import (
    build_vocab_for_classification,
    load_sentiment_dataset,
    sentiment_dataset_path_from_root,
    sentiment_lexicon_features,
    vectorize_bow_counts,
)
from core.text_utils import tokenize_words


TASK2_SMOOTH_KEYS = [
    "ppl_trigram_laplace",
    "ppl_trigram_interpolation",
    "ppl_trigram_backoff",
    "ppl_trigram_kneser_ney",
]


def _format_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return str(value)
        return round(value, 6)
    return value


def _metrics_frame(metrics: Mapping[str, Any]) -> pd.DataFrame:
    rows = [{"metric": key, "value": _format_value(value)} for key, value in metrics.items()]
    return pd.DataFrame(rows)


def _map_to_vocab(tokens: Sequence[str], vocab: set[str]) -> List[str]:
    return [tok if tok in vocab else "<UNK>" for tok in tokens]


@st.cache_resource(show_spinner=False)
def _load_lm_assets(
    news_path: str,
    max_sentences: int,
    min_freq: int,
) -> Dict[str, Any]:
    path = Path(news_path)
    sentences = load_news_sentences(path, max_sentences=max_sentences)
    if len(sentences) < 10:
        raise ValueError("Not enough sentences in corpus to build language model demo.")

    train, dev, _test = train_dev_test_split(sentences, train_ratio=0.8, dev_ratio=0.1)
    train_mapped, dev_mapped, vocab = replace_rare_with_unk(train, dev, min_freq=min_freq)
    counts = build_ngram_counts(train_mapped, vocab)
    lambdas = tune_linear_interpolation(dev_mapped, counts, step=0.1)
    return {
        "counts": counts,
        "vocab": vocab,
        "lambdas": lambdas,
        "num_sentences": len(sentences),
        "num_train_sentences": len(train_mapped),
    }


def _safe_perplexity_text(sentence: str, lm_assets: Mapping[str, Any]) -> Dict[str, float]:
    counts = lm_assets["counts"]
    vocab = lm_assets["vocab"]
    lambdas = lm_assets["lambdas"]
    tokens = tokenize_words(sentence)
    if not tokens:
        raise ValueError("Please enter a sentence with at least one token.")
    mapped = _map_to_vocab(tokens, vocab)
    seq = [mapped]
    return {
        "ppl_unigram_mle": perplexity_unigram(seq, counts, mode="mle"),
        "ppl_bigram_mle": perplexity_bigram(seq, counts, mode="mle"),
        "ppl_trigram_mle": perplexity_trigram(seq, counts, mode="mle"),
        "ppl_trigram_interpolation": perplexity_trigram(
            seq, counts, mode="interpolation", lambdas=lambdas
        ),
    }


def _token_probability_rows(sentence: str, lm_assets: Mapping[str, Any]) -> pd.DataFrame:
    counts = lm_assets["counts"]
    vocab = lm_assets["vocab"]
    tokens = tokenize_words(sentence)
    mapped = _map_to_vocab(tokens, vocab)
    rows: List[Dict[str, Any]] = []
    s3 = add_markers(mapped, 3)
    for i in range(2, len(s3)):
        u, v, w = s3[i - 2], s3[i - 1], s3[i]
        rows.append(
            {
                "context": f"{u} {v}",
                "word": w,
                "p_unigram": p_unigram_mle(w, counts),
                "p_bigram": p_bigram_mle(v, w, counts),
                "p_trigram": p_trigram_mle(u, v, w, counts),
            }
        )
    return pd.DataFrame(rows)


def _load_once_per_settings(
    state_slot: str,
    settings_key: Tuple[Any, ...],
    spinner_text: str,
    loader: Callable[[], Dict[str, Any]],
) -> Dict[str, Any]:
    key_slot = f"{state_slot}_key"
    if st.session_state.get(key_slot) == settings_key and state_slot in st.session_state:
        return st.session_state[state_slot]
    with st.spinner(spinner_text):
        assets = loader()
    st.session_state[state_slot] = assets
    st.session_state[key_slot] = settings_key
    return assets


def _stratified_take_indices(y: np.ndarray, max_samples: int) -> np.ndarray:
    total = len(y)
    if max_samples <= 0 or max_samples >= total:
        return np.arange(total, dtype=np.int64)

    rng = np.random.default_rng(42)
    classes, counts = np.unique(y, return_counts=True)
    raw = counts / max(total, 1) * max_samples
    take = np.floor(raw).astype(np.int64)
    take = np.maximum(take, 1)
    take = np.minimum(take, counts)

    while int(take.sum()) < max_samples:
        spare = counts - take
        candidate = int(np.argmax(spare))
        if spare[candidate] <= 0:
            break
        take[candidate] += 1

    while int(take.sum()) > max_samples:
        candidate = int(np.argmax(take))
        if take[candidate] <= 1:
            break
        take[candidate] -= 1

    parts: List[np.ndarray] = []
    for cls, n_take in zip(classes.tolist(), take.tolist()):
        cls_idx = np.where(y == cls)[0]
        rng.shuffle(cls_idx)
        parts.append(cls_idx[:n_take])

    out = np.concatenate(parts).astype(np.int64, copy=False)
    rng.shuffle(out)
    if len(out) > max_samples:
        out = out[:max_samples]
    return out


@st.cache_resource(show_spinner=False)
def _train_sentiment_demo(
    root_path: str,
    dataset_path_input: str,
    max_samples: int,
    max_vocab: int,
    min_freq: int,
) -> Dict[str, Any]:
    root = Path(root_path)
    dataset_path = (
        Path(dataset_path_input)
        if dataset_path_input.strip()
        else sentiment_dataset_path_from_root(root)
    )
    texts, labels, data_source = load_sentiment_dataset(dataset_path)
    if len(texts) < 200:
        raise ValueError(f"Sentiment dataset too small or invalid: {data_source}")

    y = np.array(labels, dtype=np.int64)
    subset_idx = _stratified_take_indices(y, max_samples)
    texts_sub = [texts[int(i)] for i in subset_idx.tolist()]
    y_sub = y[subset_idx]

    train_size = max(int(0.8 * len(y_sub)), 1)
    x_train_text = texts_sub[:train_size]
    x_test_text = texts_sub[train_size:]
    y_train = y_sub[:train_size]
    y_test = y_sub[train_size:]

    vocab = build_vocab_for_classification(x_train_text, min_freq=min_freq, max_vocab=max_vocab)
    xtr_counts = vectorize_bow_counts(x_train_text, vocab)
    xte_counts = vectorize_bow_counts(x_test_text, vocab)
    xtr_lex = sentiment_lexicon_features(x_train_text)
    xte_lex = sentiment_lexicon_features(x_test_text)
    xtr = np.hstack([xtr_counts, xtr_lex])
    xte = np.hstack([xte_counts, xte_lex])

    model = LogisticBinary(lr=0.2, epochs=35, reg_type="l2", reg_strength=1e-4).fit(xtr, y_train)
    pred = model.predict(xte)
    test_metrics = classification_metrics(y_test, pred) if len(y_test) else {"accuracy": 0.0}

    return {
        "model": model,
        "vocab": vocab,
        "dataset_path": str(dataset_path),
        "num_samples": len(texts_sub),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "test_accuracy": float(test_metrics.get("accuracy", 0.0)),
    }


def _predict_sentiment(text: str, sentiment_assets: Mapping[str, Any]) -> Dict[str, Any]:
    vocab = sentiment_assets["vocab"]
    model = sentiment_assets["model"]
    x_count = vectorize_bow_counts([text], vocab)
    x_lex = sentiment_lexicon_features([text])
    x = np.hstack([x_count, x_lex])
    proba = float(model.predict_proba(x)[0])
    label = "positive" if proba >= 0.5 else "negative"
    return {"label": label, "positive_probability": proba}


def _build_task4_vocabs(
    feats: Sequence[Mapping[str, float | str]],
    max_vocab_tokens: int,
) -> Tuple[Dict[str, int], Dict[str, int]]:
    prev_counts = Counter(str(f["prev_tok"]) for f in feats)
    next_counts = Counter(str(f["next_tok"]) for f in feats)
    half = max(max_vocab_tokens // 2, 1)
    prev_vocab = {tok: i for i, (tok, _c) in enumerate(prev_counts.most_common(half))}
    next_vocab = {tok: i for i, (tok, _c) in enumerate(next_counts.most_common(half))}
    return prev_vocab, next_vocab


def _vectorize_task4_with_vocabs(
    feats: Sequence[Mapping[str, float | str]],
    prev_vocab: Mapping[str, int],
    next_vocab: Mapping[str, int],
) -> np.ndarray:
    n_num = 10
    off_prev = n_num
    off_next = n_num + len(prev_vocab)
    dim = n_num + len(prev_vocab) + len(next_vocab)
    x = np.zeros((len(feats), dim), dtype=np.float32)

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


@st.cache_resource(show_spinner=False)
def _train_task4_demo(
    news_path: str,
    max_docs: int,
    max_examples: int,
    max_vocab_tokens: int,
) -> Dict[str, Any]:
    feats, y = extract_dot_examples(
        Path(news_path),
        max_docs=max_docs,
        max_examples=max_examples,
    )
    if len(y) < 1000:
        raise ValueError("Not enough dot examples to train sentence boundary demo model.")

    prev_vocab, next_vocab = _build_task4_vocabs(feats, max_vocab_tokens=max_vocab_tokens)
    x = _vectorize_task4_with_vocabs(feats, prev_vocab, next_vocab)
    x[:, :2] = np.clip(x[:, :2], 0.0, 30.0)

    xtr, xdv, xte, ytr, ydv, yte = split_train_dev_test_xy(
        x,
        y,
        test_ratio=0.2,
        dev_ratio_within_train=0.1,
    )
    model = LogisticBinary(lr=0.2, epochs=40, reg_type="l2", reg_strength=1e-4).fit(xtr, ytr)
    threshold, dev_acc = select_best_threshold_by_accuracy(model.predict_proba(xdv), ydv)
    pred_test = (model.predict_proba(xte) >= threshold).astype(np.int64)
    m_test = classification_metrics(yte, pred_test)
    return {
        "model": model,
        "threshold": threshold,
        "prev_vocab": prev_vocab,
        "next_vocab": next_vocab,
        "num_examples": len(y),
        "test_accuracy": float(m_test["accuracy"]),
        "dev_accuracy": float(dev_acc),
    }


def _extract_dot_features_for_text(text: str) -> List[Dict[str, float | str]]:
    features: List[Dict[str, float | str]] = []
    for match in re.finditer(r"\.", text):
        i = match.start()
        left = text[:i]
        right = text[i + 1 :]
        prev_match = re.search(r"(\w+)$", left, flags=re.UNICODE)
        next_match = re.match(r"^\s*([\w]+)", right, flags=re.UNICODE)
        prev_tok = prev_match.group(1) if prev_match else ""
        next_tok = next_match.group(1) if next_match else ""
        next_initial = next_tok[:1]
        prev_tok_l = prev_tok.lower()

        features.append(
            {
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
                "prev_is_single_upper": float(len(prev_tok) == 1 and prev_tok.isupper()),
                "dot_index": float(i),
            }
        )
    return features


def _predict_dot_boundaries(text: str, task4_assets: Mapping[str, Any]) -> pd.DataFrame:
    feats = _extract_dot_features_for_text(text)
    if not feats:
        return pd.DataFrame(columns=["dot_index", "prev_tok", "next_tok", "p_end", "prediction"])
    x = _vectorize_task4_with_vocabs(feats, task4_assets["prev_vocab"], task4_assets["next_vocab"])
    x[:, :2] = np.clip(x[:, :2], 0.0, 30.0)
    proba = task4_assets["model"].predict_proba(x)
    threshold = float(task4_assets["threshold"])

    rows: List[Dict[str, Any]] = []
    for f, p in zip(feats, proba.tolist()):
        rows.append(
            {
                "dot_index": int(f["dot_index"]),
                "prev_tok": f["prev_tok"],
                "next_tok": f["next_tok"],
                "p_end": float(p),
                "prediction": "Sentence End" if p >= threshold else "Not End",
            }
        )
    return pd.DataFrame(rows)


def _run_results_panel(
    news_path: Path,
    root_path: Path,
    max_sentences: int,
    min_freq: int,
    max_docs: int,
    max_examples: int,
    max_vocab_tokens: int,
) -> None:
    st.subheader("Task Results")
    if "task_results" not in st.session_state:
        st.session_state["task_results"] = {}

    col1, col2, col3, col4, col5 = st.columns(5)
    if col1.button("Run Task 1", use_container_width=True):
        with st.spinner("Running Task 1..."):
            st.session_state["task_results"]["Task 1"] = run_task1(
                news_path=news_path,
                max_sentences=max_sentences,
                min_freq=min_freq,
            )
    if col2.button("Run Task 2", use_container_width=True):
        with st.spinner("Running Task 2..."):
            metrics = run_task2(
                news_path=news_path,
                max_sentences=max_sentences,
                min_freq=min_freq,
            )
            metrics = dict(metrics)
            best_key = min(TASK2_SMOOTH_KEYS, key=lambda k: metrics[k])
            metrics["best_smoothing_by_ppl"] = best_key
            st.session_state["task_results"]["Task 2"] = metrics
    if col3.button("Run Task 3", use_container_width=True):
        with st.spinner("Running Task 3..."):
            st.session_state["task_results"]["Task 3"] = run_task3(root_path)
    if col4.button("Run Task 4", use_container_width=True):
        with st.spinner("Running Task 4..."):
            st.session_state["task_results"]["Task 4"] = run_task4(
                news_path=news_path,
                max_docs=max_docs,
                max_examples=max_examples,
                max_vocab_tokens=max_vocab_tokens,
            )
    if col5.button("Run All", use_container_width=True):
        with st.spinner("Running all tasks..."):
            t12 = run_task1_task2(
                news_path=news_path,
                max_sentences=max_sentences,
                min_freq=min_freq,
            )
            task1 = {
                "num_sentences": t12.get("num_sentences"),
                "vocab_size": t12.get("vocab_size"),
                "ppl_unigram_mle": t12.get("ppl_unigram_mle"),
                "ppl_bigram_mle": t12.get("ppl_bigram_mle"),
                "ppl_trigram_mle": t12.get("ppl_trigram_mle"),
            }
            task2 = {
                "num_sentences": t12.get("num_sentences"),
                "vocab_size": t12.get("vocab_size"),
                "interp_lambda1": t12.get("interp_lambda1"),
                "interp_lambda2": t12.get("interp_lambda2"),
                "interp_lambda3": t12.get("interp_lambda3"),
                "ppl_trigram_laplace": t12.get("ppl_trigram_laplace"),
                "ppl_trigram_interpolation": t12.get("ppl_trigram_interpolation"),
                "ppl_trigram_backoff": t12.get("ppl_trigram_backoff"),
                "ppl_trigram_kneser_ney": t12.get("ppl_trigram_kneser_ney"),
            }
            task2["best_smoothing_by_ppl"] = min(TASK2_SMOOTH_KEYS, key=lambda k: task2[k])
            task3 = run_task3(root_path)
            task4 = run_task4(
                news_path=news_path,
                max_docs=max_docs,
                max_examples=max_examples,
                max_vocab_tokens=max_vocab_tokens,
            )
            st.session_state["task_results"] = {
                "Task 1": task1,
                "Task 2": task2,
                "Task 3": task3,
                "Task 4": task4,
            }

    if st.button("Clear Results"):
        st.session_state["task_results"] = {}

    if not st.session_state["task_results"]:
        st.info("Run any task to see metrics.")
        return

    for task_name, metrics in st.session_state["task_results"].items():
        st.markdown(f"**{task_name}**")
        st.dataframe(_metrics_frame(metrics), use_container_width=True, hide_index=True)


def _run_lm_demo_panel(news_path: Path, max_sentences: int, min_freq: int) -> None:
    st.subheader("Unigram / Bigram / Trigram Demo")
    st.caption("Type a sentence and inspect perplexity and token-level probabilities.")
    sentence = st.text_area(
        "Input sentence",
        value="Bugun hava yaxsidir ve men NLP oyrenirem.",
        height=90,
        key="lm_demo_text",
    )
    if st.button("Analyze LM Sentence", key="lm_demo_btn"):
        lm_assets = _load_once_per_settings(
            state_slot="lm_assets",
            settings_key=(str(news_path), int(max_sentences), int(min_freq)),
            spinner_text="Loading language model assets...",
            loader=lambda: _load_lm_assets(str(news_path), int(max_sentences), int(min_freq)),
        )
        with st.spinner("Scoring sentence..."):
            ppl = _safe_perplexity_text(sentence, lm_assets)
            probs = _token_probability_rows(sentence, lm_assets)

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Unigram PPL", _format_value(ppl["ppl_unigram_mle"]))
        col2.metric("Bigram PPL", _format_value(ppl["ppl_bigram_mle"]))
        col3.metric("Trigram PPL", _format_value(ppl["ppl_trigram_mle"]))
        col4.metric("Trigram Interp PPL", _format_value(ppl["ppl_trigram_interpolation"]))
        st.dataframe(probs, use_container_width=True, hide_index=True)


def _run_sentiment_demo_panel(root_path: Path) -> None:
    st.subheader("Sentiment Demo")
    st.caption("Simple interactive predictor trained from your sentiment dataset.")

    col1, col2, col3 = st.columns(3)
    max_samples = col1.number_input(
        "Demo max samples", min_value=500, max_value=50000, value=6000, step=500
    )
    max_vocab = col2.number_input(
        "Demo max vocab", min_value=1000, max_value=50000, value=12000, step=500
    )
    min_freq = col3.number_input("Min token freq", min_value=1, max_value=10, value=2, step=1)

    dataset_default = str(sentiment_dataset_path_from_root(root_path))
    dataset_input = st.text_input("Dataset path (optional override)", value=dataset_default)

    sentiment_text = st.text_area(
        "Text to classify",
        value="Bu film cox gozel idi!",
        height=90,
        key="sentiment_demo_text",
    )

    settings_key = (
        str(root_path),
        dataset_input.strip(),
        int(max_samples),
        int(max_vocab),
        int(min_freq),
    )
    if st.button("Predict Sentiment", key="sent_demo_btn"):
        assets = _load_once_per_settings(
            state_slot="sentiment_assets",
            settings_key=settings_key,
            spinner_text="Training sentiment demo model...",
            loader=lambda: _train_sentiment_demo(
                str(root_path),
                dataset_input.strip(),
                int(max_samples),
                int(max_vocab),
                int(min_freq),
            ),
        )
        prediction = _predict_sentiment(sentiment_text, assets)
        st.metric("Prediction", prediction["label"])
        st.metric("Positive Probability", _format_value(prediction["positive_probability"]))
        st.caption(
            f"Demo model uses {assets['num_samples']} samples from {assets['dataset_path']} "
            f"(test accuracy ~ {assets['test_accuracy']:.4f})."
        )


def _run_task4_demo_panel(news_path: Path) -> None:
    st.subheader("Sentence Boundary Demo")
    st.caption("Predict whether each '.' in your text is a sentence boundary.")

    col1, col2, col3 = st.columns(3)
    max_docs = col1.number_input(
        "Demo max docs", min_value=1000, max_value=50000, value=10000, step=1000
    )
    max_examples = col2.number_input(
        "Demo max examples", min_value=2000, max_value=100000, value=25000, step=1000
    )
    max_vocab_tokens = col3.number_input(
        "Demo vocab cap", min_value=1000, max_value=20000, value=4000, step=500
    )

    text = st.text_area(
        "Text with dots",
        value="Dr. Ali bu gun geldi. O saat 14.30-da ders kecdi. Cox maraqli idi.",
        height=110,
        key="task4_demo_text",
    )

    settings_key = (
        str(news_path),
        int(max_docs),
        int(max_examples),
        int(max_vocab_tokens),
    )
    if st.button("Analyze Dots", key="task4_demo_btn"):
        assets = _load_once_per_settings(
            state_slot="task4_assets",
            settings_key=settings_key,
            spinner_text="Training sentence boundary demo model...",
            loader=lambda: _train_task4_demo(
                str(news_path),
                int(max_docs),
                int(max_examples),
                int(max_vocab_tokens),
            ),
        )
        pred_df = _predict_dot_boundaries(text, assets)
        if pred_df.empty:
            st.warning("No '.' found in text.")
            return
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Demo model trained with {assets['num_examples']} examples "
            f"(dev acc: {assets['dev_accuracy']:.4f}, test acc: {assets['test_accuracy']:.4f})."
        )


def main() -> None:
    st.set_page_config(page_title="NLP Project 2 Localhost UI", layout="wide")
    st.title("NLP Project 2 - Localhost UI")
    st.caption("Results dashboard + simple demos for unigram/bigram/trigram, sentiment, and sentence boundary.")

    with st.sidebar:
        st.header("Run Settings")
        news_path_input = st.text_input("News corpus path", value=str(NEWS_CORPUS_PATH))
        root_path_input = st.text_input("Project root path", value=str(ROOT))
        max_sentences = st.number_input(
            "Task1/2 max sentences", min_value=1000, max_value=200000, value=120000, step=1000
        )
        min_freq = st.number_input("Task1/2 min freq", min_value=1, max_value=10, value=2, step=1)
        max_docs = st.number_input(
            "Task4 max docs", min_value=1000, max_value=100000, value=30000, step=1000
        )
        max_examples = st.number_input(
            "Task4 max examples", min_value=2000, max_value=200000, value=60000, step=1000
        )
        max_vocab_tokens = st.number_input(
            "Task4 max vocab tokens", min_value=1000, max_value=30000, value=6000, step=500
        )

    news_path = Path(news_path_input)
    root_path = Path(root_path_input)
    if not news_path.exists():
        st.error(f"News corpus not found: {news_path}")
        return
    if not root_path.exists():
        st.error(f"Project root not found: {root_path}")
        return

    tab_results, tab_lm, tab_sentiment, tab_boundary = st.tabs(
        ["Results", "Unigram/Bigram", "Sentiment", "Sentence Boundary"]
    )

    try:
        with tab_results:
            _run_results_panel(
                news_path,
                root_path,
                int(max_sentences),
                int(min_freq),
                int(max_docs),
                int(max_examples),
                int(max_vocab_tokens),
            )
        with tab_lm:
            _run_lm_demo_panel(news_path, int(max_sentences), int(min_freq))
        with tab_sentiment:
            _run_sentiment_demo_panel(root_path)
        with tab_boundary:
            _run_task4_demo_panel(news_path)
    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
