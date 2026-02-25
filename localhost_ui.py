#!/usr/bin/env python3
from __future__ import annotations

import math
import re
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
from core.sentence_boundary_task_v2 import fit_task4_v2_assets
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


@st.cache_resource(show_spinner=False)
def _train_task4_demo(
    dataset_path: str,
    max_examples: int,
) -> Dict[str, Any]:
    return fit_task4_v2_assets(
        dataset_path=dataset_path,
        max_examples=max_examples,
    )


def _dataset_signature(path: Path) -> Tuple[str, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def _extract_dot_features_for_text(text: str) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for match in re.finditer(r"\.", text):
        i = match.start()

        prev_token = ""
        j = i - 1
        while j >= 0 and text[j].isalpha():
            prev_token = text[j] + prev_token
            j -= 1

        next_token = ""
        j = i + 1
        while j < len(text) and text[j].isspace():
            j += 1
        while j < len(text) and text[j].isalpha():
            next_token += text[j]
            j += 1

        rows.append(
            {
                "dot_index": i,
                "prev_token": prev_token.lower(),
                "next_token": next_token.lower(),
                "prev_len": len(prev_token),
                "next_len": len(next_token),
                "is_digit_before": int(i > 0 and text[i - 1].isdigit()),
            }
        )
    return rows


def _split_by_boundary_indices(text: str, boundary_indices: Sequence[int]) -> List[str]:
    if not boundary_indices:
        cleaned = text.strip()
        return [cleaned] if cleaned else []

    ordered = sorted({int(idx) for idx in boundary_indices})
    out: List[str] = []
    start = 0
    for idx in ordered:
        end = idx + 1
        piece = text[start:end].strip()
        if piece:
            out.append(piece)
        start = end
    tail = text[start:].strip()
    if tail:
        out.append(tail)
    return out


def _predict_dot_boundaries(
    text: str, task4_assets: Mapping[str, Any]
) -> Tuple[pd.DataFrame, List[str]]:
    rows = _extract_dot_features_for_text(text)
    if not rows:
        empty = pd.DataFrame(
            columns=["dot_index", "prev_token", "next_token", "p_end", "prediction"]
        )
        return empty, []

    feature_rows = [
        {
            "prev_token": row["prev_token"],
            "next_token": row["next_token"],
            "prev_len": row["prev_len"],
            "next_len": row["next_len"],
            "is_digit_before": row["is_digit_before"],
        }
        for row in rows
    ]
    x = task4_assets["vectorizer"].transform(feature_rows)
    probs = task4_assets["best_model"].predict_proba(x)[:, 1]
    threshold = float(task4_assets["best_threshold"])

    boundary_indices: List[int] = []
    pred_rows: List[Dict[str, Any]] = []
    for row, prob in zip(rows, probs.tolist()):
        is_end = float(prob) >= threshold
        if is_end:
            boundary_indices.append(int(row["dot_index"]))
        pred_rows.append(
            {
                "dot_index": int(row["dot_index"]),
                "prev_token": str(row["prev_token"]),
                "next_token": str(row["next_token"]),
                "p_end": float(prob),
                "prediction": "Sentence End" if is_end else "Not End",
            }
        )

    sentences = _split_by_boundary_indices(text, boundary_indices)
    return pd.DataFrame(pred_rows), sentences


def _run_results_panel(
    news_path: Path,
    root_path: Path,
    task4_dataset_path: Path,
    max_sentences: int,
    min_freq: int,
    max_examples: int,
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
                dataset_path=task4_dataset_path,
                max_examples=max_examples,
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
                dataset_path=task4_dataset_path,
                max_examples=max_examples,
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


def _run_task4_demo_panel(task4_dataset_path: Path, default_max_examples: int) -> None:
    st.subheader("Sentence Boundary Demo")
    st.caption(
        "Predict whether each '.' in your text is a sentence boundary. "
        "The demo model is trained once per dataset/settings and then reused."
    )

    demo_default = min(max(int(default_max_examples), 2000), 100000)
    max_examples = st.number_input(
        "Demo max examples",
        min_value=2000,
        max_value=100000,
        value=demo_default,
        step=1000,
        key="task4_demo_max_examples",
    )

    text = st.text_area(
        "Text with dots",
        value="Dr. Ali bu gun geldi. O saat 14.30-da ders kecdi. Cox maraqli idi.",
        height=110,
        key="task4_demo_text",
    )

    settings_key = (_dataset_signature(task4_dataset_path), int(max_examples))
    if st.button("Analyze Dots", key="task4_demo_btn"):
        assets = _load_once_per_settings(
            state_slot="task4_v2_assets",
            settings_key=settings_key,
            spinner_text="Training sentence boundary demo model (first time only)...",
            loader=lambda: _train_task4_demo(
                str(task4_dataset_path),
                int(max_examples),
            ),
        )
        pred_df, sentences = _predict_dot_boundaries(text, assets)
        if pred_df.empty:
            st.warning("No '.' found in text.")
            return

        if sentences:
            sentence_df = pd.DataFrame(
                {
                    "sentence_no": list(range(1, len(sentences) + 1)),
                    "sentence": sentences,
                }
            )
            st.markdown("**Separated Sentences**")
            st.dataframe(sentence_df, use_container_width=True, hide_index=True)

        st.markdown("**Dot-level Decisions**")
        st.dataframe(pred_df, use_container_width=True, hide_index=True)
        st.caption(
            f"Model uses {int(assets['metrics']['num_examples'])} examples "
            f"(best penalty: {assets['best_penalty'].upper()}, "
            f"test F1: {assets['metrics']['best_test_f1']:.4f})."
        )


def main() -> None:
    st.set_page_config(page_title="NLP Project 2 Localhost UI", layout="wide")
    st.title("NLP Project 2 - Localhost UI")
    st.caption("Results dashboard + simple demos for unigram/bigram/trigram, sentiment, and sentence boundary.")

    with st.sidebar:
        st.header("Run Settings")
        news_path_input = st.text_input("News corpus path", value=str(NEWS_CORPUS_PATH))
        root_path_input = st.text_input("Project root path", value=str(ROOT))
        task4_dataset_input = st.text_input(
            "Task4 v2 dataset path", value=str(ROOT / "dot_labeled_data.csv")
        )
        max_sentences = st.number_input(
            "Task1/2 max sentences", min_value=1000, max_value=200000, value=120000, step=1000
        )
        min_freq = st.number_input("Task1/2 min freq", min_value=1, max_value=10, value=2, step=1)
        max_examples = st.number_input(
            "Task4 max examples", min_value=2000, max_value=200000, value=60000, step=1000
        )

    news_path = Path(news_path_input)
    root_path = Path(root_path_input)
    task4_dataset_path = Path(task4_dataset_input)
    if not news_path.exists():
        st.error(f"News corpus not found: {news_path}")
        return
    if not root_path.exists():
        st.error(f"Project root not found: {root_path}")
        return
    if not task4_dataset_path.exists():
        st.error(f"Task4 v2 dataset not found: {task4_dataset_path}")
        return

    tab_results, tab_lm, tab_sentiment, tab_boundary = st.tabs(
        ["Results", "Unigram/Bigram", "Sentiment", "Sentence Boundary"]
    )

    try:
        with tab_results:
            _run_results_panel(
                news_path,
                root_path,
                task4_dataset_path,
                int(max_sentences),
                int(min_freq),
                int(max_examples),
            )
        with tab_lm:
            _run_lm_demo_panel(news_path, int(max_sentences), int(min_freq))
        with tab_sentiment:
            _run_sentiment_demo_panel(root_path)
        with tab_boundary:
            _run_task4_demo_panel(task4_dataset_path, int(max_examples))
    except Exception as exc:
        st.exception(exc)


if __name__ == "__main__":
    main()
