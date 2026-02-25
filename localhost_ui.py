#!/usr/bin/env python3
from __future__ import annotations

import math
import re
import hashlib
import pickle
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
from core.ml import LogisticBinary
from core.paths import NEWS_CORPUS_PATH, ROOT
from core.reporting import save_metrics_text
from core.sentence_boundary_task_v2 import fit_task4_v2_assets
from core.sentiment_task import (
    SKLEARN_AVAILABLE,
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

LM_DEFAULT_EXAMPLE_SENTENCE = "Bugun hava yaxsidir ve men NLP oyrenirem."
LM_SMOOTHING_TO_KEY = {
    "Laplace": "ppl_trigram_laplace",
    "Interpolation": "ppl_trigram_interpolation",
    "Backoff": "ppl_trigram_backoff",
    "Kneser-Ney": "ppl_trigram_kneser_ney",
}


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


def _path_signature(path: Path) -> Tuple[str, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return str(resolved), int(stat.st_size), int(stat.st_mtime_ns)


def _lm_demo_assets_file(
    news_resolved: Path,
    news_size: int,
    news_mtime_ns: int,
    max_sentences: int,
    min_freq: int,
) -> Path:
    key = (
        "lm_demo_v1|"
        f"{news_resolved}|{int(news_size)}|{int(news_mtime_ns)}|"
        f"{int(max_sentences)}|{int(min_freq)}"
    )
    model_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return ROOT / "models" / "lm_demo" / f"lm_demo_{model_hash}.pkl"


def _load_pickled_lm_assets(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    required = {
        "counts",
        "vocab",
        "lambdas",
        "num_sentences",
        "num_train_sentences",
        "example_sentence",
        "example_perplexities",
        "example_token_rows",
    }
    if not required.issubset(payload):
        return None
    return payload


def _lm_sentence_perplexities(
    mapped_tokens: Sequence[str],
    counts: Any,
    lambdas: Tuple[float, float, float],
) -> Dict[str, float]:
    seq = [list(mapped_tokens)]
    return {
        "ppl_unigram_mle": perplexity_unigram(seq, counts, mode="mle"),
        "ppl_unigram_laplace": perplexity_unigram(seq, counts, mode="laplace"),
        "ppl_bigram_mle": perplexity_bigram(seq, counts, mode="mle"),
        "ppl_bigram_laplace": perplexity_bigram(seq, counts, mode="laplace"),
        "ppl_trigram_mle": perplexity_trigram(seq, counts, mode="mle"),
        "ppl_trigram_laplace": perplexity_trigram(seq, counts, mode="laplace"),
        "ppl_trigram_interpolation": perplexity_trigram(
            seq, counts, mode="interpolation", lambdas=lambdas
        ),
        "ppl_trigram_backoff": perplexity_trigram(seq, counts, mode="backoff", d=0.75),
        "ppl_trigram_kneser_ney": perplexity_trigram(seq, counts, mode="kneser_ney", d=0.75),
    }


def _token_probability_rows_mapped(
    mapped_tokens: Sequence[str],
    counts: Any,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    s3 = add_markers(list(mapped_tokens), 3)
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
    return rows


@st.cache_resource(show_spinner=False)
def _load_lm_assets(
    news_path: str,
    max_sentences: int,
    min_freq: int,
) -> Dict[str, Any]:
    path = Path(news_path)
    news_resolved_str, news_size, news_mtime_ns = _path_signature(path)
    cache_file = _lm_demo_assets_file(
        news_resolved=Path(news_resolved_str),
        news_size=news_size,
        news_mtime_ns=news_mtime_ns,
        max_sentences=max_sentences,
        min_freq=min_freq,
    )
    payload = _load_pickled_lm_assets(cache_file)
    if payload is not None:
        payload["loaded_from_disk"] = 1.0
        payload["cache_path"] = str(cache_file)
        return payload

    sentences = load_news_sentences(path, max_sentences=max_sentences)
    if len(sentences) < 10:
        raise ValueError("Not enough sentences in corpus to build language model demo.")

    train, dev, _test = train_dev_test_split(sentences, train_ratio=0.8, dev_ratio=0.1)
    train_mapped, dev_mapped, vocab = replace_rare_with_unk(train, dev, min_freq=min_freq)
    counts = build_ngram_counts(train_mapped, vocab)
    lambdas = tune_linear_interpolation(dev_mapped, counts, step=0.1)
    example_tokens = tokenize_words(LM_DEFAULT_EXAMPLE_SENTENCE)
    example_mapped = _map_to_vocab(example_tokens, vocab)
    assets = {
        "counts": counts,
        "vocab": vocab,
        "lambdas": lambdas,
        "num_sentences": len(sentences),
        "num_train_sentences": len(train_mapped),
        "example_sentence": LM_DEFAULT_EXAMPLE_SENTENCE,
        "example_perplexities": _lm_sentence_perplexities(example_mapped, counts, lambdas),
        "example_token_rows": _token_probability_rows_mapped(example_mapped, counts),
        "loaded_from_disk": 0.0,
        "cache_path": str(cache_file),
    }
    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        with cache_file.open("wb") as f:
            pickle.dump(assets, f, protocol=pickle.HIGHEST_PROTOCOL)
    except (OSError, pickle.PickleError):
        pass
    return assets


def _mapped_sentence_or_raise(sentence: str, vocab: set[str]) -> List[str]:
    tokens = tokenize_words(sentence)
    if not tokens:
        raise ValueError("Please enter a sentence with at least one token.")
    return _map_to_vocab(tokens, vocab)


def _safe_perplexity_text(sentence: str, lm_assets: Mapping[str, Any]) -> Dict[str, float]:
    counts = lm_assets["counts"]
    vocab = lm_assets["vocab"]
    lambdas = lm_assets["lambdas"]
    mapped = _mapped_sentence_or_raise(sentence, vocab)
    return _lm_sentence_perplexities(mapped, counts, lambdas)


def _token_probability_rows(sentence: str, lm_assets: Mapping[str, Any]) -> pd.DataFrame:
    counts = lm_assets["counts"]
    vocab = lm_assets["vocab"]
    mapped = _mapped_sentence_or_raise(sentence, vocab)
    rows = _token_probability_rows_mapped(mapped, counts)
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


def _sentiment_demo_model_file(
    root: Path,
    dataset_resolved: Path,
    dataset_size: int,
    dataset_mtime_ns: int,
    max_samples: int,
    max_vocab: int,
    min_freq: int,
    label_scheme: str,
) -> Path:
    key = (
        "sentiment_demo_v4|"
        f"{dataset_resolved}|{int(dataset_size)}|{int(dataset_mtime_ns)}|"
        f"{int(max_samples)}|{int(max_vocab)}|{int(min_freq)}|{label_scheme}"
    )
    model_hash = hashlib.sha256(key.encode("utf-8")).hexdigest()[:24]
    return root / "models" / "sentiment_demo" / f"sentiment_demo_{model_hash}.pkl"


def _sentiment_label_name(label_value: int, label_scheme: str) -> str:
    if label_scheme == "ternary":
        return {-1: "negative", 0: "neutral", 1: "positive"}.get(label_value, str(label_value))
    if label_value == 1:
        return "positive"
    if label_value == 0:
        return "negative"
    return str(label_value)


def _load_pickled_sentiment_assets(path: Path) -> Dict[str, Any] | None:
    try:
        with path.open("rb") as f:
            payload = pickle.load(f)
    except (OSError, pickle.PickleError, EOFError, AttributeError, ValueError):
        return None
    if not isinstance(payload, dict):
        return None
    if "model" not in payload or "vocab" not in payload:
        return None
    return payload


@st.cache_resource(show_spinner=False)
def _train_sentiment_demo(
    root_path: str,
    dataset_path_input: str,
    max_samples: int,
    max_vocab: int,
    min_freq: int,
    label_scheme: str,
) -> Dict[str, Any]:
    root = Path(root_path)
    label_scheme = (label_scheme or "binary").strip().lower()
    if label_scheme not in {"binary", "ternary"}:
        raise ValueError("label_scheme must be 'binary' or 'ternary'.")

    if dataset_path_input.strip():
        raw_dataset_path = Path(dataset_path_input.strip())
        dataset_path = raw_dataset_path if raw_dataset_path.is_absolute() else (root / raw_dataset_path)
    else:
        dataset_path = sentiment_dataset_path_from_root(root)
    dataset_resolved = dataset_path.resolve()
    dataset_stat = dataset_resolved.stat()
    model_file = _sentiment_demo_model_file(
        root=root,
        dataset_resolved=dataset_resolved,
        dataset_size=int(dataset_stat.st_size),
        dataset_mtime_ns=int(dataset_stat.st_mtime_ns),
        max_samples=int(max_samples),
        max_vocab=int(max_vocab),
        min_freq=int(min_freq),
        label_scheme=label_scheme,
    )
    payload = _load_pickled_sentiment_assets(model_file)
    if payload is not None:
        payload.setdefault("label_scheme", label_scheme)
        if "class_values" not in payload:
            if payload["label_scheme"] == "binary":
                payload["class_values"] = [0, 1]
            elif hasattr(payload.get("model"), "classes_"):
                payload["class_values"] = [int(v) for v in payload["model"].classes_.tolist()]
        payload["loaded_from_disk"] = 1.0
        payload["model_path"] = str(model_file)
        return payload

    if label_scheme == "binary":
        legacy_key = (
            "sentiment_demo_v1|"
            f"{dataset_resolved}|{int(dataset_stat.st_size)}|{int(dataset_stat.st_mtime_ns)}|"
            f"{int(max_samples)}|{int(max_vocab)}|{int(min_freq)}"
        )
        legacy_hash = hashlib.sha256(legacy_key.encode("utf-8")).hexdigest()[:24]
        legacy_file = root / ".cache" / "sentiment_demo" / f"{legacy_hash}.pkl"
        payload = _load_pickled_sentiment_assets(legacy_file)
        if payload is not None:
            payload.setdefault("label_scheme", "binary")
            payload.setdefault("class_values", [0, 1])
            payload["loaded_from_disk"] = 1.0
            payload["model_path"] = str(model_file)
            try:
                model_file.parent.mkdir(parents=True, exist_ok=True)
                with model_file.open("wb") as f:
                    pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
            except (OSError, pickle.PickleError):
                pass
            return payload

    texts, labels, data_source = load_sentiment_dataset(dataset_path, label_scheme=label_scheme)
    if len(texts) < 200:
        raise ValueError(f"Sentiment dataset too small or invalid: {data_source}")

    y = np.array(labels, dtype=np.int64)
    class_values_all = np.unique(y)
    if len(class_values_all) < 2:
        raise ValueError(
            "Dataset must contain at least two classes after applying the selected label config."
        )
    subset_idx = _stratified_take_indices(y, max_samples)
    texts_sub = [texts[int(i)] for i in subset_idx.tolist()]
    y_sub = y[subset_idx]

    train_size = max(int(0.8 * len(y_sub)), 1)
    x_train_text = texts_sub[:train_size]
    x_test_text = texts_sub[train_size:]
    y_train = y_sub[:train_size]
    y_test = y_sub[train_size:]

    model: Any
    pred: np.ndarray
    vocab = build_vocab_for_classification(x_train_text, min_freq=min_freq, max_vocab=max_vocab)
    vectorizer: Any | None = None
    uses_sklearn_sparse = 0.0

    if SKLEARN_AVAILABLE:
        try:
            from scipy import sparse
            from sklearn.feature_extraction.text import CountVectorizer
            from sklearn.linear_model import LogisticRegression

            vectorizer = CountVectorizer(
                lowercase=True,
                token_pattern=r"(?u)\b\w+\b",
                min_df=max(int(min_freq), 1),
                max_features=max(int(max_vocab), 1),
            )
            xtr_counts_sparse = vectorizer.fit_transform(x_train_text)
            xte_counts_sparse = vectorizer.transform(x_test_text)
            xtr_lex_sparse = sparse.csr_matrix(sentiment_lexicon_features(x_train_text))
            xte_lex_sparse = sparse.csr_matrix(sentiment_lexicon_features(x_test_text))
            xtr_sparse = sparse.hstack([xtr_counts_sparse, xtr_lex_sparse], format="csr")
            xte_sparse = sparse.hstack([xte_counts_sparse, xte_lex_sparse], format="csr")

            model = LogisticRegression(
                C=1.0,
                solver="liblinear",
                max_iter=3000,
                random_state=42,
            ).fit(xtr_sparse, y_train)
            pred = model.predict(xte_sparse).astype(np.int64)
            uses_sklearn_sparse = 1.0
            vocab = dict(vectorizer.vocabulary_)
        except Exception:
            vectorizer = None

    if vectorizer is None:
        classes_in_train = np.unique(y_train)
        if label_scheme != "binary" or set(classes_in_train.tolist()) != {0, 1}:
            raise RuntimeError(
                "The selected label config requires scikit-learn in this demo. "
                "Install scikit-learn or use the binary (0/1) label config."
            )
        xtr_counts = vectorize_bow_counts(x_train_text, vocab)
        xte_counts = vectorize_bow_counts(x_test_text, vocab)
        xtr_lex = sentiment_lexicon_features(x_train_text)
        xte_lex = sentiment_lexicon_features(x_test_text)
        xtr = np.hstack([xtr_counts, xtr_lex])
        xte = np.hstack([xte_counts, xte_lex])
        model = LogisticBinary(lr=0.2, epochs=35, reg_type="l2", reg_strength=1e-4).fit(xtr, y_train)
        pred = model.predict(xte)

    test_accuracy = float((pred == y_test).mean()) if len(y_test) else 0.0
    class_values_sorted = sorted({int(v) for v in np.unique(y_sub).tolist()})

    assets = {
        "model": model,
        "vocab": vocab,
        "vectorizer": vectorizer,
        "uses_sklearn_sparse": uses_sklearn_sparse,
        "dataset_path": str(dataset_path),
        "label_scheme": label_scheme,
        "class_values": class_values_sorted,
        "num_samples": len(texts_sub),
        "train_samples": len(y_train),
        "test_samples": len(y_test),
        "test_accuracy": test_accuracy,
        "loaded_from_disk": 0.0,
        "model_path": str(model_file),
    }
    try:
        model_file.parent.mkdir(parents=True, exist_ok=True)
        with model_file.open("wb") as f:
            pickle.dump(assets, f, protocol=pickle.HIGHEST_PROTOCOL)
    except (OSError, pickle.PickleError):
        pass
    return assets


def _predict_sentiment(text: str, sentiment_assets: Mapping[str, Any]) -> Dict[str, Any]:
    model = sentiment_assets["model"]
    label_scheme = str(sentiment_assets.get("label_scheme", "binary"))
    if float(sentiment_assets.get("uses_sklearn_sparse", 0.0)) >= 0.5:
        from scipy import sparse

        vectorizer = sentiment_assets["vectorizer"]
        x_count = vectorizer.transform([text])
        x_lex = sparse.csr_matrix(sentiment_lexicon_features([text]))
        x = sparse.hstack([x_count, x_lex], format="csr")
    else:
        vocab = sentiment_assets["vocab"]
        x_count = vectorize_bow_counts([text], vocab)
        x_lex = sentiment_lexicon_features([text])
        x = np.hstack([x_count, x_lex])

    raw_proba = model.predict_proba(x)
    class_probabilities: Dict[str, float] = {}
    pred_class = 0
    if isinstance(raw_proba, np.ndarray) and raw_proba.ndim == 2:
        if hasattr(model, "classes_"):
            model_classes = [int(v) for v in model.classes_.tolist()]
        else:
            model_classes = [int(v) for v in sentiment_assets.get("class_values", [0, 1])]
            if len(model_classes) != raw_proba.shape[1]:
                model_classes = list(range(raw_proba.shape[1]))
        row = raw_proba[0]
        for idx, cls in enumerate(model_classes):
            class_probabilities[str(int(cls))] = float(row[idx])
        pred_class = int(model_classes[int(np.argmax(row))])
    else:
        positive_prob = float(raw_proba[0])
        class_probabilities["0"] = float(1.0 - positive_prob)
        class_probabilities["1"] = positive_prob
        pred_class = 1 if positive_prob >= 0.5 else 0

    named_probabilities = {
        _sentiment_label_name(int(cls), label_scheme): prob
        for cls, prob in class_probabilities.items()
    }
    positive_probability = class_probabilities.get("1")
    return {
        "label": _sentiment_label_name(pred_class, label_scheme),
        "label_id": int(pred_class),
        "positive_probability": (
            float(positive_probability) if positive_probability is not None else None
        ),
        "class_probabilities": class_probabilities,
        "named_probabilities": named_probabilities,
    }


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
    return _path_signature(path)


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


def _task_result_txt_path(root_path: Path, task_number: int) -> Path:
    task_id = int(task_number)
    return root_path / f"Task{task_id}" / f"task{task_id}_results.txt"


def _persist_task_result_txt(root_path: Path, task_number: int, metrics: Mapping[str, Any]) -> Path:
    title = f"Task {int(task_number)}"
    out_path = _task_result_txt_path(root_path, int(task_number))
    return save_metrics_text(out_path, title, metrics)


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
    if "task_result_files" not in st.session_state:
        st.session_state["task_result_files"] = {}

    col1, col2, col3, col4, col5 = st.columns(5)
    if col1.button("Run Task 1", use_container_width=True):
        with st.spinner("Running Task 1..."):
            metrics = run_task1(
                news_path=news_path,
                max_sentences=max_sentences,
                min_freq=min_freq,
            )
            st.session_state["task_results"]["Task 1"] = metrics
            saved_path = _persist_task_result_txt(root_path, 1, metrics)
            st.session_state["task_result_files"]["Task 1"] = str(saved_path)
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
            saved_path = _persist_task_result_txt(root_path, 2, metrics)
            st.session_state["task_result_files"]["Task 2"] = str(saved_path)
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
            saved_task1 = _persist_task_result_txt(root_path, 1, task1)
            saved_task2 = _persist_task_result_txt(root_path, 2, task2)
            st.session_state["task_result_files"]["Task 1"] = str(saved_task1)
            st.session_state["task_result_files"]["Task 2"] = str(saved_task2)

    if st.button("Clear Results"):
        st.session_state["task_results"] = {}
        st.session_state["task_result_files"] = {}

    if not st.session_state["task_results"]:
        st.info("Run any task to see metrics.")
        return

    for task_name, metrics in st.session_state["task_results"].items():
        st.markdown(f"**{task_name}**")
        st.dataframe(_metrics_frame(metrics), use_container_width=True, hide_index=True)
        saved_file = st.session_state.get("task_result_files", {}).get(task_name)
        if saved_file:
            st.caption(f"Saved metrics: `{saved_file}`")


def _run_lm_demo_panel(news_path: Path, max_sentences: int, min_freq: int) -> None:
    st.subheader("Unigram / Bigram / Trigram Demo")
    st.caption(
        "Type a sentence and inspect perplexity and token-level probabilities. "
        "The default example is precomputed and reused."
    )
    smoothing_choice = st.selectbox(
        "Advanced trigram smoothing",
        options=list(LM_SMOOTHING_TO_KEY.keys()),
        index=1,
        key="lm_demo_smoothing_choice",
    )
    sentence = st.text_area(
        "Input sentence",
        value=LM_DEFAULT_EXAMPLE_SENTENCE,
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
        normalized_sentence = sentence.strip()
        example_sentence = str(lm_assets.get("example_sentence", LM_DEFAULT_EXAMPLE_SENTENCE))
        if normalized_sentence == example_sentence:
            ppl = dict(lm_assets["example_perplexities"])
            probs = pd.DataFrame(lm_assets["example_token_rows"])
            source = "disk cache" if float(lm_assets.get("loaded_from_disk", 0.0)) >= 0.5 else "memory cache"
            st.caption(f"Used precomputed example output ({source}).")
        else:
            with st.spinner("Scoring sentence..."):
                ppl = _safe_perplexity_text(normalized_sentence, lm_assets)
                probs = _token_probability_rows(normalized_sentence, lm_assets)

        selected_trigram_key = LM_SMOOTHING_TO_KEY[smoothing_choice]
        use_laplace_for_lower_orders = smoothing_choice == "Laplace"
        unigram_key = "ppl_unigram_laplace" if use_laplace_for_lower_orders else "ppl_unigram_mle"
        bigram_key = "ppl_bigram_laplace" if use_laplace_for_lower_orders else "ppl_bigram_mle"
        unigram_label = "Laplace" if use_laplace_for_lower_orders else "MLE"
        bigram_label = "Laplace" if use_laplace_for_lower_orders else "MLE"
        col1, col2, col3, col4 = st.columns(4)
        col1.metric(f"Unigram PPL ({unigram_label})", _format_value(ppl[unigram_key]))
        col2.metric(f"Bigram PPL ({bigram_label})", _format_value(ppl[bigram_key]))
        col3.metric(f"Trigram PPL ({smoothing_choice})", _format_value(ppl[selected_trigram_key]))
        col4.metric("Trigram PPL (MLE)", _format_value(ppl["ppl_trigram_mle"]))

        smooth_rows = [
            {"smoothing": name, "trigram_ppl": _format_value(ppl[key])}
            for name, key in LM_SMOOTHING_TO_KEY.items()
        ]
        st.markdown("**Trigram Smoothing Comparison**")
        st.dataframe(pd.DataFrame(smooth_rows), use_container_width=True, hide_index=True)
        st.markdown("**Token-level Probabilities (MLE)**")
        st.dataframe(probs, use_container_width=True, hide_index=True)


def _run_sentiment_demo_panel(root_path: Path) -> None:
    st.subheader("Sentiment Demo")
    st.caption("Simple interactive predictor trained from your sentiment dataset.")

    col1, col2, col3, col4 = st.columns(4)
    max_samples = col1.number_input(
        "Demo max samples", min_value=0, value=0, step=500
    )
    max_vocab = col2.number_input(
        "Demo max vocab", min_value=1000, max_value=50000, value=25000, step=500
    )
    min_freq = col3.number_input("Min token freq", min_value=1, max_value=10, value=2, step=1)
    label_choice = col4.selectbox(
        "Label config",
        options=[
            "Binary (0=negative, 1=positive)",
            "Ternary (-1=negative, 0=neutral, 1=positive)",
        ],
        index=0,
        key="sentiment_label_config_choice",
    )
    label_scheme = "ternary" if label_choice.startswith("Ternary") else "binary"

    dataset_auto = sentiment_dataset_path_from_root(root_path)
    dataset_v1 = root_path / "sentiment_dataset" / "dataset_v1.csv"
    dataset_csv = root_path / "sentiment_dataset" / "dataset.csv"
    dataset_choice = st.selectbox(
        "Dataset source",
        options=["Auto (prefer dataset_v1.csv)", "dataset_v1.csv", "dataset.csv", "Custom path"],
        index=0,
        key="sentiment_dataset_choice",
    )
    if dataset_choice == "dataset_v1.csv":
        dataset_input = str(dataset_v1)
    elif dataset_choice == "dataset.csv":
        dataset_input = str(dataset_csv)
    elif dataset_choice == "Custom path":
        dataset_input = st.text_input("Custom dataset path", value=str(dataset_auto))
    else:
        dataset_input = ""

    sentiment_text = st.text_area(
        "Text to classify",
        value="Bu film cox gozel idi!",
        height=90,
        key="sentiment_demo_text",
    )

    settings_key = (
        str(root_path),
        dataset_input.strip(),
        label_scheme,
        int(max_samples),
        int(max_vocab),
        int(min_freq),
    )
    if st.button("Predict Sentiment", key="sent_demo_btn"):
        assets = _load_once_per_settings(
            state_slot="sentiment_assets",
            settings_key=settings_key,
            spinner_text="Loading sentiment demo model (trains once if missing)...",
            loader=lambda: _train_sentiment_demo(
                str(root_path),
                dataset_input.strip(),
                int(max_samples),
                int(max_vocab),
                int(min_freq),
                label_scheme,
            ),
        )
        prediction = _predict_sentiment(sentiment_text, assets)
        st.metric("Prediction", f"{prediction['label']} ({prediction['label_id']})")
        if prediction.get("positive_probability") is not None:
            st.metric("Positive Probability", _format_value(prediction["positive_probability"]))
        prob_rows = [
            {"label": label_name, "probability": prob}
            for label_name, prob in prediction.get("named_probabilities", {}).items()
        ]
        if prob_rows:
            prob_df = pd.DataFrame(prob_rows).sort_values("probability", ascending=False)
            st.dataframe(prob_df, use_container_width=True, hide_index=True)
        source_label = "loaded" if float(assets.get("loaded_from_disk", 0.0)) >= 0.5 else "trained"
        st.caption(
            f"Demo model ({source_label}) uses {assets['num_samples']} samples from {assets['dataset_path']} "
            f"with {assets.get('label_scheme', 'binary')} labels "
            f"(test accuracy ~ {assets['test_accuracy']:.4f}). "
            f"Saved at: {assets.get('model_path', 'n/a')}"
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
