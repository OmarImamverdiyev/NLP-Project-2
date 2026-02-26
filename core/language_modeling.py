from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import numpy as np

from core.paths import SEED
from core.text_utils import sentence_split, tokenize_words


np.random.seed(SEED)


def load_news_sentences(path: Path, max_sentences: int | None = None) -> List[List[str]]:
    sentences: List[List[str]] = []
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            for sent in sentence_split(line):
                toks = tokenize_words(sent)
                if toks:
                    sentences.append(toks)
                    if max_sentences is not None and len(sentences) >= max_sentences:
                        return sentences
    return sentences


def train_dev_test_split(
    seqs: Sequence[List[str]],
    train_ratio: float = 0.8,
    dev_ratio: float = 0.1,
) -> Tuple[List[List[str]], List[List[str]], List[List[str]]]:
    idx = np.arange(len(seqs))
    np.random.shuffle(idx)
    n = len(seqs)
    n_train = int(n * train_ratio)
    n_dev = int(n * dev_ratio)
    train = [seqs[i] for i in idx[:n_train]]
    dev = [seqs[i] for i in idx[n_train : n_train + n_dev]]
    test = [seqs[i] for i in idx[n_train + n_dev :]]
    return train, dev, test


def replace_rare_with_unk(
    train_sents: Sequence[List[str]],
    other_sents: Sequence[List[str]],
    min_freq: int = 2,
) -> Tuple[List[List[str]], List[List[str]], set[str]]:
    freq = Counter()
    for sent in train_sents:
        freq.update(sent)

    vocab = {w for w, c in freq.items() if c >= min_freq}
    vocab.add("<UNK>")

    def map_sent(sent: List[str]) -> List[str]:
        return [w if w in vocab else "<UNK>" for w in sent]

    mapped_train = [map_sent(s) for s in train_sents]
    mapped_other = [map_sent(s) for s in other_sents]
    return mapped_train, mapped_other, vocab


def add_markers(sent: List[str], n: int) -> List[str]:
    if n == 1:
        return sent + ["</s>"]
    return ["<s>"] * (n - 1) + sent + ["</s>"]


@dataclass
class NgramCounts:
    unigram: Counter
    bigram: Counter
    trigram: Counter
    unigram_ctx: Counter
    bigram_ctx: Counter
    total_unigrams: int
    vocab: set[str]
    followers_bigram_ctx: Dict[Tuple[str, str], int]
    followers_unigram_ctx: Dict[str, int]
    predecessors_word: Dict[str, int]
    total_bigram_types: int


def build_ngram_counts(train_sents: Sequence[List[str]], vocab: set[str]) -> NgramCounts:
    unigram = Counter()
    bigram = Counter()
    trigram = Counter()
    unigram_ctx = Counter()
    bigram_ctx = Counter()

    for sent in train_sents:
        s1 = add_markers(sent, 1)
        unigram.update(s1)

        s2 = add_markers(sent, 2)
        for i in range(1, len(s2)):
            h, w = s2[i - 1], s2[i]
            bigram[(h, w)] += 1
            unigram_ctx[h] += 1

        s3 = add_markers(sent, 3)
        for i in range(2, len(s3)):
            u, v, w = s3[i - 2], s3[i - 1], s3[i]
            trigram[(u, v, w)] += 1
            bigram_ctx[(u, v)] += 1

    followers_bigram_ctx = defaultdict(int)
    for (u, v, _w) in trigram.keys():
        followers_bigram_ctx[(u, v)] += 1

    followers_unigram_ctx = defaultdict(int)
    predecessors_word = defaultdict(int)
    for (u, v) in bigram.keys():
        followers_unigram_ctx[u] += 1
        predecessors_word[v] += 1

    return NgramCounts(
        unigram=unigram,
        bigram=bigram,
        trigram=trigram,
        unigram_ctx=unigram_ctx,
        bigram_ctx=bigram_ctx,
        total_unigrams=sum(unigram.values()),
        vocab=vocab,
        followers_bigram_ctx=dict(followers_bigram_ctx),
        followers_unigram_ctx=dict(followers_unigram_ctx),
        predecessors_word=dict(predecessors_word),
        total_bigram_types=len(bigram),
    )


def p_unigram_mle(w: str, c: NgramCounts) -> float:
    return c.unigram[w] / c.total_unigrams if c.total_unigrams else 0.0


def p_bigram_mle(h: str, w: str, c: NgramCounts) -> float:
    denom = c.unigram_ctx[h]
    if denom == 0:
        return 0.0
    return c.bigram[(h, w)] / denom


def p_trigram_mle(u: str, v: str, w: str, c: NgramCounts) -> float:
    denom = c.bigram_ctx[(u, v)]
    if denom == 0:
        return 0.0
    return c.trigram[(u, v, w)] / denom


def p_unigram_laplace(w: str, c: NgramCounts) -> float:
    v = len(c.vocab)
    return (c.unigram[w] + 1.0) / (c.total_unigrams + v)


def p_bigram_laplace(h: str, w: str, c: NgramCounts) -> float:
    v = len(c.vocab)
    return (c.bigram[(h, w)] + 1.0) / (c.unigram_ctx[h] + v)


def p_trigram_laplace(u: str, v: str, w: str, c: NgramCounts) -> float:
    vv = len(c.vocab)
    return (c.trigram[(u, v, w)] + 1.0) / (c.bigram_ctx[(u, v)] + vv)


def p_bigram_backoff(h: str, w: str, c: NgramCounts, d: float = 0.75) -> float:
    ch = c.unigram_ctx[h]
    if ch == 0:
        return p_unigram_mle(w, c)

    c_hw = c.bigram[(h, w)]
    n1_h = c.followers_unigram_ctx.get(h, 0)
    lambda_h = (d * n1_h) / ch
    first = max(c_hw - d, 0.0) / ch
    return first + lambda_h * p_unigram_mle(w, c)


def p_trigram_backoff(u: str, v: str, w: str, c: NgramCounts, d: float = 0.75) -> float:
    ch = c.bigram_ctx[(u, v)]
    if ch == 0:
        return p_bigram_backoff(v, w, c, d=d)

    c_uvw = c.trigram[(u, v, w)]
    n1_uv = c.followers_bigram_ctx.get((u, v), 0)
    lambda_uv = (d * n1_uv) / ch
    first = max(c_uvw - d, 0.0) / ch
    return first + lambda_uv * p_bigram_backoff(v, w, c, d=d)


def p_continuation_kn(w: str, c: NgramCounts) -> float:
    if c.total_bigram_types == 0:
        return 0.0
    return c.predecessors_word.get(w, 0) / c.total_bigram_types


def p_bigram_kn(h: str, w: str, c: NgramCounts, d: float = 0.75) -> float:
    ch = c.unigram_ctx[h]
    if ch == 0:
        return p_continuation_kn(w, c)

    c_hw = c.bigram[(h, w)]
    n1_h = c.followers_unigram_ctx.get(h, 0)
    lambda_h = (d * n1_h) / ch
    first = max(c_hw - d, 0.0) / ch
    return first + lambda_h * p_continuation_kn(w, c)


def p_trigram_kn(u: str, v: str, w: str, c: NgramCounts, d: float = 0.75) -> float:
    ch = c.bigram_ctx[(u, v)]
    if ch == 0:
        return p_bigram_kn(v, w, c, d=d)

    c_uvw = c.trigram[(u, v, w)]
    n1_uv = c.followers_bigram_ctx.get((u, v), 0)
    lambda_uv = (d * n1_uv) / ch
    first = max(c_uvw - d, 0.0) / ch
    return first + lambda_uv * p_bigram_kn(v, w, c, d=d)


def tune_linear_interpolation(
    dev_sents: Sequence[List[str]],
    c: NgramCounts,
    step: float = 0.1,
) -> Tuple[float, float, float]:
    best = (0.2, 0.3, 0.5)
    best_nll = float("inf")
    vals = [round(i * step, 10) for i in range(int(1 / step) + 1)]

    for l1 in vals:
        for l2 in vals:
            l3 = 1.0 - l1 - l2
            if l3 < 0:
                continue

            nll = 0.0
            n = 0
            for sent in dev_sents:
                s3 = add_markers(sent, 3)
                for i in range(2, len(s3)):
                    u, v, w = s3[i - 2], s3[i - 1], s3[i]
                    p = (
                        l1 * p_unigram_mle(w, c)
                        + l2 * p_bigram_mle(v, w, c)
                        + l3 * p_trigram_mle(u, v, w, c)
                    )
                    p = max(p, 1e-12)
                    nll -= math.log(p)
                    n += 1
            if n and nll < best_nll:
                best_nll = nll
                best = (l1, l2, l3)
    return best


def tune_bigram_interpolation(
    dev_sents: Sequence[List[str]],
    c: NgramCounts,
    step: float = 0.1,
) -> Tuple[float, float]:
    best = (0.3, 0.7)
    best_nll = float("inf")
    vals = [round(i * step, 10) for i in range(int(1 / step) + 1)]

    for l1 in vals:
        l2 = 1.0 - l1
        if l2 < 0:
            continue

        nll = 0.0
        n = 0
        for sent in dev_sents:
            s2 = add_markers(sent, 2)
            for i in range(1, len(s2)):
                h, w = s2[i - 1], s2[i]
                p = l1 * p_unigram_mle(w, c) + l2 * p_bigram_mle(h, w, c)
                p = max(p, 1e-12)
                nll -= math.log(p)
                n += 1

        if n and nll < best_nll:
            best_nll = nll
            best = (l1, l2)
    return best


def perplexity_unigram(
    sents: Sequence[List[str]],
    c: NgramCounts,
    mode: str = "mle",
) -> float:
    logprob = 0.0
    n = 0
    for sent in sents:
        s1 = add_markers(sent, 1)
        for w in s1:
            if mode == "mle":
                p = p_unigram_mle(w, c)
            elif mode == "laplace":
                p = p_unigram_laplace(w, c)
            else:
                raise ValueError(f"Unknown unigram mode: {mode}")
            if p <= 0:
                return float("inf")
            logprob += math.log(p)
            n += 1
    return math.exp(-logprob / max(n, 1))


def perplexity_bigram(
    sents: Sequence[List[str]],
    c: NgramCounts,
    mode: str = "mle",
    lambdas: Tuple[float, float] = (0.3, 0.7),
    d: float = 0.75,
) -> float:
    logprob = 0.0
    n = 0
    l1, l2 = lambdas
    for sent in sents:
        s2 = add_markers(sent, 2)
        for i in range(1, len(s2)):
            h, w = s2[i - 1], s2[i]
            if mode == "mle":
                p = p_bigram_mle(h, w, c)
            elif mode == "laplace":
                p = p_bigram_laplace(h, w, c)
            elif mode == "interpolation":
                p = l1 * p_unigram_mle(w, c) + l2 * p_bigram_mle(h, w, c)
            elif mode == "backoff":
                p = p_bigram_backoff(h, w, c, d=d)
            elif mode == "kneser_ney":
                p = p_bigram_kn(h, w, c, d=d)
            else:
                raise ValueError(f"Unknown bigram mode: {mode}")
            if p <= 0:
                return float("inf")
            logprob += math.log(p)
            n += 1
    return math.exp(-logprob / max(n, 1))


def perplexity_trigram(
    sents: Sequence[List[str]],
    c: NgramCounts,
    mode: str = "mle",
    lambdas: Tuple[float, float, float] = (0.2, 0.3, 0.5),
    d: float = 0.75,
) -> float:
    logprob = 0.0
    n = 0
    l1, l2, l3 = lambdas
    for sent in sents:
        s3 = add_markers(sent, 3)
        for i in range(2, len(s3)):
            u, v, w = s3[i - 2], s3[i - 1], s3[i]
            if mode == "mle":
                p = p_trigram_mle(u, v, w, c)
            elif mode == "laplace":
                p = p_trigram_laplace(u, v, w, c)
            elif mode == "interpolation":
                p = (
                    l1 * p_unigram_mle(w, c)
                    + l2 * p_bigram_mle(v, w, c)
                    + l3 * p_trigram_mle(u, v, w, c)
                )
            elif mode == "backoff":
                p = p_trigram_backoff(u, v, w, c, d=d)
            elif mode == "kneser_ney":
                p = p_trigram_kn(u, v, w, c, d=d)
            else:
                raise ValueError(f"Unknown trigram mode: {mode}")

            if p <= 0:
                return float("inf")
            logprob += math.log(max(p, 1e-12))
            n += 1
    return math.exp(-logprob / max(n, 1))


def _p_bigram_by_mode(
    h: str,
    w: str,
    c: NgramCounts,
    mode: str,
    interp_lambdas: Tuple[float, float] = (0.3, 0.7),
    d: float = 0.75,
) -> float:
    l1, l2 = interp_lambdas
    if mode == "mle":
        return p_bigram_mle(h, w, c)
    if mode == "laplace":
        return p_bigram_laplace(h, w, c)
    if mode == "interpolation":
        return l1 * p_unigram_mle(w, c) + l2 * p_bigram_mle(h, w, c)
    if mode == "backoff":
        return p_bigram_backoff(h, w, c, d=d)
    if mode == "kneser_ney":
        return p_bigram_kn(h, w, c, d=d)
    raise ValueError(f"Unknown bigram mode: {mode}")


def _p_trigram_by_mode(
    u: str,
    v: str,
    w: str,
    c: NgramCounts,
    mode: str,
    interp_lambdas: Tuple[float, float, float] = (0.2, 0.3, 0.5),
    d: float = 0.75,
) -> float:
    l1, l2, l3 = interp_lambdas
    if mode == "mle":
        return p_trigram_mle(u, v, w, c)
    if mode == "laplace":
        return p_trigram_laplace(u, v, w, c)
    if mode == "interpolation":
        return (
            l1 * p_unigram_mle(w, c)
            + l2 * p_bigram_mle(v, w, c)
            + l3 * p_trigram_mle(u, v, w, c)
        )
    if mode == "backoff":
        return p_trigram_backoff(u, v, w, c, d=d)
    if mode == "kneser_ney":
        return p_trigram_kn(u, v, w, c, d=d)
    raise ValueError(f"Unknown trigram mode: {mode}")


def save_smoothed_ngram_tables(
    c: NgramCounts,
    out_dir: Path,
    bigram_methods: Sequence[str] = ("laplace", "interpolation", "backoff", "kneser_ney"),
    trigram_methods: Sequence[str] = ("laplace", "interpolation", "backoff", "kneser_ney"),
    bigram_interp_lambdas: Tuple[float, float] = (0.3, 0.7),
    trigram_interp_lambdas: Tuple[float, float, float] = (0.2, 0.3, 0.5),
    d: float = 0.75,
) -> Dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    ranked_bigrams = sorted(
        c.bigram.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1]),
    )
    ranked_trigrams = sorted(
        c.trigram.items(),
        key=lambda item: (-item[1], item[0][0], item[0][1], item[0][2]),
    )
    saved: Dict[str, Path] = {}
    for mode in bigram_methods:
        path = out_dir / f"bigrams_{mode}.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write("order: bigram\n")
            f.write(f"method: {mode}\n")
            if mode == "interpolation":
                f.write(
                    "interp_lambdas: "
                    f"unigram={bigram_interp_lambdas[0]:.6f},"
                    f"bigram={bigram_interp_lambdas[1]:.6f}\n"
                )
            f.write("history\tword\tcount\tprobability\n")
            for (h, w), count in ranked_bigrams:
                p = _p_bigram_by_mode(
                    h,
                    w,
                    c,
                    mode=mode,
                    interp_lambdas=bigram_interp_lambdas,
                    d=d,
                )
                f.write(f"{h}\t{w}\t{count}\t{p:.12f}\n")
        saved[f"bigram_{mode}"] = path

    for mode in trigram_methods:
        path = out_dir / f"trigrams_{mode}.txt"
        with path.open("w", encoding="utf-8") as f:
            f.write("order: trigram\n")
            f.write(f"method: {mode}\n")
            if mode == "interpolation":
                f.write(
                    "interp_lambdas: "
                    f"unigram={trigram_interp_lambdas[0]:.6f},"
                    f"bigram={trigram_interp_lambdas[1]:.6f},"
                    f"trigram={trigram_interp_lambdas[2]:.6f}\n"
                )
            f.write("history1\thistory2\tword\tcount\tprobability\n")
            for (u, v, w), count in ranked_trigrams:
                p = _p_trigram_by_mode(
                    u,
                    v,
                    w,
                    c,
                    mode=mode,
                    interp_lambdas=trigram_interp_lambdas,
                    d=d,
                )
                f.write(f"{u}\t{v}\t{w}\t{count}\t{p:.12f}\n")
        saved[f"trigram_{mode}"] = path
    return saved


def _prepare_lm_task_data(
    news_path: Path,
    max_sentences: int = 120000,
    min_freq: int = 2,
) -> Tuple[List[List[str]], set[str], List[List[str]], List[List[str]], NgramCounts]:
    sentences = load_news_sentences(news_path, max_sentences=max_sentences)
    train, dev, test = train_dev_test_split(sentences, train_ratio=0.8, dev_ratio=0.1)

    train_mapped, dev_mapped, vocab = replace_rare_with_unk(train, dev, min_freq=min_freq)
    _, test_mapped, _ = replace_rare_with_unk(train, test, min_freq=min_freq)
    counts = build_ngram_counts(train_mapped, vocab)
    return sentences, vocab, dev_mapped, test_mapped, counts


def _task1_metrics(
    sentences: Sequence[List[str]],
    vocab: set[str],
    test_mapped: Sequence[List[str]],
    counts: NgramCounts,
) -> Dict[str, float]:
    ppl_uni_mle = perplexity_unigram(test_mapped, counts, mode="mle")
    ppl_bi_mle = perplexity_bigram(test_mapped, counts, mode="mle")
    ppl_tri_mle = perplexity_trigram(test_mapped, counts, mode="mle")
    return {
        "num_sentences": float(len(sentences)),
        "vocab_size": float(len(vocab)),
        "ppl_unigram_mle": ppl_uni_mle,
        "ppl_bigram_mle": ppl_bi_mle,
        "ppl_trigram_mle": ppl_tri_mle,
    }


def _task2_metrics(
    sentences: Sequence[List[str]],
    vocab: set[str],
    dev_mapped: Sequence[List[str]],
    test_mapped: Sequence[List[str]],
    counts: NgramCounts,
) -> Dict[str, float]:
    bigram_interp_lambdas = tune_bigram_interpolation(dev_mapped, counts, step=0.1)
    trigram_interp_lambdas = tune_linear_interpolation(dev_mapped, counts, step=0.1)
    ppl_bigram_lap = perplexity_bigram(test_mapped, counts, mode="laplace")
    ppl_bigram_interp = perplexity_bigram(
        test_mapped,
        counts,
        mode="interpolation",
        lambdas=bigram_interp_lambdas,
    )
    ppl_bigram_backoff = perplexity_bigram(test_mapped, counts, mode="backoff", d=0.75)
    ppl_bigram_kn = perplexity_bigram(test_mapped, counts, mode="kneser_ney", d=0.75)
    ppl_lap = perplexity_trigram(test_mapped, counts, mode="laplace")
    ppl_interp = perplexity_trigram(
        test_mapped, counts, mode="interpolation", lambdas=trigram_interp_lambdas
    )
    ppl_backoff = perplexity_trigram(test_mapped, counts, mode="backoff", d=0.75)
    ppl_kn = perplexity_trigram(test_mapped, counts, mode="kneser_ney", d=0.75)
    return {
        "num_sentences": float(len(sentences)),
        "vocab_size": float(len(vocab)),
        "bigram_interp_lambda1": bigram_interp_lambdas[0],
        "bigram_interp_lambda2": bigram_interp_lambdas[1],
        "ppl_bigram_laplace": ppl_bigram_lap,
        "ppl_bigram_interpolation": ppl_bigram_interp,
        "ppl_bigram_backoff": ppl_bigram_backoff,
        "ppl_bigram_kneser_ney": ppl_bigram_kn,
        "interp_lambda1": trigram_interp_lambdas[0],
        "interp_lambda2": trigram_interp_lambdas[1],
        "interp_lambda3": trigram_interp_lambdas[2],
        "ppl_trigram_laplace": ppl_lap,
        "ppl_trigram_interpolation": ppl_interp,
        "ppl_trigram_backoff": ppl_backoff,
        "ppl_trigram_kneser_ney": ppl_kn,
    }


def run_task1(
    news_path: Path,
    max_sentences: int = 120000,
    min_freq: int = 2,
) -> Dict[str, float]:
    sentences, vocab, _dev_mapped, test_mapped, counts = _prepare_lm_task_data(
        news_path=news_path,
        max_sentences=max_sentences,
        min_freq=min_freq,
    )
    return _task1_metrics(sentences, vocab, test_mapped, counts)


def run_task2(
    news_path: Path,
    max_sentences: int = 120000,
    min_freq: int = 2,
    txt_dir: Path | None = None,
) -> Dict[str, float]:
    sentences, vocab, dev_mapped, test_mapped, counts = _prepare_lm_task_data(
        news_path=news_path,
        max_sentences=max_sentences,
        min_freq=min_freq,
    )
    metrics = _task2_metrics(sentences, vocab, dev_mapped, test_mapped, counts)
    if txt_dir is not None:
        save_smoothed_ngram_tables(
            counts,
            txt_dir,
            bigram_interp_lambdas=(
                metrics["bigram_interp_lambda1"],
                metrics["bigram_interp_lambda2"],
            ),
            trigram_interp_lambdas=(
                metrics["interp_lambda1"],
                metrics["interp_lambda2"],
                metrics["interp_lambda3"],
            ),
            d=0.75,
        )
    return metrics


def run_task1_task2(
    news_path: Path,
    max_sentences: int = 120000,
    min_freq: int = 2,
) -> Dict[str, float]:
    sentences, vocab, dev_mapped, test_mapped, counts = _prepare_lm_task_data(
        news_path=news_path,
        max_sentences=max_sentences,
        min_freq=min_freq,
    )
    t1 = _task1_metrics(sentences, vocab, test_mapped, counts)
    t2 = _task2_metrics(sentences, vocab, dev_mapped, test_mapped, counts)
    return {**t1, **t2}
