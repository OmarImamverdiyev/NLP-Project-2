#!/usr/bin/env python3
"""
NLP Project 2 orchestrator.

Task implementations live in task-specific modules:
  - Task 1/2: core.language_modeling
  - Task 3:   core.sentiment_task
  - Task 4:   core.sentence_boundary_task_v2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict

from core.language_modeling import (
    run_task1 as _run_task1,
    run_task1_task2 as _run_task1_task2,
    run_task2 as _run_task2,
)
from core.paths import NEWS_CORPUS_PATH, ROOT
from core.reporting import print_section
from core.sentiment_task import run_task3 as _run_task3
from core.sentence_boundary_task_v2 import run_task4 as _run_task4


def run_task1(
    news_path: Path,
    max_sentences: int = 120000,
    min_freq: int = 2,
) -> Dict[str, float]:
    return _run_task1(news_path=news_path, max_sentences=max_sentences, min_freq=min_freq)


def run_task2(
    news_path: Path,
    max_sentences: int = 120000,
    min_freq: int = 2,
) -> Dict[str, float]:
    return _run_task2(news_path=news_path, max_sentences=max_sentences, min_freq=min_freq)


def run_task1_task2(
    news_path: Path,
    max_sentences: int = 120000,
    min_freq: int = 2,
) -> Dict[str, float]:
    return _run_task1_task2(news_path=news_path, max_sentences=max_sentences, min_freq=min_freq)


def run_task3(root: Path) -> Dict[str, float]:
    return _run_task3(root)


def run_task4(
    news_path: Path | None = None,
    max_docs: int | None = 30000,
    max_examples: int | None = 60000,
    max_vocab_tokens: int = 6000,
    dataset_path: Path | None = None,
) -> Dict[str, float | str]:
    _ = news_path, max_docs, max_vocab_tokens  # kept for backward-compatible call sites
    return _run_task4(
        dataset_path=dataset_path,
        max_examples=max_examples,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run NLP Project 2 tasks")
    parser.add_argument("--max-sentences", type=int, default=120000)
    parser.add_argument("--min-freq", type=int, default=2)
    parser.add_argument("--skip-task4", action="store_true")
    parser.add_argument("--task4-dataset", type=Path, default=ROOT / "dot_labeled_data.csv")
    parser.add_argument("--max-docs", type=int, default=30000)
    parser.add_argument("--max-examples", type=int, default=60000)
    parser.add_argument("--max-vocab-tokens", type=int, default=6000)
    args = parser.parse_args()

    if not NEWS_CORPUS_PATH.exists():
        raise FileNotFoundError(f"News corpus not found: {NEWS_CORPUS_PATH}")

    t12 = run_task1_task2(
        news_path=NEWS_CORPUS_PATH,
        max_sentences=args.max_sentences,
        min_freq=args.min_freq,
    )
    print_section("Task 1 + Task 2", t12)

    t3 = run_task3(ROOT)
    print_section("Task 3", t3)

    if not args.skip_task4:
        t4 = run_task4(
            dataset_path=args.task4_dataset,
            max_docs=args.max_docs,
            max_examples=args.max_examples,
            max_vocab_tokens=args.max_vocab_tokens,
        )
        print_section("Task 4", t4)

    bigram_smooth_keys = [
        "ppl_bigram_laplace",
        "ppl_bigram_interpolation",
        "ppl_bigram_backoff",
        "ppl_bigram_kneser_ney",
    ]
    trigram_smooth_keys = [
        "ppl_trigram_laplace",
        "ppl_trigram_interpolation",
        "ppl_trigram_backoff",
        "ppl_trigram_kneser_ney",
    ]
    vals_bigram = {k: t12[k] for k in bigram_smooth_keys}
    vals_trigram = {k: t12[k] for k in trigram_smooth_keys}
    vals_overall = {k: t12[k] for k in bigram_smooth_keys + trigram_smooth_keys}
    best_bigram = min(vals_bigram.items(), key=lambda kv: kv[1])[0]
    best_trigram = min(vals_trigram.items(), key=lambda kv: kv[1])[0]
    best_overall = min(vals_overall.items(), key=lambda kv: kv[1])[0]
    print(f"\nBest bigram smoothing by perplexity: {best_bigram}")
    print(f"Best trigram smoothing by perplexity: {best_trigram}")
    print(f"Best overall smoothing by perplexity: {best_overall}")


if __name__ == "__main__":
    main()

